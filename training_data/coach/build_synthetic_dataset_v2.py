"""Generate a quote-faithful synthetic Coach ChatML dataset.

This script mirrors the caregiver synthetic-data workflow while keeping Coach
responses very close to transcript quotes.

Design goals
------------
- Parse transcript markdown for both 1st-skills (Anne) and 2nd-skills (Maya)
  coaching examples.
- Preserve Coach language by reusing exact feedback paragraphs and only doing
  light recombination within the same (parent, phase) slice.
- Add an explicit system prompt + phase insert to every ChatML row so the
  model sees the same phase-conditioned structure used during evaluation.
- Balance rows across both caregivers for each Coach phase.
- Emit:
    1) full dataset JSONL
    2) deterministic train/eval splits
    3) stats JSON
    4) H6 fixture JSONL for notebook-driven scoring

Run
---
python training_data/coach/build_synthetic_dataset_v2.py

Outputs (default)
-----------------
training_data/coach/coach_synthetic_dataset_v2.jsonl
training_data/coach/coach_synthetic_dataset_v2.train.jsonl
training_data/coach/coach_synthetic_dataset_v2.eval.jsonl
training_data/coach/coach_synthetic_dataset_v2.stats.json
training_data/coach/coach_h6_fixtures_v2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
TRANSCRIPTS_DIR = REPO_ROOT / "training_data" / "transcripts"
COACH_SYSTEM_PROMPT_PATH = HERE / "coach-system-prompt.md"
GRADING_LOGIC_JSONL = HERE / "train_grading_logic.jsonl"

PARENT_IDS = ("anne_palmer", "maya_pena")
COACH_PHASES = ("COUNSEL", "LISTEN", "EMPATHIZE", "ANSWER_RECOMMEND", "SUMMATIVE")

PHASE_SYSTEM_INSERTS: Dict[str, str] = {
    "COUNSEL": (
        "Evaluate only the COUNSEL phase. Focus on whether the clinician includes "
        "the key recommendation elements: age timing, cancer prevention, safety, "
        "recommendation today, and second-dose return plan."
    ),
    "LISTEN": (
        "Evaluate only the LISTEN phase. Focus on whether the clinician uses an "
        "Explore or Restate listening behavior before moving into explanation."
    ),
    "EMPATHIZE": (
        "Evaluate only the EMPATHIZE phase. Focus on acknowledge/normalize/validate "
        "language that is responsive to the caregiver concern before factual answering."
    ),
    "ANSWER_RECOMMEND": (
        "Evaluate only ANSWER + RECOMMEND behavior. Focus on concern-specific, "
        "evidence-aligned answering followed by a clear recommendation statement."
    ),
    "SUMMATIVE": (
        "Generate end-of-session summative coaching feedback. Keep supportive tone, "
        "highlight strengths, and provide forward-looking improvement guidance."
    ),
    "GRADING_LOGIC": (
        "Explain internal coaching policy/rubric behavior clearly without exposing a "
        "numeric score to the learner-facing response."
    ),
}

# Only split on top-level headings. Transcript phase sections are '# ...',
# while in-section coach feedback markers are typically '## ...'.
SECTION_HEADER_RE = re.compile(r"^\s*#\s+\*{0,2}(.+?)\*{0,2}\s*$")
SPEAKER_RE = re.compile(
    r"^\s*\*\*(Clinician, MD|CLINICIAN|Clinician|NURSE|Doctor|DOCTOR|ANNE PALMER|MAYA PENA|Maya Pena):\*\*\s*(.*)$",
    re.IGNORECASE,
)
INLINE_COACH_NOTE_RE = re.compile(r"^\s*\[COACH:\s*(.+?)\]\s*$", re.IGNORECASE)
COACH_FEEDBACK_HEADER_RE = re.compile(r"COACH\s+FEEDBACK", re.IGNORECASE)
SUMMATIVE_TITLE_RE = re.compile(r"SUMMATIVE", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"^\s*---+\s*$")

OPENING_SENTENCE_RE = re.compile(r"(?:^|\s)(Thank you[^.\n]{8,260}\.|I appreciate[^.\n]{8,260}\.)", re.IGNORECASE)
IMPROVEMENT_SENTENCE_RE = re.compile(r"(?:^|\s)(Try next time[^.\n]{8,260}\.)", re.IGNORECASE)
CLOSING_SENTENCE_RE = re.compile(
    r"(?:^|\s)(?:This session[^.\n]{10,280}\.|Repeated practice[^.\n]{10,280}\.)",
    re.IGNORECASE,
)


@dataclass
class CoachCase:
    parent_id: str
    clear_phase: str
    fixture_type: str
    prompt: str
    expected: str
    source_file: str


@dataclass
class DatasetRow:
    parent_id: str
    clear_phase: str
    fixture_type: str
    case_bucket: str
    messages: List[Dict[str, str]]
    meta: Dict[str, Any]


def normalize_md_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", value)
    value = value.replace("**", "")
    value = re.sub(r"(?<!\*)\*(?!\*)", "", value)
    return value.strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collapse_paragraphs(text: str) -> str:
    parts = [normalize_whitespace(p) for p in re.split(r"\n\s*\n", text) if normalize_whitespace(p)]
    return "\n\n".join(parts)


def split_sections(content: str) -> List[Tuple[str, str]]:
    lines = content.split("\n")
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_title, current_lines
        if current_title or any(line.strip() for line in current_lines):
            sections.append((current_title, "\n".join(current_lines)))
        current_title = ""
        current_lines = []

    for line in lines:
        match = SECTION_HEADER_RE.match(line)
        if match:
            flush()
            current_title = normalize_md_text(match.group(1))
            continue
        current_lines.append(line)

    flush()
    return sections


def infer_parent_id(file_name: str, content: str) -> str:
    low_name = file_name.lower()
    low = content.lower()
    if "maya pena" in low or "luna" in low or "infertil" in low:
        return "maya_pena"
    if "anne palmer" in low or "riley" in low:
        return "anne_palmer"
    if "2nd skills practice" in low_name:
        return "maya_pena"
    if "1st skills practice" in low_name:
        return "anne_palmer"
    return ""


def phase_from_title(title: str) -> str:
    low = title.lower()
    if "counsel" in low:
        return "COUNSEL"
    if "listen" in low:
        return "LISTEN"
    if "empath" in low:
        return "EMPATHIZE"
    if "answer" in low or "recommend" in low:
        return "ANSWER_RECOMMEND"
    if "summative" in low:
        return "SUMMATIVE"
    return ""


def phase_from_coach_note(note: str) -> str:
    low = note.lower()
    if "counsel" in low:
        return "COUNSEL"
    if "listen" in low:
        return "LISTEN"
    if "empathy" in low or "empath" in low:
        return "EMPATHIZE"
    if "answer" in low or "recommend" in low:
        return "ANSWER_RECOMMEND"
    return ""


def extract_speaker_turns(section_body: str) -> List[Tuple[str, str]]:
    turns: List[Tuple[str, str]] = []
    current_speaker = ""
    current_chunks: List[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_chunks
        if current_speaker and current_chunks:
            merged = " ".join(chunk.strip() for chunk in current_chunks if chunk.strip())
            merged = normalize_md_text(merged)
            merged = normalize_whitespace(merged)
            if merged:
                turns.append((current_speaker.lower(), merged))
        current_speaker = ""
        current_chunks = []

    for raw_line in section_body.split("\n"):
        line = raw_line.strip()
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            flush()
            current_speaker = speaker_match.group(1)
            current_chunks = [speaker_match.group(2)] if speaker_match.group(2) else []
            continue

        if not current_speaker:
            continue

        if not line:
            flush()
            continue

        if line.startswith("#") or line.startswith("---"):
            flush()
            continue

        if line.startswith("**") and line.endswith("**"):
            flush()
            continue

        if line.startswith("[") and line.endswith("]"):
            flush()
            continue

        current_chunks.append(line)

    flush()
    return turns


def extract_prompt(section_body: str, fixture_type: str) -> str:
    if fixture_type == "summative":
        return "[Full Conversation History]"

    turns = extract_speaker_turns(section_body)
    for speaker, utterance in turns:
        if speaker in {"clinician, md", "clinician", "nurse", "doctor"}:
            return utterance
    return ""


def clean_feedback_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^\*+\s*", "", text)
    text = re.sub(r"^-\s+", "", text)
    text = re.sub(r"^\*\*\(Feedback with rationale\):\*\*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\*\*COACH:?\*\*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^COACH:?$", "", text, flags=re.IGNORECASE)
    text = text.replace("**", "")
    text = normalize_md_text(text)
    return normalize_whitespace(text)


def extract_feedback_text(section_body: str, section_title: str) -> str:
    lines = section_body.split("\n")
    is_summative = bool(SUMMATIVE_TITLE_RE.search(section_title))

    start_idx = 0
    if is_summative:
        for i, line in enumerate(lines):
            if re.search(r"\*\*COACH:?\*\*", line, flags=re.IGNORECASE):
                start_idx = i + 1
                break
    else:
        found = False
        for i, line in enumerate(lines):
            if COACH_FEEDBACK_HEADER_RE.search(line):
                start_idx = i + 1
                found = True
                break
        if not found:
            return ""

    paragraphs: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            para = normalize_whitespace(" ".join(current))
            if para:
                paragraphs.append(para)
        current = []

    for raw_line in lines[start_idx:]:
        stripped = raw_line.strip()

        if SECTION_HEADER_RE.match(stripped):
            break
        if SEPARATOR_RE.match(stripped):
            break
        if not stripped:
            flush()
            continue
        if COACH_FEEDBACK_HEADER_RE.search(stripped):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue

        cleaned = clean_feedback_line(stripped)
        if not cleaned:
            continue
        current.append(cleaned)

    flush()
    return collapse_paragraphs("\n\n".join(paragraphs))


def extract_inline_coach_cases(
    section_body: str,
    parent_id: str,
    source_file: str,
) -> List[CoachCase]:
    """Extract clinician->inline-coach-note pairs from long scenario transcripts.

    Some 2nd-skills markdown files encode formative feedback as inline
    bracketed notes (e.g., [COACH: ...]) instead of dedicated
    "COACH FEEDBACK" blocks. This parser converts those notes into
    quote-faithful formative cases.
    """
    cases: List[CoachCase] = []
    clinician_speakers = {"clinician, md", "clinician", "nurse", "doctor"}

    current_clinician = ""
    collecting_clinician = False

    for raw_line in section_body.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            speaker = speaker_match.group(1).lower()
            text = normalize_whitespace(normalize_md_text(speaker_match.group(2)))
            if speaker in clinician_speakers:
                current_clinician = text
                collecting_clinician = True
            else:
                collecting_clinician = False
            continue

        coach_note_match = INLINE_COACH_NOTE_RE.match(line)
        if coach_note_match and current_clinician:
            note_text = normalize_whitespace(normalize_md_text(coach_note_match.group(1)))
            clear_phase = phase_from_coach_note(note_text)
            if clear_phase:
                cases.append(
                    CoachCase(
                        parent_id=parent_id,
                        clear_phase=clear_phase,
                        fixture_type="formative",
                        prompt=current_clinician,
                        expected=note_text,
                        source_file=source_file,
                    )
                )
            continue

        if collecting_clinician:
            if line.startswith("[") or line.startswith("#") or line.startswith("**"):
                continue
            extra = normalize_whitespace(normalize_md_text(line))
            if extra:
                current_clinician = normalize_whitespace(f"{current_clinician} {extra}")

    return cases


def transcript_markdown_paths() -> List[Path]:
    coach_transcript_paths = sorted(
        p for p in HERE.glob("*.md") if "transcript" in p.name.lower()
    )
    first_skills_paths = sorted(
        p for p in TRANSCRIPTS_DIR.glob("*.md") if "transcript" in p.name.lower()
    )
    return [*coach_transcript_paths, *first_skills_paths]


def all_coach_markdown_paths() -> List[Path]:
    return sorted(HERE.glob("*.md"))


def collect_style_hints(paths: Sequence[Path]) -> Dict[str, List[str]]:
    openings: set[str] = set()
    improvements: set[str] = set()
    closings: set[str] = set()

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in OPENING_SENTENCE_RE.finditer(text):
            sentence = normalize_whitespace(normalize_md_text(match.group(0)))
            if sentence:
                openings.add(sentence)
        for match in IMPROVEMENT_SENTENCE_RE.finditer(text):
            sentence = normalize_whitespace(normalize_md_text(match.group(0)))
            if sentence:
                improvements.add(sentence)
        for match in CLOSING_SENTENCE_RE.finditer(text):
            sentence = normalize_whitespace(normalize_md_text(match.group(0)))
            if sentence:
                closings.add(sentence)

    return {
        "openings": sorted(openings),
        "improvements": sorted(improvements),
        "closings": sorted(closings),
    }


def collect_transcript_cases(paths: Sequence[Path]) -> List[CoachCase]:
    cases: List[CoachCase] = []

    for path in paths:
        content = path.read_text(encoding="utf-8", errors="ignore")
        parent_id = infer_parent_id(path.name, content)
        if parent_id not in PARENT_IDS:
            continue

        sections = split_sections(content)
        for title, body in sections:
            phase = phase_from_title(title)
            if not phase:
                continue

            fixture_type = "summative" if phase == "SUMMATIVE" else "formative"
            if fixture_type != "summative":
                # Capture inline [COACH: ...] notes even when no explicit
                # "COACH FEEDBACK" sub-header exists in the section.
                cases.extend(extract_inline_coach_cases(body, parent_id=parent_id, source_file=path.name))

            prompt = extract_prompt(body, fixture_type=fixture_type)
            expected = extract_feedback_text(body, section_title=title)

            if not prompt or not expected:
                continue

            cases.append(
                CoachCase(
                    parent_id=parent_id,
                    clear_phase=phase,
                    fixture_type=fixture_type,
                    prompt=prompt,
                    expected=expected,
                    source_file=path.name,
                )
            )

    # Deduplicate exact repeats produced by overlapping extraction passes.
    unique_cases: List[CoachCase] = []
    seen: set[Tuple[str, str, str, str, str]] = set()
    for case in cases:
        key = (case.parent_id, case.clear_phase, case.fixture_type, case.prompt, case.expected)
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append(case)

    return unique_cases


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def collect_grading_logic_cases(path: Path) -> List[CoachCase]:
    cases: List[CoachCase] = []
    for row in read_jsonl(path):
        messages = row.get("messages", [])
        if len(messages) < 2:
            continue
        prompt = normalize_whitespace(str(messages[0].get("content", "")))
        expected = collapse_paragraphs(str(messages[1].get("content", "")))
        if not prompt or not expected:
            continue
        cases.append(
            CoachCase(
                parent_id="coach_policy",
                clear_phase="GRADING_LOGIC",
                fixture_type="grading_logic",
                prompt=prompt,
                expected=expected,
                source_file=path.name,
            )
        )
    return cases


def stable_bucket(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:12]


def split_paragraphs(text: str) -> List[str]:
    return [normalize_whitespace(p) for p in re.split(r"\n\s*\n", text) if normalize_whitespace(p)]


def dedupe_in_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def synthesize_feedback(
    rng: random.Random,
    phase_cases: Sequence[CoachCase],
    style_hints: Dict[str, List[str]],
) -> str:
    if not phase_cases:
        return ""

    if len(phase_cases) == 1 or rng.random() < 0.60:
        return phase_cases[rng.randrange(len(phase_cases))].expected

    first, second = rng.sample(list(phase_cases), 2)
    first_pars = split_paragraphs(first.expected)
    second_pars = split_paragraphs(second.expected)

    combined: List[str] = []
    if first_pars:
        combined.append(first_pars[0])
    if len(first_pars) > 1 and rng.random() < 0.50:
        combined.append(first_pars[1])
    if second_pars:
        combined.append(second_pars[-1])

    combined = dedupe_in_order(combined)
    text = "\n\n".join(combined)

    text_low = text.lower()
    if style_hints.get("openings") and not (text_low.startswith("thank you") or text_low.startswith("i appreciate")):
        if rng.random() < 0.35:
            opening = rng.choice(style_hints["openings"])
            text = f"{opening}\n\n{text}" if text else opening

    if style_hints.get("improvements") and "try next time" not in text.lower():
        if rng.random() < 0.30:
            improvement = rng.choice(style_hints["improvements"])
            text = f"{text}\n\n{improvement}" if text else improvement

    return collapse_paragraphs(text)


def build_system_message(base_prompt: str, parent_id: str, clear_phase: str) -> str:
    phase_insert = PHASE_SYSTEM_INSERTS.get(clear_phase, "")
    return (
        f"{base_prompt}\n\n"
        f"<ACTIVE_CLEAR_PHASE>\n"
        f"Parent: {parent_id}\n"
        f"Phase: {clear_phase}\n"
        f"{phase_insert}\n"
        f"</ACTIVE_CLEAR_PHASE>"
    )


def phase_case_pools(cases: Sequence[CoachCase]) -> Dict[Tuple[str, str], List[CoachCase]]:
    pools: Dict[Tuple[str, str], List[CoachCase]] = defaultdict(list)
    for case in cases:
        pools[(case.parent_id, case.clear_phase)].append(case)
    return pools


def generate_quote_faithful_rows(
    rng: random.Random,
    base_prompt: str,
    transcript_cases: Sequence[CoachCase],
    grading_logic_cases: Sequence[CoachCase],
    style_hints: Dict[str, List[str]],
    per_phase_per_parent: int,
    summative_per_parent: int,
    grading_logic_copies: int,
) -> List[DatasetRow]:
    rows: List[DatasetRow] = []
    pools = phase_case_pools(transcript_cases)

    # Include all source transcript rows as anchors.
    for case in transcript_cases:
        system_msg = build_system_message(base_prompt, case.parent_id, case.clear_phase)
        rows.append(
            DatasetRow(
                parent_id=case.parent_id,
                clear_phase=case.clear_phase,
                fixture_type=case.fixture_type,
                case_bucket=f"{case.parent_id}|{case.clear_phase}|{stable_bucket(case.prompt)}",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": case.prompt},
                    {"role": "assistant", "content": case.expected},
                ],
                meta={"kind": "anchor", "source_file": case.source_file},
            )
        )

    # Balanced synthetic rows across parent x phase.
    for parent_id in PARENT_IDS:
        for clear_phase in COACH_PHASES:
            phase_cases = pools.get((parent_id, clear_phase), [])
            if not phase_cases:
                continue

            target_count = summative_per_parent if clear_phase == "SUMMATIVE" else per_phase_per_parent
            for _ in range(target_count):
                picked_prompt_case = rng.choice(phase_cases)
                prompt = picked_prompt_case.prompt
                expected = synthesize_feedback(rng, phase_cases, style_hints)
                system_msg = build_system_message(base_prompt, parent_id, clear_phase)

                rows.append(
                    DatasetRow(
                        parent_id=parent_id,
                        clear_phase=clear_phase,
                        fixture_type="summative" if clear_phase == "SUMMATIVE" else "formative",
                        case_bucket=f"{parent_id}|{clear_phase}|{stable_bucket(prompt)}",
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": expected},
                        ],
                        meta={
                            "kind": "quote_faithful_mix",
                            "pool_size": len(phase_cases),
                            "source_file": picked_prompt_case.source_file,
                        },
                    )
                )

    # Append grading-logic exemplars as system-conditioned Q/A rows.
    for _ in range(max(1, grading_logic_copies)):
        for case in grading_logic_cases:
            system_msg = build_system_message(base_prompt, "coach_policy", case.clear_phase)
            rows.append(
                DatasetRow(
                    parent_id="coach_policy",
                    clear_phase=case.clear_phase,
                    fixture_type="grading_logic",
                    case_bucket=f"coach_policy|GRADING_LOGIC|{stable_bucket(case.prompt)}",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": case.prompt},
                        {"role": "assistant", "content": case.expected},
                    ],
                    meta={"kind": "policy_qa", "source_file": case.source_file},
                )
            )

    rng.shuffle(rows)
    return rows


def holdout_split(
    rows: Sequence[DatasetRow],
    rng: random.Random,
    eval_fraction: float,
) -> Tuple[List[DatasetRow], List[DatasetRow]]:
    buckets_by_parent: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        if row.case_bucket not in buckets_by_parent[row.parent_id]:
            buckets_by_parent[row.parent_id].append(row.case_bucket)

    holdouts: set[Tuple[str, str]] = set()
    for parent_id, buckets in buckets_by_parent.items():
        selected = list(buckets)
        rng.shuffle(selected)
        take = max(1, int(round(len(selected) * eval_fraction)))
        for bucket in selected[:take]:
            holdouts.add((parent_id, bucket))

    train_rows: List[DatasetRow] = []
    eval_rows: List[DatasetRow] = []
    for row in rows:
        key = (row.parent_id, row.case_bucket)
        if key in holdouts:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def write_jsonl_rows(rows: Sequence[DatasetRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({"messages": row.messages}, ensure_ascii=False))
            fh.write("\n")


def write_fixture_jsonl(
    transcript_cases: Sequence[CoachCase],
    grading_logic_cases: Sequence[CoachCase],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for case in transcript_cases:
            fh.write(
                json.dumps(
                    {
                        "parent_label": case.parent_id,
                        "clear_phase": case.clear_phase,
                        "fixture_type": case.fixture_type,
                        "prompt": case.prompt,
                        "expected": case.expected,
                        "source_file": case.source_file,
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

        for case in grading_logic_cases:
            fh.write(
                json.dumps(
                    {
                        "parent_label": case.parent_id,
                        "clear_phase": case.clear_phase,
                        "fixture_type": case.fixture_type,
                        "prompt": case.prompt,
                        "expected": case.expected,
                        "source_file": case.source_file,
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")


def summarize(
    rows: Sequence[DatasetRow],
    transcript_cases: Sequence[CoachCase],
) -> Dict[str, Any]:
    by_parent: Counter[str] = Counter()
    by_phase: Counter[str] = Counter()
    by_fixture_type: Counter[str] = Counter()
    by_kind: Counter[str] = Counter()
    assistant_counts: Counter[str] = Counter()

    source_feedback = {case.expected for case in transcript_cases}
    exact_quote_rows = 0

    for row in rows:
        by_parent[row.parent_id] += 1
        by_phase[row.clear_phase] += 1
        by_fixture_type[row.fixture_type] += 1
        by_kind[str(row.meta.get("kind", "unknown"))] += 1
        assistant_text = row.messages[-1]["content"] if row.messages else ""
        assistant_counts[assistant_text] += 1
        if assistant_text in source_feedback:
            exact_quote_rows += 1

    return {
        "row_count": len(rows),
        "by_parent": dict(by_parent),
        "by_phase": dict(by_phase),
        "by_fixture_type": dict(by_fixture_type),
        "by_kind": dict(by_kind),
        "unique_assistant_strings": len(assistant_counts),
        "max_assistant_string_repeats": assistant_counts.most_common(1)[0][1] if assistant_counts else 0,
        "quote_anchor_ratio": round(exact_quote_rows / len(rows), 4) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-phase-per-parent", type=int, default=120)
    parser.add_argument("--summative-per-parent", type=int, default=40)
    parser.add_argument("--grading-logic-copies", type=int, default=10)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument(
        "--out",
        type=str,
        default=str(HERE / "coach_synthetic_dataset_v2.jsonl"),
    )
    parser.add_argument(
        "--fixtures-out",
        type=str,
        default=str(HERE / "coach_h6_fixtures_v2.jsonl"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    coach_prompt = COACH_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    transcript_paths = transcript_markdown_paths()
    all_coach_md_paths = all_coach_markdown_paths()

    transcript_cases = collect_transcript_cases(transcript_paths)
    grading_logic_cases = collect_grading_logic_cases(GRADING_LOGIC_JSONL)
    style_hints = collect_style_hints(all_coach_md_paths)

    if not transcript_cases:
        raise RuntimeError("No transcript coach cases were extracted. Check markdown source formatting.")

    rows = generate_quote_faithful_rows(
        rng=rng,
        base_prompt=coach_prompt,
        transcript_cases=transcript_cases,
        grading_logic_cases=grading_logic_cases,
        style_hints=style_hints,
        per_phase_per_parent=args.per_phase_per_parent,
        summative_per_parent=args.summative_per_parent,
        grading_logic_copies=args.grading_logic_copies,
    )

    train_rows, eval_rows = holdout_split(
        rows,
        rng=random.Random(args.seed + 1),
        eval_fraction=args.eval_fraction,
    )

    out_path = Path(args.out)
    train_path = out_path.with_name(out_path.stem + ".train.jsonl")
    eval_path = out_path.with_name(out_path.stem + ".eval.jsonl")
    stats_path = out_path.with_name(out_path.stem + ".stats.json")

    write_jsonl_rows(rows, out_path)
    write_jsonl_rows(train_rows, train_path)
    write_jsonl_rows(eval_rows, eval_path)

    fixtures_path = Path(args.fixtures_out)
    write_fixture_jsonl(transcript_cases, grading_logic_cases, fixtures_path)

    stats = {
        "source_markdown_count": len(all_coach_md_paths),
        "source_transcript_markdown_count": len(transcript_paths),
        "source_transcript_case_count": len(transcript_cases),
        "source_grading_logic_case_count": len(grading_logic_cases),
        "style_hint_counts": {
            "openings": len(style_hints.get("openings", [])),
            "improvements": len(style_hints.get("improvements", [])),
            "closings": len(style_hints.get("closings", [])),
        },
        "total": summarize(rows, transcript_cases),
        "train": summarize(train_rows, transcript_cases),
        "eval": summarize(eval_rows, transcript_cases),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Read coach markdown files: {len(all_coach_md_paths)}")
    print(f"Read transcript markdown files: {len(transcript_paths)}")
    print(f"Extracted transcript cases: {len(transcript_cases)}")
    print(f"Extracted grading-logic Q/A cases: {len(grading_logic_cases)}")
    print(f"Wrote dataset: {out_path}")
    print(f"Wrote train split: {train_path}")
    print(f"Wrote eval split: {eval_path}")
    print(f"Wrote H6 fixtures: {fixtures_path}")
    print(f"Wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
