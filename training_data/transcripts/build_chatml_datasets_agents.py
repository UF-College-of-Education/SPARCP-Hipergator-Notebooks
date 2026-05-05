"""Build ChatML training datasets from transcript markdown files.

Expected layout (same directory as this script):
    build_chatml_datasets.py
    Anne/
        *.md
    Maya/
        *.md

Outputs:
    - parent-1st-skills-practice-transcripts.jsonl   (from Anne/)
    - parent-2nd-skills-practice-transcripts.jsonl   (from Maya/)
    - C-LEAR_Coach/train.jsonl                       (combined coach output)

Output format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import re
from pathlib import Path


FOLDER_TO_CAREGIVER = {
    "anne": {
        "canonical": "ANNE PALMER",
        "aliases": {"ANNE PALMER", "ANNE"},
    },
    "maya": {
        "canonical": "MAYA PENA",
        "aliases": {"MAYA PENA", "MAYA"},
    },
}

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = SCRIPT_DIR

ANNE_OUTPUT = SCRIPT_DIR / "parent-1st-skills-practice-transcripts.jsonl"
MAYA_OUTPUT = SCRIPT_DIR / "parent-2nd-skills-practice-transcripts.jsonl"
COACH_OUTPUT = SCRIPT_DIR / "C-LEAR_Coach" / "train.jsonl"

SECTION_HEADER_RE = re.compile(r"^#\s+\*\*(.+?)\*\*\s*$", re.MULTILINE)
SPEAKER_RE = re.compile(r"^(?:\*\*)?(.+?)(?::)(?:\*\*)?\s*(.*)$")
COACH_FEEDBACK_HEADER_RE = re.compile(
    r"^\s*##?\s+\*\*COACH FEEDBACK.*\*\*\s*$",
    re.IGNORECASE,
)
SUMMATIVE_HEADER_RE = re.compile(
    r"SUMMATIVE(?:\s+COACHING)?\s+FEEDBACK",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", value)
    value = value.replace("**", "")
    value = re.sub(r"(?<!\*)\*(?!\*)", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_speaker_name(name: str) -> str:
    value = normalize_text(name).upper().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def speaker_is_clinician(speaker_name: str) -> bool:
    return normalize_speaker_name(speaker_name) in {
        "CLINICIAN, MD",
        "CLINICIAN",
        "NURSE",
        "DOCTOR",
        "PROVIDER",
    }


def speaker_is_caregiver(speaker_name: str, caregiver_aliases: set[str]) -> bool:
    return normalize_speaker_name(speaker_name) in caregiver_aliases


def split_sections(content: str):
    matches = list(SECTION_HEADER_RE.finditer(content))
    if not matches:
        return [("FULL_TRANSCRIPT", content)]

    sections = []

    if matches[0].start() > 0:
        preamble = content[:matches[0].start()]
        if preamble.strip():
            sections.append(("FULL_TRANSCRIPT_PREAMBLE", preamble))

    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        title = normalize_text(match.group(1))
        body = content[start:end]
        sections.append((title, body))

    return sections


def extract_speaker_turns(section_body: str):
    turns = []
    current_speaker = None
    current_chunks = []

    def flush_current():
        nonlocal current_speaker, current_chunks
        if current_speaker and current_chunks:
            merged = " ".join(chunk.strip() for chunk in current_chunks if chunk.strip())
            merged = normalize_text(merged)
            if merged:
                turns.append((current_speaker, merged))
        current_speaker = None
        current_chunks = []

    for raw_line in section_body.split("\n"):
        line = raw_line.strip()
        speaker_match = SPEAKER_RE.match(line)

        if speaker_match:
            raw_speaker = speaker_match.group(1).strip()
            raw_text = speaker_match.group(2).strip()
            normalized_speaker = normalize_speaker_name(raw_speaker)

            # Skip coach metadata / annotation lines as speakers
            if normalized_speaker.startswith("[") or normalized_speaker in {"COACH", "PARENT"}:
                flush_current()
                continue

            flush_current()
            current_speaker = raw_speaker
            current_chunks = [raw_text] if raw_text else []
            continue

        if current_speaker is None:
            continue

        if not line:
            flush_current()
            continue

        if line.startswith("#") or line.startswith("---"):
            flush_current()
            continue

        if line.startswith("[") and line.endswith("]"):
            flush_current()
            continue

        if line.startswith("**") and line.endswith("**"):
            flush_current()
            continue

        current_chunks.append(line)

    flush_current()
    return turns


def extract_feedback_text(section_body: str):
    lines = section_body.split("\n")
    start_index = None
    for index, line in enumerate(lines):
        if COACH_FEEDBACK_HEADER_RE.match(line.strip()):
            start_index = index + 1
            break

    if start_index is None:
        return ""

    feedback_lines = lines[start_index:]
    paragraphs = []
    current = []

    def flush_paragraph():
        nonlocal current
        if current:
            paragraph = " ".join(current).strip()
            paragraph = normalize_text(paragraph)
            if paragraph:
                paragraphs.append(paragraph)
        current = []

    for raw_line in feedback_lines:
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if stripped == "---":
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            break

        cleaned = stripped
        cleaned = re.sub(r"^\*\s+", "", cleaned)
        cleaned = re.sub(r"^\*\*COACH:?\*\*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^COACH:?$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"^\*\*\(Feedback with rationale\):\*\*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = normalize_text(cleaned)
        if cleaned:
            current.append(cleaned)

    flush_paragraph()
    return "\n\n".join(paragraphs).strip()


def extract_summative_feedback_text(section_body: str):
    lines = section_body.split("\n")
    paragraphs = []
    current = []

    def flush_paragraph():
        nonlocal current
        if current:
            paragraph = " ".join(current).strip()
            paragraph = normalize_text(paragraph)
            if paragraph:
                paragraphs.append(paragraph)
        current = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if not stripped:
            flush_paragraph()
            continue

        if stripped == "---":
            flush_paragraph()
            continue

        cleaned = stripped
        cleaned = re.sub(r"^\*\s+", "", cleaned)
        cleaned = re.sub(r"^\*\*COACH:?\*\*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^COACH:?$", "", cleaned, flags=re.IGNORECASE)
        cleaned = normalize_text(cleaned)

        if cleaned:
            current.append(cleaned)

    flush_paragraph()
    return "\n\n".join(paragraphs).strip()


def pair_turns(turns, caregiver_aliases: set[str]):
    """Pair each clinician/provider utterance with the next caregiver utterance."""
    records = []
    pending_clinician = None

    for speaker, text in turns:
        if speaker_is_clinician(speaker):
            pending_clinician = text
            continue

        if speaker_is_caregiver(speaker, caregiver_aliases):
            if pending_clinician:
                records.append(
                    {
                        "messages": [
                            {"role": "user", "content": pending_clinician},
                            {"role": "assistant", "content": text},
                        ]
                    }
                )
                pending_clinician = None

    return records


def build_records_for_file(file_path: Path):
    raw_content = file_path.read_text(encoding="utf-8")
    content = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    sections = split_sections(content)

    caregiver_records = []
    coach_records = []
    summative_feedback_text = ""

    folder_name = file_path.parent.name.lower()
    caregiver_info = FOLDER_TO_CAREGIVER.get(folder_name)

    if caregiver_info is None:
        print(f"Skipping {file_path} (unknown folder/persona: {folder_name})")
        return [], [], folder_name

    caregiver_aliases = {
        normalize_speaker_name(alias)
        for alias in caregiver_info["aliases"]
    }

    for title, body in sections:
        turns = extract_speaker_turns(body)

        paired_records = pair_turns(turns, caregiver_aliases)
        caregiver_records.extend(paired_records)

        clinician_utterance = next(
            (text for speaker, text in turns if speaker_is_clinician(speaker)),
            "",
        )

        feedback_text = extract_feedback_text(body)
        if feedback_text and clinician_utterance:
            coach_records.append(
                {
                    "messages": [
                        {"role": "user", "content": clinician_utterance},
                        {"role": "assistant", "content": feedback_text},
                    ]
                }
            )

        if SUMMATIVE_HEADER_RE.search(title):
            summative_feedback_text = extract_summative_feedback_text(body)

    # Fallback for freeform transcripts where the main dialogue is not captured well by sections
    if not caregiver_records:
        turns = extract_speaker_turns(content)
        caregiver_records.extend(pair_turns(turns, caregiver_aliases))

    # Fallback for summative section phrasing variants
    if not summative_feedback_text and SUMMATIVE_HEADER_RE.search(content):
        lines = content.split("\n")
        start_index = None
        for index, line in enumerate(lines):
            if line.strip().startswith("#") and SUMMATIVE_HEADER_RE.search(line):
                start_index = index + 1
                break

        if start_index is not None:
            summative_feedback_text = extract_summative_feedback_text(
                "\n".join(lines[start_index:])
            )

    if summative_feedback_text:
        coach_records.append(
            {
                "messages": [
                    {"role": "user", "content": "[Full Conversation History]"},
                    {"role": "assistant", "content": summative_feedback_text},
                ]
            }
        )

    print(
        f"Processed {file_path.name} [{folder_name}] | "
        f"caregiver_records={len(caregiver_records)} | "
        f"coach_records={len(coach_records)}"
    )

    return caregiver_records, coach_records, folder_name


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def main():
    transcript_files = sorted(
        [
            path
            for path in TRANSCRIPTS_DIR.rglob("*.md")
            if path.name.lower() != "readme.md"
            and path.parent.name.lower() in {"anne", "maya"}
        ],
        key=lambda p: (p.parent.name.lower(), p.name.lower()),
    )

    anne_records = []
    maya_records = []
    coach_records_all = []

    for transcript_file in transcript_files:
        caregiver_records, coach_records, folder_name = build_records_for_file(transcript_file)

        if folder_name == "anne":
            anne_records.extend(caregiver_records)
        elif folder_name == "maya":
            maya_records.extend(caregiver_records)

        coach_records_all.extend(coach_records)

    write_jsonl(ANNE_OUTPUT, anne_records)
    write_jsonl(MAYA_OUTPUT, maya_records)
    write_jsonl(COACH_OUTPUT, coach_records_all)

    print(f"Processed transcripts: {len(transcript_files)}")
    print(f"Anne caregiver records: {len(anne_records)} -> {ANNE_OUTPUT}")
    print(f"Maya caregiver records: {len(maya_records)} -> {MAYA_OUTPUT}")
    print(f"Coach records: {len(coach_records_all)} -> {COACH_OUTPUT}")


if __name__ == "__main__":
    main()