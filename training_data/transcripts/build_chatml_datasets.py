"""Build ChatML training datasets from transcript markdown files.

This script scans `training_data/transcripts/*.md` and produces:
    - `Caregiver/train.jsonl`
    - `C-LEAR_Coach/train.jsonl`

Output format (one JSON object per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Extraction rules:
    - Caregiver dataset:
            user      = clinician utterance (`Clinician, MD`)
            assistant = caregiver utterance (`ANNE PALMER`)
    - Coach dataset:
            user      = clinician practice attempt
            assistant = exact plain-text feedback from `COACH FEEDBACK` sections
    - Summative coach record:
            user      = "[Full Conversation History]"
            assistant = exact plain-text summative feedback

No grading/scoring inference is performed. No structured feedback schema is generated.

Run:
    python training_data/transcripts/build_chatml_datasets.py
"""

import json
import re
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS_DIR = WORKSPACE_ROOT / "training_data" / "transcripts"
CAREGIVER_OUTPUT = WORKSPACE_ROOT / "Caregiver" / "train.jsonl"
COACH_OUTPUT = WORKSPACE_ROOT / "C-LEAR_Coach" / "train.jsonl"


SECTION_HEADER_RE = re.compile(r"^#\s+\*\*(.+?)\*\*\s*$", re.MULTILINE)
SPEAKER_RE = re.compile(r"^\*\*(Clinician, MD|ANNE PALMER):\*\*\s*(.*)$")
COACH_FEEDBACK_HEADER_RE = re.compile(r"^\s*##\s+\*\*COACH FEEDBACK.*\*\*\s*$", re.IGNORECASE)
SUMMATIVE_HEADER_RE = re.compile(r"1st Skills Practice:\s*SUMMATIVE FEEDBACK", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """Normalize markdown-ish text while preserving sentence meaning.

    - Converts line endings to `\n`
    - Unescapes markdown punctuation (for example: escaped period to period)
    - Removes emphasis markers (`*`, `**`)
    - Collapses excess blank lines
    """
    value = text.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", r"\1", value)
    value = value.replace("**", "")
    value = re.sub(r"(?<!\*)\*(?!\*)", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def split_sections(content: str):
    """Split transcript markdown into top-level section tuples: (title, body)."""
    matches = list(SECTION_HEADER_RE.finditer(content))
    sections = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        title = normalize_text(match.group(1))
        body = content[start:end]
        sections.append((title, body))
    return sections


def extract_speaker_turns(section_body: str):
    """Extract ordered speaker turns from a section body.

    Returns a list of tuples: (speaker_name, utterance_text).
    """
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
            flush_current()
            current_speaker = speaker_match.group(1)
            current_chunks = [speaker_match.group(2)] if speaker_match.group(2) else []
            continue

        if current_speaker is None:
            continue

        if not line:
            flush_current()
            continue

        if line.startswith("#") or line.startswith("---") or line.startswith("**COACH"):
            flush_current()
            continue

        if line.startswith("**") and line.endswith("**"):
            flush_current()
            continue

        current_chunks.append(line)

    flush_current()
    return turns


def extract_feedback_text(section_body: str):
    """Extract plain-text feedback paragraphs from COACH FEEDBACK blocks.

    Returns an empty string when a section has no feedback block.
    """
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
        cleaned = re.sub(r"^\*\*\(Feedback with rationale\):\*\*$", "", cleaned, flags=re.IGNORECASE)

        cleaned = normalize_text(cleaned)
        if cleaned:
            current.append(cleaned)

    flush_paragraph()
    return "\n\n".join(paragraphs).strip()


def extract_summative_feedback_text(section_body: str):
    """Extract plain-text summative feedback from the SUMMATIVE section body."""
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


def build_records_for_file(file_path: Path):
    """Build caregiver and coach ChatML records for one transcript file.

    Returns:
        (caregiver_records, coach_records)
    """
    raw_content = file_path.read_text(encoding="utf-8")
    content = raw_content.replace("\r\n", "\n").replace("\r", "\n")
    sections = split_sections(content)

    caregiver_records = []
    coach_records = []
    summative_feedback_text = ""

    for title, body in sections:
        turns = extract_speaker_turns(body)
        clinician_utterance = next((text for speaker, text in turns if speaker == "Clinician, MD"), "")
        caregiver_utterance = next((text for speaker, text in turns if speaker == "ANNE PALMER"), "")

        if clinician_utterance and caregiver_utterance:
            caregiver_records.append(
                {
                    "messages": [
                        {"role": "user", "content": clinician_utterance},
                        {"role": "assistant", "content": caregiver_utterance},
                    ]
                }
            )

        feedback_text = extract_feedback_text(body)
        if feedback_text:
            if clinician_utterance:
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

    if summative_feedback_text:
        coach_records.append(
            {
                "messages": [
                    {"role": "user", "content": "[Full Conversation History]"},
                    {"role": "assistant", "content": summative_feedback_text},
                ]
            }
        )

    return caregiver_records, coach_records


def write_jsonl(path: Path, records):
    """Write records as UTF-8 newline-delimited JSON (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def main():
    """Entry point: parse all transcript markdown files and write both datasets."""
    transcript_files = sorted(
        (
            path
            for path in TRANSCRIPTS_DIR.glob("*.md")
            if path.name.lower() != "readme.md"
        ),
        key=lambda p: p.name.lower(),
    )
    caregiver_records_all = []
    coach_records_all = []

    for transcript_file in transcript_files:
        caregiver_records, coach_records = build_records_for_file(transcript_file)
        caregiver_records_all.extend(caregiver_records)
        coach_records_all.extend(coach_records)

    write_jsonl(CAREGIVER_OUTPUT, caregiver_records_all)
    write_jsonl(COACH_OUTPUT, coach_records_all)

    print(f"Processed transcripts: {len(transcript_files)}")
    print(f"Caregiver records: {len(caregiver_records_all)} -> {CAREGIVER_OUTPUT}")
    print(f"Coach records: {len(coach_records_all)} -> {COACH_OUTPUT}")


if __name__ == "__main__":
    main()
