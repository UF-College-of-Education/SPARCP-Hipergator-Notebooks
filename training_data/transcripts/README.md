# Transcript to ChatML Dataset Builder

This folder contains source transcript markdown files and the converter script used to generate training datasets.

## Script

- `build_chatml_datasets.py`

## Inputs

- All markdown files in `training_data/transcripts/*.md`

## Outputs

- `Caregiver/train.jsonl`
- `C-LEAR_Coach/train.jsonl`

Both outputs use one-line-per-record HuggingFace ChatML JSON objects:

`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

## Extraction Rules

### Caregiver dataset

- `user`: exact clinician text (`Clinician, MD`)
- `assistant`: exact caregiver text (`ANNE PALMER`)

### C-LEAR Coach dataset

- `user`: exact clinician practice attempt from the same section
- `assistant`: exact plain-text feedback paragraph(s) from `COACH FEEDBACK` blocks
- No inferred grades/scores
- No stringified JSON schema in assistant content

### Summative record

For each transcript, one additional Coach record is appended:

- `user`: `[Full Conversation History]`
- `assistant`: exact plain-text text from the `SUMMATIVE FEEDBACK` section

## Run

```bash
python training_data/transcripts/build_chatml_datasets.py
```

## Notes

- The script is deterministic (files processed in sorted filename order).
- The script overwrites both output `train.jsonl` files on each run.
