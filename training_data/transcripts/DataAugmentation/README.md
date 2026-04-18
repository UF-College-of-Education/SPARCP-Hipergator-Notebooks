# SPARC Data Augmentation

This tool generates synthetic variations of clinical transcripts for C-LEAR skills practice. It reads your "base" transcripts and uses an LLM pipeline to generate numerous stylistically different variations while strictly preserving the markdown formatting and the specific clinician performance level.

## 1. Prerequisites

Ensure you have Conda installed. Run the conda setup script and activate it

```bash
bashs setup_conda_env.sh
conda activate sparc_augmentation
```

Set your Navigator API credentials as environment variables in your terminal:

```bash
export OPENAI_API_KEY="your-navigator-api-key"
export OPENAI_API_BASE="https://api.ai.it.ufl.edu/v1" # Optional if you need a custom endpoint
```

## 2. Prepare Your Base Data

The script needs examples to work from. 

1. Ensure the base data directory exists (the script will also create this for you if you run it once):
   ```bash
   mkdir -p training_data/base_transcripts/1st_skills
   ```
2. Drop your "base" `.md` transcript files into the `training_data/base_transcripts/1st_skills/` folder. The script will generate variations for every file it finds in this directory.

## 3. Run the Script

Execute the pipeline:

```bash
python generate_transcripts.py
```

The script runs concurrently and will output the newly generated synthetic transcripts into the `training_data/synthetic_markdown/` folder.
