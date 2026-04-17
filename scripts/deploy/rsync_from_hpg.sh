#!/usr/bin/env bash
#
# rsync_from_hpg.sh
#
# Staging script: pull the trained artefacts from HiPerGator
# (/blue/jasondeanarnold/SPARCP/...) into a local staging directory on
# your laptop or the PubApp VM, preserving checksums. Run from a host
# that has SSH access to HiPerGator.
#
# Usage:
#   HPG_USER=jasondeanarnold \
#   HPG_HOST=hpg.rc.ufl.edu \
#   STAGING_DIR=/tmp/sparcp_staging \
#   ./rsync_from_hpg.sh
#
# After this completes, run layout_pubapp_models.sh on the PubApp VM
# (passing --source $STAGING_DIR) to materialize the /pubapps/SPARCP/...
# tree that sparc-backend.container expects.

set -euo pipefail

HPG_USER="${HPG_USER:?set HPG_USER to your HiPerGator GatorLink id}"
HPG_HOST="${HPG_HOST:-hpg.rc.ufl.edu}"
HPG_ROOT="${HPG_ROOT:-/blue/jasondeanarnold/SPARCP}"
STAGING_DIR="${STAGING_DIR:?set STAGING_DIR (e.g. /tmp/sparcp_staging)}"

RSYNC_OPTS=(
  -av
  --partial
  --progress
  --human-readable
  --info=stats2,progress2
)

mkdir -p "$STAGING_DIR"

# ---- 1. Base model (Llama 3.1 8B Instruct local weights, ~45 GB) -----------
# main.py picks this up via SPARC_BASE_MODEL; avoids a Hub download on boot.
echo "==> [1/4] Base model weights"
rsync "${RSYNC_OPTS[@]}" \
  "${HPG_USER}@${HPG_HOST}:${HPG_ROOT}/trained_models/meta_llama/Llama3.1-8B-Instruct/" \
  "${STAGING_DIR}/models/meta_llama/Llama3.1-8B-Instruct/"

# ---- 2. Caregiver adapter (Anne + Maya on synthetic_dataset-3000) ----------
# `CaregiverAgent/full/` is the production adapter — layout_pubapp_models.sh
# aliases it to `/pubapps/SPARCP/models/CaregiverAgent/` so main.py's default
# adapter path resolves. The sibling `full-d/`, `full-synthetic-d/`, and
# `full-llama-4-9-26/` directories are snapshots kept around for comparing
# against v3 / v4 dataset styles; they are pulled too (so you can swap at
# deploy time with CAREGIVER_RUN=...) but never aliased by default.
echo "==> [2/4] CaregiverAgent LoRA adapter (full, + A/B snapshots)"
rsync "${RSYNC_OPTS[@]}" --exclude='checkpoint-*' \
  "${HPG_USER}@${HPG_HOST}:${HPG_ROOT}/trained_models/live_jupyter_runs/CaregiverAgent/" \
  "${STAGING_DIR}/models/CaregiverAgent/"

# ---- 3. C-LEAR Coach adapter (train.jsonl) ---------------------------------
# If the coach run hasn't landed yet this will be skipped; main.py handles
# a missing optional adapter gracefully (see load_models()).
echo "==> [3/4] C-LEAR_CoachAgent LoRA adapter (optional)"
if ssh "${HPG_USER}@${HPG_HOST}" test -d "${HPG_ROOT}/trained_models/live_jupyter_runs/C-LEAR_CoachAgent"; then
  rsync "${RSYNC_OPTS[@]}" --exclude='checkpoint-*' \
    "${HPG_USER}@${HPG_HOST}:${HPG_ROOT}/trained_models/live_jupyter_runs/C-LEAR_CoachAgent/" \
    "${STAGING_DIR}/models/C-LEAR_CoachAgent/"
else
  echo "    (skipping — C-LEAR_CoachAgent not present on HiPerGator yet)"
fi

# ---- 4. Chroma RAG index (0.5.x on-disk format, ~3.5 GB) -------------------
# We rsync the CONTENTS of sparc_training_markdown_kb/ into the PubApp
# Chroma root so chromadb.PersistentClient(path=...) sees chroma.sqlite3
# at its top level. The collection name is preserved inside the sqlite.
echo "==> [4/4] Chroma RAG index (sparc_training_markdown_kb)"
rsync "${RSYNC_OPTS[@]}" \
  "${HPG_USER}@${HPG_HOST}:${HPG_ROOT}/trained_models/vector_db/sparc_training_markdown_kb/" \
  "${STAGING_DIR}/rag/chroma/"

echo
echo "Staging complete: ${STAGING_DIR}"
echo "Next: copy to the PubApp VM and run layout_pubapp_models.sh"
