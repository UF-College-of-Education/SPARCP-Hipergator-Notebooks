#!/usr/bin/env bash
#
# layout_pubapp_models.sh
#
# Runs on the PubApp VM. Takes a staging tree produced by rsync_from_hpg.sh
# and materializes the /pubapps/SPARCP/ tree that sparc-backend.container
# expects.
#
# Key responsibilities:
#   1. Move base model weights into place.
#   2. Pick the correct Caregiver checkpoint (full-synthetic-d — Anne + Maya
#      trained on synthetic_dataset-3000-v2) and alias it as CaregiverAgent/.
#   3. Pick the correct Coach checkpoint and alias it as C-LEAR_CoachAgent/.
#      (Silently skipped if the upstream adapter isn't on disk yet.)
#   4. Install the Chroma index.
#   5. Install the guardrails config from this repo (if present).
#   6. Create the audio cache directory.
#
# Usage (run as a user with write access to /pubapps/SPARCP):
#   sudo -u pubapps ./layout_pubapp_models.sh --source /tmp/sparcp_staging
#
# Re-running is idempotent; existing symlinks are recreated atomically.

set -euo pipefail

STAGING_DIR=""
PUBAPPS_ROOT="${PUBAPPS_ROOT:-/pubapps/SPARCP}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# Which Caregiver run to treat as the "production" Caregiver adapter.
# `full/` = canonical Anne + Maya adapter trained on synthetic_dataset-3000
# (the *-v2 file). Siblings like `full-d/`, `full-synthetic-d/`, and
# `full-llama-4-9-26/` are A/B snapshots kept around for comparing output
# styles against v3 / v4 datasets — do not deploy those by default.
CAREGIVER_RUN="${CAREGIVER_RUN:-full}"
COACH_RUN="${COACH_RUN:-full}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) STAGING_DIR="$2"; shift 2 ;;
    --caregiver-run) CAREGIVER_RUN="$2"; shift 2 ;;
    --coach-run) COACH_RUN="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$STAGING_DIR" ]]; then
  echo "ERROR: --source <staging_dir> is required" >&2
  exit 1
fi

install_symlink() {
  local src="$1" dst="$2"
  rm -rf "$dst"
  ln -sfn "$src" "$dst"
}

mkdir -p \
  "$PUBAPPS_ROOT/models" \
  "$PUBAPPS_ROOT/models/meta_llama" \
  "$PUBAPPS_ROOT/rag" \
  "$PUBAPPS_ROOT/guardrails" \
  "$PUBAPPS_ROOT/config" \
  "$PUBAPPS_ROOT/audio_cache"

# ---- 1. Base model --------------------------------------------------------
echo "==> Installing base model weights"
rsync -a --delete \
  "$STAGING_DIR/models/meta_llama/Llama3.1-8B-Instruct/" \
  "$PUBAPPS_ROOT/models/meta_llama/Llama3.1-8B-Instruct/"

# ---- 2. Caregiver adapter -------------------------------------------------
CAREGIVER_SRC="$STAGING_DIR/models/CaregiverAgent/$CAREGIVER_RUN"
if [[ ! -f "$CAREGIVER_SRC/adapter_config.json" ]]; then
  echo "ERROR: Caregiver adapter not found at $CAREGIVER_SRC" >&2
  echo "       Production default is CAREGIVER_RUN=full." >&2
  echo "       A/B comparison runs available on HPG: full-d, full-synthetic-d, full-llama-4-9-26" >&2
  exit 2
fi
echo "==> Installing Caregiver adapter from $CAREGIVER_SRC"
rsync -a --delete "$CAREGIVER_SRC/" "$PUBAPPS_ROOT/models/CaregiverAgent/"

# ---- 3. Coach adapter (optional) ------------------------------------------
COACH_SRC_ROOT="$STAGING_DIR/models/C-LEAR_CoachAgent"
COACH_SRC="$COACH_SRC_ROOT/$COACH_RUN"
if [[ -f "$COACH_SRC/adapter_config.json" ]]; then
  echo "==> Installing Coach adapter from $COACH_SRC"
  rsync -a --delete "$COACH_SRC/" "$PUBAPPS_ROOT/models/C-LEAR_CoachAgent/"
else
  echo "==> Coach adapter not available — /v1/chat will return coach_feedback_meta.reason=coach_adapter_unavailable"
  rm -rf "$PUBAPPS_ROOT/models/C-LEAR_CoachAgent"
fi

# ---- 4. Chroma RAG index --------------------------------------------------
echo "==> Installing Chroma index"
rsync -a --delete "$STAGING_DIR/rag/chroma/" "$PUBAPPS_ROOT/rag/chroma/"

# ---- 5. Guardrails config from this repo ----------------------------------
if [[ -d "$REPO_ROOT/scripts/deploy/guardrails" ]]; then
  echo "==> Installing NeMo Guardrails config"
  rsync -a --delete \
    "$REPO_ROOT/scripts/deploy/guardrails/" \
    "$PUBAPPS_ROOT/guardrails/"
else
  echo "    (scripts/deploy/guardrails not found in repo — leaving $PUBAPPS_ROOT/guardrails untouched)"
fi

# ---- 6. Audio cache -------------------------------------------------------
chmod 775 "$PUBAPPS_ROOT/audio_cache"

echo
echo "Layout complete:"
printf '  base_model      : %s\n' "$PUBAPPS_ROOT/models/meta_llama/Llama3.1-8B-Instruct"
printf '  caregiver       : %s  (from %s)\n' "$PUBAPPS_ROOT/models/CaregiverAgent" "$CAREGIVER_RUN"
if [[ -d "$PUBAPPS_ROOT/models/C-LEAR_CoachAgent" ]]; then
  printf '  coach           : %s  (from %s)\n' "$PUBAPPS_ROOT/models/C-LEAR_CoachAgent" "$COACH_RUN"
fi
printf '  rag             : %s\n' "$PUBAPPS_ROOT/rag/chroma"
printf '  guardrails      : %s\n' "$PUBAPPS_ROOT/guardrails"
printf '  firebase (manual): %s\n' "$PUBAPPS_ROOT/config/firebase-credentials.json"
