#!/usr/bin/env bash
set -euo pipefail

LOCK="$HOME/.sparc-watchdog.lock"
exec 9>"$LOCK"
flock -n 9 || exit 0

HEALTH_URL="http://127.0.0.1:8081/health"
APP_DIR="/pubapps/jasondeanarnold/SPARCP/repo/v2"
CONDA_SH="/pubapps/jasondeanarnold/SPARCP/miniforge/etc/profile.d/conda.sh"
CONDA_ENV="/pubapps/jasondeanarnold/SPARCP/conda_envs/sparc_backend"
LOG="$HOME/sparc-backend.log"

# If healthy, do nothing
if curl -fsS --max-time 3 "$HEALTH_URL" >/dev/null 2>&1; then
  exit 0
fi

# If already starting/running, do nothing
if pgrep -f "uvicorn main:app --host 0.0.0.0 --port 8081" >/dev/null 2>&1; then
  exit 0
fi

cd "$APP_DIR"

# ── Email alert when server is found down ─────────────────────────────────────
# Rate-limited to one email per 10 minutes so a crash-loop doesn't spam the inbox.
ALERT_EMAIL="jayrosen@ufl.edu"
ALERT_COOLDOWN="$HOME/.sparc-alert-cooldown"
_now=$(date +%s)
_last=0
if [ -f "$ALERT_COOLDOWN" ]; then
  _last=$(cat "$ALERT_COOLDOWN" 2>/dev/null || echo 0)
  # guard against corrupt / non-numeric content
  [[ "$_last" =~ ^[0-9]+$ ]] || _last=0
fi
if [ $(( _now - _last )) -ge 600 ]; then
  printf '%s' "$_now" > "$ALERT_COOLDOWN" 2>/dev/null || true
  _hname=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo "unknown")
  _ts=$(date -u '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || date)
  printf 'The SPARC-P uvicorn server (port 8081) was found DOWN at %s on %s.\n\nThe watchdog is restarting it now.\nCheck the log at: %s\n' \
    "$_ts" "$_hname" "$LOG" \
    | mail -s "[SPARC-P] Server Down — Restarting" "$ALERT_EMAIL" 2>/dev/null || true
fi
# ─────────────────────────────────────────────────────────────────────────────

nohup /bin/bash -lc "
  source '$CONDA_SH' &&
  conda activate '$CONDA_ENV' &&
  export SPARC_MODEL_BASE_PATH=/pubapps/jasondeanarnold/SPARCP/models &&
  export SPARC_FIREBASE_CREDS=/pubapps/jasondeanarnold/SPARCP/config/firebase-credentials.json &&
  export SPARC_BASE_MODEL=/pubapps/jasondeanarnold/SPARCP/models/meta_llama/Llama3.1-8B-Instruct &&
  export SPARC_TTS_BACKEND=kokoro &&
  export SPARC_KOKORO_DEVICE=cuda &&
  export SPARC_USE_ADAPTERS=false &&
  export SPARC_ENABLE_GUARDRAILS=false &&
  export SPARC_ENABLE_RAG_CHAT=true &&
  export SPARC_RAG_PERSIST_DIR=/pubapps/jasondeanarnold/SPARCP/rag/chroma/sparc_curated_rag_kb &&
  export SPARC_RAG_COLLECTION=sparc_curated_rag_kb &&
  export SPARC_LLM_TIMEOUT_SECONDS=60 &&
  export SPARC_COACH_TIMEOUT_SECONDS=35 &&
  export SPARC_CHAT_MAX_INPUT_TOKENS=3500 &&
  export SPARC_COACH_MAX_INPUT_TOKENS=5500 &&
  export SPARC_VERBOSE_CHAT_LOGS=true &&
  export SPARC_VERBOSE_CHAT_LOG_PREVIEW_CHARS=1500 &&
  export HF_HOME=/pubapps/jasondeanarnold/SPARCP/hf_cache &&
  export HF_HUB_OFFLINE=1 &&
  export TRANSFORMERS_OFFLINE=1 &&
  export HF_ENABLE_PARALLEL_LOADING=false &&
  export HF_PARALLEL_LOADING_WORKERS=1 &&
  export HF_PARALLEL_LOADING_NUM_WORKERS=1 &&
  export TRANSFORMERS_LOADING_WORKERS=1 &&
  export SPARC_GUARDRAILS_DIR=/pubapps/jasondeanarnold/SPARCP/guardrails &&
  export SPARC_LLM_QUANTIZATION=none &&
  export SPARC_LLM_DTYPE=bfloat16 &&
  export SPARC_BACKEND_FIRESTORE_LOGS=true &&
  export SPARC_SKIP_CUDA_ALLOC_WARMUP=1 &&
  export SPARC_ALERT_EMAIL=jayrosen@ufl.edu &&
  export HF_TOKEN=\$(cat /pubapps/jasondeanarnold/.hf_token) &&
  export HUGGING_FACE_HUB_TOKEN=\$HF_TOKEN &&
  exec uvicorn main:app --host 0.0.0.0 --port 8081
" >> "$LOG" 2>&1 &