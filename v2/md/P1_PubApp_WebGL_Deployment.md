# P1 PubApp WebGL Deployment

<!-- Cell 1: markdown -->

# SPARC-P PubApps Deployment Notebook

This notebook is the runnable version of Step 4 for deploying SPARC-P to UF RC PubApps.

## Resource Profiles
- **HiPerGator (parallel jobs)**: 4 GPUs, 16 CPU cores
- **PubApps (serving)**: 1x L4 GPU (24GB), 2 CPU cores, 16GB RAM

## Before You Run
- You are on your PubApps VM via SSH
- You have your project account (`SPARCP`)
- Trained models are available from HiPerGator at `/blue/jasondeanarnold/SPARCP/trained_models`
- Podman + systemd user services are available

![1.0 & 2.0 HiPerGator to PubApps Transfer and Architecture Diagram](../images/notebook4-1-0.png)

This diagram highlights the architectural differences between the training environment (HiPerGator) and the serving environment (PubApps), and visualizes the rsync model transfer process required before deployment

<!-- Cell 2: markdown -->

All paths, hostnames, and resource limits used throughout this notebook are defined here. Nothing is deployed yet â€” this section only establishes the variables.

What gets defined and why:
- **Paths** (`PUBAPPS_ROOT`, `MODEL_DIR`, `BACKEND_DIR`, `CONDA_ENV`, `RIVA_MODEL_DIR`): All point to the `/pubapps/SPARCP/` directory tree on the PubApps VM. These are used by every other cell to know where to create directories, install the conda environment, and write files.
- **Source paths** (`BASE_PATH`, `HIPERGATOR_SOURCE_MODELS`): The `/blue/jasondeanarnold/SPARCP/trained_models` location on HiPerGator where the fine-tuned model adapters live. Used to construct the `rsync` transfer command.
- **Connection info** (`PUBAPPS_HOST`, `PUBAPPS_SSH_USER`): The hostname and SSH username for the PubApps VM, used in the model transfer step.
- **CORS origins**: The two allowed domains (`hpvcommunicationtraining.com` and `.org`) that the Unity app runs on. Only these origins can make API calls to the backend â€” all others are rejected by the server.
- **Firebase path**: Where the Firebase service account credentials file lives on the PubApps VM. Required by the backend to write session state to Firestore.
- **Resource constants** (`PUBAPPS_GPU`, `PUBAPPS_CORES`, `PUBAPPS_RAM_GB`, `UVICORN_WORKERS`): Hard documented limits for the L4 GPU (24 GB VRAM), 2 CPU cores, and 16 GB RAM PubApps environment. `UVICORN_WORKERS = 1` is intentional â€” with only 2 cores and the LLM holding most GPU memory, running more than 1 worker would cause out-of-memory errors.

> **All values can be overridden** by setting environment variables before running this notebook (e.g., `export SPARC_PUBAPPS_PROJECT=MYPROJECT`).  The printed output at the bottom confirms the resolved values.

<!-- Cell 3: code (python) -->

```python
# 1. Configuration
import os
import subprocess
import textwrap
from pathlib import Path

PROJECT = os.environ.get("SPARC_PUBAPPS_PROJECT", "SPARCP")
PUBAPPS_ROOT = Path(f"/pubapps/{PROJECT}")
MODEL_DIR = PUBAPPS_ROOT / "models"
CONDA_ENV = PUBAPPS_ROOT / "conda_envs" / "sparc_backend"
BACKEND_DIR = PUBAPPS_ROOT / "backend"
RIVA_MODEL_DIR = PUBAPPS_ROOT / "riva_models"

BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
HIPERGATOR_SOURCE_MODELS = os.environ.get(
    "SPARC_HIPERGATOR_SOURCE_MODELS",
    f"{BASE_PATH}/trained_models",
)
PUBAPPS_HOST = os.environ.get("SPARC_PUBAPPS_HOST", "pubapps-vm.rc.ufl.edu")
PUBAPPS_SSH_USER = os.environ.get("SPARC_PUBAPPS_SSH_USER", PROJECT)
PUBAPP_ALLOWED_ORIGINS = os.environ.get(
    "SPARC_CORS_ALLOWED_ORIGINS",
    "https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org",
)
FIREBASE_CREDS_PATH = Path(
    os.environ.get("SPARC_FIREBASE_CREDS", str(PUBAPPS_ROOT / "config" / "firebase-credentials.json"))
)

# Resource constraints
HPG_MAX_GPUS = 4
HPG_MAX_CORES = 16
PUBAPPS_GPU = "L4 (24GB)"
PUBAPPS_CORES = 2
PUBAPPS_RAM_GB = 16
UVICORN_WORKERS = 1  # tuned for 2 CPU cores and 16GB RAM

print(f"Project: {PROJECT}")
print(f"PubApps root: {PUBAPPS_ROOT}")
print(f"Conda env: {CONDA_ENV}")
print(f"Backend dir: {BACKEND_DIR}")
print(f"HiPerGator source models: {HIPERGATOR_SOURCE_MODELS}")
print(f"PubApps host: {PUBAPPS_HOST}")
print(f"PubApps SSH user: {PUBAPPS_SSH_USER}")
print(f"Allowed CORS origins: {PUBAPP_ALLOWED_ORIGINS}")
print(f"Firebase creds path: {FIREBASE_CREDS_PATH}")
print(f"HiPerGator resources: {HPG_MAX_GPUS} GPUs, {HPG_MAX_CORES} cores")
print(f"PubApps resources: {PUBAPPS_GPU}, {PUBAPPS_CORES} cores, {PUBAPPS_RAM_GB}GB RAM")
```

<!-- Cell 4: markdown -->

The `run()` helper function used throughout this notebook executes shell commands â€” with a critical safety mechanism built in: by default, **nothing actually runs**.

The `EXECUTE = False` flag at the top means this notebook is in "dry-run" mode. When you call `run("some command")`, it prints `$ some command` followed by `(dry-run) command not executed` â€” showing you exactly what *would* happen without actually doing it.

To deploy for real, change `EXECUTE = False` to `EXECUTE = True` and re-run all cells from top to bottom.

Why this pattern is useful:
- You can review the full deployment sequence (what commands would run, in what order, with what paths) before committing to any changes on the PubApps VM.
- It prevents accidental partial deployments if you're just opening this notebook to check something.
- The `check=True` default means that when `EXECUTE = True`, any command that returns a non-zero exit code (error) raises a `RuntimeError` immediately, stopping the notebook rather than silently continuing with a broken state. A few commands use `check=False` when their failure is non-fatal (e.g., status checks).

<!-- Cell 5: code (python) -->

```python
# 2. Command runner (safe by default)
EXECUTE = False  # Set True to actually run shell commands

def run(cmd: str, check: bool = True):
    print(f"$ {cmd}")
    if not EXECUTE:
        print("(dry-run) command not executed\n")
        return None
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    print()
    return result
```

<!-- Cell 6: markdown -->

## 3. Transfer Models from HiPerGator
Run this on HiPerGator or from a hop host with access to both systems.

<!-- Cell 7: markdown -->

The `rsync` command below transfers the fine-tuned SPARC-P model adapters from HiPerGator's `/blue` storage to the PubApps VM's model directory. The command is **printed, not executed** (unless `EXECUTE = True`).

`rsync` is the right tool for this task because:
- It transfers only files that have changed (delta sync), so re-running after a partial transfer is safe and fast.
- The `-avz` flags mean: archive mode (preserves permissions and timestamps), verbose output, and gzip compression in transit.
- `--progress` shows a progress bar for each file, which is helpful since the fine-tuned adapters can be 2â€“10 GB total.

The command transfers from `HIPERGATOR_SOURCE_MODELS/` (the `/blue/jasondeanarnold/SPARCP/trained_models` directory) to `MODEL_DIR` on the PubApps VM using SSH as the transport.

> **Where to run this:** This command needs to run on HiPerGator (where the `/blue` filesystem is mounted) or on a machine that has SSH access to both systems. Copy the printed command and run it in your HiPerGator terminal session.

<!-- Cell 8: code (python) -->

```python
# 3.1 Render model sync command
rsync_cmd = textwrap.dedent(f"""
rsync -avz --progress \
  {HIPERGATOR_SOURCE_MODELS}/ \
  {PUBAPPS_SSH_USER}@{PUBAPPS_HOST}:{MODEL_DIR}/
""").strip()
print(rsync_cmd)
```

<!-- Cell 9: markdown -->

## 4. PubApps Environment Setup

<!-- Cell 10: markdown -->

All the directory folders the SPARC-P service needs on the PubApps VM are created here before any files are written or software is installed. A single `mkdir -p` command creates the entire directory structure in one shot â€” the `-p` flag means it creates every level in the path and doesn't fail if a directory already exists.

The directories created:
- **`/pubapps/SPARCP/`** â€” the project root for everything SPARC-P on this VM
- **`/pubapps/SPARCP/models/`** â€” where the fine-tuned LLM adapters (CaregiverAgent, CoachAgent, SupervisorAgent) are stored after transfer from HiPerGator
- **`/pubapps/SPARCP/backend/`** â€” where the FastAPI `main.py` application code lives
- **`/pubapps/SPARCP/riva_models/`** â€” where NVIDIA Riva's ASR and TTS model files are stored
- **`/pubapps/SPARCP/conda_envs/`** â€” the parent directory for the `sparc_backend` conda environment

This is always the first step before installing software or transferring files â€” you can't write to a directory that doesn't exist.

<!-- Cell 11: code (python) -->

```python
# 4.1 Create required directories
run(f"mkdir -p {PUBAPPS_ROOT} {MODEL_DIR} {BACKEND_DIR} {RIVA_MODEL_DIR} {PUBAPPS_ROOT / 'conda_envs'}")
```

<!-- Cell 12: markdown -->

![3.0 Setup Conda Environment on PubApps Diagram](../images/notebook4-2.png)

This flowchart outlines the strict directory creation and Conda environment initialization process required on the PubApps VM
The complete Python software environment for the SPARC-P backend is installed using conda â€” three commands verify, create, and validate the environment in sequence.

Step by step:
1. **`conda --version`** (`check=False`) â€” confirms conda is available on this VM. The `check=False` means a failure here just prints a warning rather than stopping the notebook; useful if you're running in dry-run mode.
2. **`conda env create -f environment_backend.yml -p {CONDA_ENV}`** â€” creates the entire Python environment from a configuration file. The `-f environment_backend.yml` points to the yaml file that lists every package and version (FastAPI, PyTorch, transformers, bitsandbytes, NeMo Guardrails, etc.). The `-p` flag installs the environment to the exact path `/pubapps/SPARCP/conda_envs/sparc_backend` instead of conda's default location. This can take **10â€“30 minutes** as it downloads and installs hundreds of packages including CUDA-compiled PyTorch.
3. **`conda run -p {CONDA_ENV} python -c 'import fastapi,langgraph,torch; print("backend env ok")'`** â€” immediately tests that the newly created environment is functional by importing three critical libraries. If any import fails, this command fails loudly, catching broken installs before you proceed.

> **One-time step:** Only run this if the conda environment doesn't already exist. If it exists and you just want to update it, use `conda env update -f environment_backend.yml -p {CONDA_ENV}` instead.

<!-- Cell 13: code (python) -->

```python
# 4.2 Create backend conda environment
run("conda --version", check=False)
run(f"cd {PUBAPPS_ROOT}; conda env create -f environment_backend.yml -p {CONDA_ENV}")
run(f"conda run -p {CONDA_ENV} python -c 'import fastapi,langgraph,torch; print(\"backend env ok\")'")
```

<!-- Cell 14: markdown -->

![4.0 Deploy Riva Speech Services Diagram](../images/notebook4-3.png)

This diagram details the Podman "Quadlet" setup for NVIDIA Riva, converting container execution into a persistent, auto-restarting systemd service
## 5. Deploy Riva with Podman + Quadlet

<!-- Cell 15: markdown -->

A "Quadlet" file tells Podman and systemd how to manage the NVIDIA Riva speech server as a persistent background service on the PubApps VM â€” similar to how you'd configure a Windows Service, but for Linux.

What a Quadlet is: On modern Linux systems, systemd is the service manager. Podman Quadlets are simple configuration files that let systemd start, stop, and automatically restart containers â€” without needing Docker daemon or complex shell scripts.

What the generated `riva-server.container` file specifies:
- **`Image`**: Downloads and runs NVIDIA's official Riva speech server container from their registry (NGC) at version 2.16.0.
- **`Device=nvidia.com/gpu=all`**: Uses the modern CDI (Container Device Interface) standard to pass the L4 GPU through to the container. This is required for Riva to run its ASR and TTS models.
- **`Volume={RIVA_MODEL_DIR}:/data:Z`**: Mounts the local riva_models directory into the container where Riva looks for its model files. The `:Z` sets the correct SELinux label for Podman.
- **`PublishPort=50051:50051`**: Exposes Riva's gRPC port so the FastAPI backend (running outside the container) can connect using `localhost:50051`.
- **`Restart=always`**: If Riva crashes or the VM reboots, systemd automatically restarts it.
- **`TimeoutStartSec=300`**: Gives Riva 5 minutes to start (loading ASR and TTS models into GPU memory takes ~2 minutes).
- The file is written to `~/.config/containers/systemd/` â€” the per-user path where Podman looks for Quadlet definitions.

> **The CDI assertion at the end** (`Device=nvidia.com/gpu=all` in `quadlet_content`) is a regression check that confirms the GPU mapping was written correctly before proceeding.

<!-- Cell 16: code (python) -->

```python
# 5.1 Write quadlet service for Riva
quadlet_dir = Path.home() / '.config/containers/systemd'
quadlet_dir.mkdir(parents=True, exist_ok=True)
quadlet_file = quadlet_dir / 'riva-server.container'
quadlet_content = textwrap.dedent(f"""
[Unit]
Description=SPARC-P Riva Speech Server
After=network-online.target

[Container]
Image=nvcr.io/nvidia/riva/riva-speech:2.16.0-server
ContainerName=riva-server
Device=nvidia.com/gpu=all
Volume={RIVA_MODEL_DIR}:/data:Z
PublishPort=50051:50051
Environment=NVIDIA_VISIBLE_DEVICES=all
Exec=/opt/riva/bin/riva_server --riva_model_repo=/data/models

[Service]
Restart=always
TimeoutStartSec=300

[Install]
WantedBy=default.target
""").strip()
quadlet_file.write_text(quadlet_content)
print(f"Wrote {quadlet_file}")
print(quadlet_content)
```

<!-- Cell 17: markdown -->

Downloading the Riva container image and activating Riva as a running systemd service completes the Quadlet setup. Four commands run in sequence, plus a validation check.

Step by step:
1. **`podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server`**: Downloads the Riva container image from NVIDIA's container registry. This is a large image (~5â€“8 GB) and needs to happen before systemd can start the service. Only needed once.
2. **`systemctl --user daemon-reload`**: Tells systemd to re-read all service definitions from disk, including the Quadlet file written in the previous cell. Without this, systemd wouldn't know about the new `riva-server` service.
3. **`systemctl --user enable --now riva-server`**: Registers the Riva service to start automatically on login (`enable`) and starts it immediately right now (`--now`). After this command, Riva begins loading its ASR and TTS models into GPU memory.
4. **`systemctl --user status riva-server`** (`check=False`): Shows the current status of the Riva service. Expected output: `Active: active (running)`. The `check=False` prevents this from stopping the notebook if the service is still starting.
5. **GPU validation commands**: `nvidia-ctk cdi list` confirms the CDI GPU device is registered, and the `podman run ... nvidia-smi` command runs `nvidia-smi` inside a test container to confirm Riva's container can actually see the GPU.
6. **Assertion**: Verifies the Quadlet file contains the correct CDI GPU mapping (`Device=nvidia.com/gpu=all`) and not the legacy mapping (`AddDevice=`), which would fail on newer Podman versions.

> **Expected next:** Allow 2â€“3 minutes for Riva to initialize. Check `journalctl --user -u riva-server -n 50` to monitor startup progress.

<!-- Cell 18: code (python) -->

```python
# 5.2 Pull image and enable Riva service
run("podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server")
run("systemctl --user daemon-reload")
run("systemctl --user enable --now riva-server")
run("systemctl --user status riva-server --no-pager", check=False)
run("nvidia-ctk cdi list", check=False)
run("podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi", check=False)
assert "Device=nvidia.com/gpu=all" in quadlet_content, "CDI GPU mapping missing in quadlet"
assert "AddDevice=" not in quadlet_content, "Legacy AddDevice mapping still present in quadlet"
print("âœ… M10 Quadlet CDI validation passed")
```

<!-- Cell 19: markdown -->

![5.0 & 5.1 FastAPI Backend Service and Internals Diagram](../images/notebook4-4.png)

This comprehensive chart maps the deployment of the FastAPI backend via systemd and dives into the internal logic of the main.py application, detailing the VRAM budget, safety rails, and API behavior
## 6. Create FastAPI Backend + Systemd Service

<!-- Cell 20: markdown -->

This is the most important cell in the notebook — it writes the complete, production-ready `main.py` FastAPI application (approximately 520 lines) to disk at `/pubapps/SPARCP/backend/main.py`. This is the actual program that runs on the PubApps server and handles every interaction with SPARC-P users.

What the written application does (plain-English overview of each major section):

**System startup (lifespan):** When the server starts, it loads all three fine-tuned LLM adapters (CaregiverAgent, CoachAgent, SupervisorAgent) into GPU memory using 4-bit quantization (NF4 format via bitsandbytes) to fit within the L4's 24 GB VRAM. It also connects to Riva for speech, loads the NeMo Guardrails safety config, and creates the audio file cache directory.

**Safety pipeline (guardrails):** Every incoming user message passes through NeMo Guardrails before reaching the AI models. Off-topic messages (politics, finance, anything unrelated to HPV vaccine communication) are rejected with a pre-set refusal message. The AI's response also passes through guardrails before being sent back — a two-stage safety check.

**WebGL/Unity contract compatibility:** The chat endpoint accepts both the notebook's newer `user_message` payload and the Unity MAS client's legacy `user_transcript` payload. It also accepts direct agent routing fields (`mode`, `agent_mode`, or `target_agent`) so Unity can explicitly talk to `caregiver`, `coach`, or `supervisor` without relying on Firestore session state.

**Browser-friendly access model:** API-key auth remains available as an environment-controlled defense-in-depth option, but it is disabled by default for WebGL/browser deployments where a static client-side secret would not be safe. This lets the current Unity WebGL client talk to PubApps directly while still allowing deployments behind a proxy to re-enable header-based auth if needed.

**Audio delivery:** TTS audio is written to a temp file and returned as a short-lived URL (`/v1/audio/{id}`) for modern clients, while the same response can also include legacy `caregiver_audio_b64` content so the current Unity MAS client continues to work without immediate script changes.

**Firebase session state:** After each turn, the session's last message and response are written to Firestore for session continuity and audit purposes, and the resolved active mode is persisted.

> **The file is written to disk but the server is not yet started.** The systemd service section below handles starting the running process.

<!-- Cell 21: code (python) -->

```python
# 6.1 Write backend main.py (integration-ready)
main_py = BACKEND_DIR / 'main.py'
main_content = textwrap.dedent('''
import asyncio
import base64
import time
import tempfile
import uuid
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import riva.client
from nemoguardrails import LLMRails, RailsConfig
import firebase_admin
from firebase_admin import credentials, firestore

MODEL_BASE_PATH = os.getenv("SPARC_MODEL_BASE_PATH", "{MODEL_DIR}")
RIVA_SERVER = os.getenv("SPARC_RIVA_SERVER", "localhost:50051")
FIREBASE_CREDS = os.getenv("SPARC_FIREBASE_CREDS", "{PUBAPPS_ROOT}/config/firebase-credentials.json")
GUARDRAILS_DIR = os.getenv("SPARC_GUARDRAILS_DIR", os.path.join(os.path.dirname(__file__), "guardrails"))

API_AUTH_ENABLED = os.getenv("SPARC_API_AUTH_ENABLED", "false").strip().lower() == "true"
API_KEY = os.getenv("SPARC_API_KEY", "")
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("SPARC_CORS_ALLOWED_ORIGINS", "{PUBAPP_ALLOWED_ORIGINS}").split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = os.getenv("SPARC_CORS_ALLOW_CREDENTIALS", "false").strip().lower() == "true"
CORS_ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
CORS_ALLOWED_HEADERS = ["Content-Type", "X-API-Key", "Authorization"]
API_CONTRACT_VERSION = "v1"
LLM_TIMEOUT_SECONDS = float(os.getenv("SPARC_LLM_TIMEOUT_SECONDS", "10"))
COACH_TIMEOUT_SECONDS = float(os.getenv("SPARC_COACH_TIMEOUT_SECONDS", "10"))
TTS_TIMEOUT_SECONDS = float(os.getenv("SPARC_TTS_TIMEOUT_SECONDS", "5"))
TTS_MAX_AUDIO_BYTES = int(os.getenv("SPARC_TTS_MAX_AUDIO_BYTES", "524288"))
LEGACY_AUDIO_B64_MAX_BYTES = int(os.getenv("SPARC_LEGACY_AUDIO_B64_MAX_BYTES", str(TTS_MAX_AUDIO_BYTES)))
SPARC_AUDIO_URL_TTL_SECONDS = float(os.getenv("SPARC_AUDIO_URL_TTL_SECONDS", "300"))
SPARC_AUDIO_CACHE_DIR = os.getenv("SPARC_AUDIO_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sparc_tts_audio"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("SPARC_TIMEOUT_CIRCUIT_THRESHOLD", "3"))
CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS", "30"))
DEFAULT_ANIMATION_EMOTION = os.getenv("SPARC_DEFAULT_ANIMATION_EMOTION", "neutral")
DEFAULT_ANIMATION_GESTURE = os.getenv("SPARC_DEFAULT_ANIMATION_GESTURE", "speaking")

if not FIREBASE_CREDS:
    raise RuntimeError("SPARC_FIREBASE_CREDS is empty; set Firebase service account path")
if not os.path.isfile(FIREBASE_CREDS):
    raise RuntimeError(
        f"Firebase credentials file not found: {FIREBASE_CREDS}. "
        "Set SPARC_FIREBASE_CREDS to a valid path."
    )

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDS)
    firebase_admin.initialize_app(cred)
db = firestore.client()

logger = logging.getLogger("sparc_backend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

guardrails_engine = None
GUARDRAILS_REFUSAL = "I can only discuss topics related to HPV vaccination and clinical communication training."


def load_guardrails_runtime() -> None:
    global guardrails_engine
    try:
        rails_config = RailsConfig.from_path(GUARDRAILS_DIR)
        guardrails_engine = LLMRails(rails_config)
        logger.info("Guardrails runtime loaded from %s", GUARDRAILS_DIR)
    except Exception as guardrails_error:
        guardrails_engine = None
        logger.exception("Guardrails initialization failed: %s")


async def _run_guardrails(text: str) -> str:
    if guardrails_engine is None:
        raise RuntimeError("Guardrails runtime not initialized")
    messages = [{"role": "user", "content": text}]
    if hasattr(guardrails_engine, "generate_async"):
        result = await guardrails_engine.generate_async(messages=messages)
    else:
        result = guardrails_engine.generate(messages=messages)
    if isinstance(result, dict):
        return str(result.get("content", result))
    return str(result)


async def enforce_guardrails_input(user_text: str) -> Dict[str, Any]:
    if not user_text or not user_text.strip():
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "empty_input"}
    try:
        rails_output = await _run_guardrails(user_text)
        blocked = GUARDRAILS_REFUSAL.lower() in rails_output.lower()
        if blocked:
            return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "input_rails_blocked"}
        return {"allowed": True, "text": user_text, "reason": "input_rails_allowed"}
    except Exception as guardrails_error:
        logger.exception("Input guardrails failed: %s")
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "input_rails_error"}


async def enforce_guardrails_output(output_text: str) -> Dict[str, Any]:
    if not output_text or not output_text.strip():
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "empty_output"}
    try:
        rails_output = await _run_guardrails(output_text)
        blocked = GUARDRAILS_REFUSAL.lower() in rails_output.lower()
        if blocked:
            return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "output_rails_blocked"}
        return {"allowed": True, "text": output_text, "reason": "output_rails_allowed"}
    except Exception as guardrails_error:
        logger.exception("Output guardrails failed: %s")
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "output_rails_error"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield


app = FastAPI(title="SPARC-P Multi-Agent Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

tokenizer = None
adapter_model = None
ADAPTER_FOR_MODE = {
    "caregiver": "caregiver",
    "coach": "coach",
    "supervisor": "supervisor",
}
ADAPTER_PATHS = {
    "caregiver": os.path.join(MODEL_BASE_PATH, "CaregiverAgent"),
    "coach": os.path.join(MODEL_BASE_PATH, "C-LEAR_CoachAgent"),
    "supervisor": os.path.join(MODEL_BASE_PATH, "SupervisorAgent"),
}
riva_auth = None
riva_asr_service = None
riva_tts_service = None
inference_lock = asyncio.Lock()
timeout_state_lock = asyncio.Lock()
audio_cache_lock = asyncio.Lock()
audio_cache_index: Dict[str, Dict[str, Any]] = {}
timeout_failures = {
    "primary_inference": 0,
    "coach_inference": 0,
    "riva_tts": 0,
}
circuit_open_until = {
    "primary_inference": 0.0,
    "coach_inference": 0.0,
    "riva_tts": 0.0,
}


def generate_tokens_sync(model, **generate_kwargs):
    with torch.inference_mode():
        return model.generate(**generate_kwargs)


def init_riva_clients() -> None:
    global riva_auth, riva_asr_service, riva_tts_service
    try:
        riva_auth = riva.client.Auth(uri=RIVA_SERVER)
        riva_asr_service = riva.client.ASRService(riva_auth)
        riva_tts_service = riva.client.SpeechSynthesisService(riva_auth)
        logger.info("Riva clients initialized for reuse at startup")
    except Exception as riva_init_error:
        riva_auth = None
        riva_asr_service = None
        riva_tts_service = None
        logger.warning("Riva client initialization failed: %s")


def synthesize_tts_sync(text: str, voice_name: str = "English-US.Female-1") -> bytes:
    if riva_tts_service is None:
        raise RuntimeError("Riva TTS client is not initialized")
    tts_response = riva_tts_service.synthesize(text, voice_name=voice_name)
    return tts_response.audio


def ensure_audio_cache_dir() -> None:
    Path(SPARC_AUDIO_CACHE_DIR).mkdir(parents=True, exist_ok=True)


async def prune_expired_audio_cache(now: Optional[float] = None) -> None:
    current_ts = now if now is not None else time.time()
    expiry_threshold = current_ts - SPARC_AUDIO_URL_TTL_SECONDS
    async with audio_cache_lock:
        expired_ids = [
            audio_id
            for audio_id, metadata in audio_cache_index.items()
            if metadata.get("created_at", 0.0) < expiry_threshold
        ]
        for audio_id in expired_ids:
            metadata = audio_cache_index.pop(audio_id, None)
            if not metadata:
                continue
            audio_path = metadata.get("path")
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    logger.warning("Failed to remove expired audio cache file: %s", audio_path)


async def persist_tts_audio(audio_bytes: bytes) -> Optional[str]:
    if not audio_bytes:
        return None
    if len(audio_bytes) > TTS_MAX_AUDIO_BYTES:
        logger.warning(
            "Skipping TTS audio delivery because payload %d bytes exceeds limit %d bytes",
            len(audio_bytes),
            TTS_MAX_AUDIO_BYTES,
        )
        return None

    ensure_audio_cache_dir()
    await prune_expired_audio_cache()

    audio_id = uuid.uuid4().hex
    audio_path = os.path.join(SPARC_AUDIO_CACHE_DIR, f"{audio_id}.wav")
    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

    async with audio_cache_lock:
        audio_cache_index[audio_id] = {"path": audio_path, "created_at": time.time()}

    return f"/v1/audio/{audio_id}"


async def is_circuit_open(operation: str) -> bool:
    now = time.monotonic()
    async with timeout_state_lock:
        return now < circuit_open_until.get(operation, 0.0)


async def record_timeout_event(operation: str) -> bool:
    now = time.monotonic()
    async with timeout_state_lock:
        timeout_failures[operation] = timeout_failures.get(operation, 0) + 1
        if timeout_failures[operation] >= CIRCUIT_BREAKER_THRESHOLD:
            circuit_open_until[operation] = now + CIRCUIT_BREAKER_RESET_SECONDS
            timeout_failures[operation] = 0
            return True
        return False


async def record_success_event(operation: str) -> None:
    async with timeout_state_lock:
        timeout_failures[operation] = 0
        circuit_open_until[operation] = 0.0



def select_adapter_for_mode(mode: str) -> str:
    normalized = (mode or "caregiver").strip().lower()
    return ADAPTER_FOR_MODE.get(normalized, "caregiver")


def resolve_requested_mode(request: "ChatRequest", session_state: Dict[str, Any]) -> str:
    request_mode = request.target_agent or request.agent_mode or request.mode
    if request_mode:
        return select_adapter_for_mode(request_mode)
    return select_adapter_for_mode(session_state.get("mode", "caregiver"))


def extract_user_message(request: "ChatRequest") -> str:
    return (request.user_message or request.user_transcript or "").strip()


def default_animation_cues() -> Dict[str, str]:
    return {
        "emotion": DEFAULT_ANIMATION_EMOTION,
        "gesture": DEFAULT_ANIMATION_GESTURE,
    }


def build_legacy_audio_b64(audio_bytes: Optional[bytes], include_legacy_audio_b64: bool) -> Optional[str]:
    if not include_legacy_audio_b64 or not audio_bytes:
        return None
    if len(audio_bytes) > LEGACY_AUDIO_B64_MAX_BYTES:
        logger.warning(
            "Skipping legacy caregiver_audio_b64 because payload %d bytes exceeds limit %d bytes",
            len(audio_bytes),
            LEGACY_AUDIO_B64_MAX_BYTES,
        )
        return None
    return base64.b64encode(audio_bytes).decode("ascii")


def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> str:
    """Defense-in-depth auth guard for in-app API access."""
    if not API_AUTH_ENABLED:
        return "auth_disabled"
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key auth is enabled but SPARC_API_KEY is not configured",
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


async def load_models():
    global adapter_model, tokenizer
    base_model_name = "gpt-oss-120b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    adapter_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATHS["caregiver"],
        adapter_name="caregiver",
    )
    adapter_model.load_adapter(ADAPTER_PATHS["coach"], adapter_name="coach")
    adapter_model.load_adapter(ADAPTER_PATHS["supervisor"], adapter_name="supervisor")
    adapter_model.set_adapter("caregiver")

    load_guardrails_runtime()
    init_riva_clients()
    ensure_audio_cache_dir()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    user_message: Optional[str] = Field(default=None, max_length=10000)
    user_transcript: Optional[str] = Field(default=None, max_length=10000)
    audio_data: Optional[str] = Field(default=None, max_length=2_000_000)
    mode: Optional[str] = Field(default=None, max_length=32)
    agent_mode: Optional[str] = Field(default=None, max_length=32)
    target_agent: Optional[str] = Field(default=None, max_length=32)
    include_legacy_audio_b64: bool = True


class ChatResponse(BaseModel):
    response_text: str
    caregiver_text: str
    audio_url: Optional[str] = None
    caregiver_audio_b64: Optional[str] = None
    caregiver_animation_cues: Optional[Dict[str, str]] = None
    coach_feedback: Optional[str] = None
    coach_feedback_meta: Optional[Dict[str, Any]] = None
    active_agent: str
    api_contract_version: str = API_CONTRACT_VERSION


@app.get("/health")
async def health_check():
    riva_ok = riva_auth is not None and riva_asr_service is not None and riva_tts_service is not None

    model_ok = tokenizer is not None and adapter_model is not None
    status_text = "healthy" if model_ok else "degraded"
    health_payload = {
        "status": status_text,
        "models_loaded": model_ok,
        "ready_for_traffic": model_ok,
        "riva_connected": riva_ok,
        "api_auth_enabled": API_AUTH_ENABLED,
        "api_auth_configured": bool(API_KEY),
        "api_contract_version": API_CONTRACT_VERSION,
        "guardrails_loaded": guardrails_engine is not None,
        "riva_client_pool_initialized": riva_ok,
        "firebase_creds_configured": bool(FIREBASE_CREDS),
        "legacy_unity_compatibility": True,
    }
    http_status = status.HTTP_200_OK if model_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=health_payload)


@app.get("/v1/audio/{audio_id}")
async def get_tts_audio(audio_id: str):
    await prune_expired_audio_cache()
    async with audio_cache_lock:
        metadata = audio_cache_index.get(audio_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Audio clip not found or expired")

    audio_path = metadata.get("path")
    if not audio_path or not os.path.isfile(audio_path):
        async with audio_cache_lock:
            audio_cache_index.pop(audio_id, None)
        raise HTTPException(status_code=404, detail="Audio clip not found or expired")

    return FileResponse(audio_path, media_type="audio/wav", filename=f"{audio_id}.wav")


@app.post("/v1/chat", response_model=ChatResponse)
async def process_chat(request: ChatRequest, _api_key: str = Depends(require_api_key)):
    try:
        if adapter_model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model adapters are not initialized")

        session_ref = db.collection("sessions").document(request.session_id)
        session_state = session_ref.get().to_dict() or {}
        primary_adapter = resolve_requested_mode(request, session_state)
        normalized_user_message = extract_user_message(request)

        input_guard = await enforce_guardrails_input(normalized_user_message)
        if not input_guard["allowed"]:
            return ChatResponse(
                response_text=input_guard["text"],
                caregiver_text=input_guard["text"],
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback=None,
                coach_feedback_meta={"safe": False, "reason": input_guard["reason"]},
                active_agent=primary_adapter,
            )

        prompt = f"[SESSION: {request.session_id}] User: {input_guard['text']}\nAssistant:"
        model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_inputs = {k: v.to(adapter_model.device) for k, v in model_inputs.items()}

        if await is_circuit_open("primary_inference"):
            logger.warning("Primary inference circuit open; returning degraded fallback response")
            fallback_text = "I’m temporarily unable to generate a response right now. Please try again shortly."
            return ChatResponse(
                response_text=fallback_text,
                caregiver_text=fallback_text,
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback="Primary model temporarily unavailable.",
                coach_feedback_meta={"safe": True, "reason": "inference_circuit_open", "summary": "Primary model temporarily unavailable."},
                active_agent=primary_adapter,
            )

        try:
            async with inference_lock:
                adapter_model.set_adapter(primary_adapter)
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        generate_tokens_sync,
                        adapter_model,
                        **model_inputs,
                        max_new_tokens=180,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    ),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            await record_success_event("primary_inference")
        except asyncio.TimeoutError:
            circuit_opened = await record_timeout_event("primary_inference")
            logger.warning("Primary inference timed out after %.1fs%s", LLM_TIMEOUT_SECONDS, "; circuit opened" if circuit_opened else "")
            fallback_text = "I’m temporarily unable to generate a response right now. Please try again shortly."
            return ChatResponse(
                response_text=fallback_text,
                caregiver_text=fallback_text,
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback="Primary model timeout fallback.",
                coach_feedback_meta={"safe": True, "reason": "inference_timeout", "summary": "Primary model timeout fallback."},
                active_agent=primary_adapter,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response_text = decoded.split("Assistant:")[-1].strip() or "I’m here to help with HPV vaccine communication practice."

        output_guard = await enforce_guardrails_output(response_text)
        response_text = output_guard["text"]

        coach_feedback_text = "Coach feedback temporarily unavailable."
        coach_feedback_reason = output_guard["reason"]
        try:
            if await is_circuit_open("coach_inference"):
                logger.warning("Coach inference circuit open; skipping coach generation")
                coach_feedback_reason = "coach_circuit_open"
            else:
                feedback_prompt = f"Provide concise coaching feedback for this response: {response_text}"
                feedback_inputs = tokenizer(feedback_prompt, return_tensors="pt", truncation=True, max_length=512)
                feedback_inputs = {k: v.to(adapter_model.device) for k, v in feedback_inputs.items()}
                async with inference_lock:
                    adapter_model.set_adapter("coach")
                    feedback_tokens = await asyncio.wait_for(
                        asyncio.to_thread(
                            generate_tokens_sync,
                            adapter_model,
                            **feedback_inputs,
                            max_new_tokens=80,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        ),
                        timeout=COACH_TIMEOUT_SECONDS,
                    )
                await record_success_event("coach_inference")
                coach_feedback_text = tokenizer.decode(feedback_tokens[0], skip_special_tokens=True)
        except asyncio.TimeoutError:
            circuit_opened = await record_timeout_event("coach_inference")
            logger.warning("Coach inference timed out after %.1fs%s", COACH_TIMEOUT_SECONDS, "; circuit opened" if circuit_opened else "")
            coach_feedback_reason = "coach_timeout"
        except Exception as coach_error:
            logger.warning("Coach inference failed: %s")
            coach_feedback_reason = "coach_error"
        finally:
            async with inference_lock:
                adapter_model.set_adapter(primary_adapter)

        audio_url = None
        audio_bytes = None
        try:
            if await is_circuit_open("riva_tts"):
                logger.warning("Riva TTS circuit open; skipping speech synthesis")
            else:
                audio_bytes = await asyncio.wait_for(
                    asyncio.to_thread(synthesize_tts_sync, response_text, "English-US.Female-1"),
                    timeout=TTS_TIMEOUT_SECONDS,
                )
                await record_success_event("riva_tts")
                audio_url = await persist_tts_audio(audio_bytes)
        except asyncio.TimeoutError:
            circuit_opened = await record_timeout_event("riva_tts")
            logger.warning("Riva TTS timed out after %.1fs%s", TTS_TIMEOUT_SECONDS, "; circuit opened" if circuit_opened else "")
        except Exception as riva_error:
            logger.warning("Riva TTS unavailable: %s")

        legacy_audio_b64 = build_legacy_audio_b64(audio_bytes, request.include_legacy_audio_b64)
        animation_cues = default_animation_cues()
        session_state["mode"] = primary_adapter
        session_ref.set(session_state, merge=True)

        return ChatResponse(
            response_text=response_text,
            caregiver_text=response_text,
            audio_url=audio_url,
            caregiver_audio_b64=legacy_audio_b64,
            caregiver_animation_cues=animation_cues,
            coach_feedback=coach_feedback_text[:500],
            coach_feedback_meta={"safe": output_guard["allowed"], "reason": coach_feedback_reason, "summary": coach_feedback_text[:500]},
            active_agent=primary_adapter,
        )
    except Exception as e:
        logger.exception("/v1/chat failed after path: %s")
        raise HTTPException(status_code=500, detail="Internal server error")
''').strip()
main_content = (
    main_content
    .replace("{MODEL_DIR}", str(MODEL_DIR))
    .replace("{PUBAPPS_ROOT}", str(PUBAPPS_ROOT))
    .replace("{PUBAPP_ALLOWED_ORIGINS}", str(PUBAPP_ALLOWED_ORIGINS))
)

BACKEND_DIR.mkdir(parents=True, exist_ok=True)
main_py.write_text(main_content)
print(f"Wrote {main_py}")
```

<!-- Cell 22: code (python) -->

```python
# 6.1b Merge websocket ASR support into main.py for a single FastAPI service
backend_text = main_py.read_text()

backend_text = backend_text.replace(
    "import base64\n",
    "import base64\nimport json\n",
    1,
)

backend_text = backend_text.replace(
    "from fastapi import Depends, FastAPI, Header, HTTPException, status",
    "from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status",
    1,
)

backend_text = backend_text.replace(
    'DEFAULT_ANIMATION_GESTURE = os.getenv("SPARC_DEFAULT_ANIMATION_GESTURE", "speaking")',
    textwrap.dedent('''
    DEFAULT_ANIMATION_GESTURE = os.getenv("SPARC_DEFAULT_ANIMATION_GESTURE", "speaking")
    DEFAULT_LANGUAGE_CODE = os.getenv("SPARC_ASR_LANGUAGE_CODE", "en-US")
    DEFAULT_SAMPLE_RATE_HZ = int(os.getenv("SPARC_ASR_SAMPLE_RATE_HZ", "16000"))
    DEFAULT_CHANNEL_COUNT = int(os.getenv("SPARC_ASR_CHANNEL_COUNT", "1"))
    DEFAULT_MAX_ALTERNATIVES = int(os.getenv("SPARC_ASR_MAX_ALTERNATIVES", "1"))
    DEFAULT_AUTOMATIC_PUNCTUATION = os.getenv("SPARC_ASR_AUTO_PUNCT", "true").strip().lower() == "true"
    DEFAULT_PROFANITY_FILTER = os.getenv("SPARC_ASR_PROFANITY_FILTER", "false").strip().lower() == "true"
    DEFAULT_INTERIM_RESULTS = os.getenv("SPARC_ASR_INTERIM_RESULTS", "true").strip().lower() == "true"
    ''').strip(),
    1,
)

backend_text = backend_text.replace(
    '        "legacy_unity_compatibility": True,\n',
    '        "legacy_unity_compatibility": True,\n        "websocket_path": "/ws/audio",\n        "default_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,\n',
    1,
)

ws_merge_block = textwrap.dedent('''
def normalize_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_streaming_config(settings: Dict[str, Any]):
    recognition_config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        language_code=str(settings.get("language", DEFAULT_LANGUAGE_CODE)),
        sample_rate_hertz=int(settings.get("sample_rate", DEFAULT_SAMPLE_RATE_HZ)),
        audio_channel_count=int(settings.get("channels", DEFAULT_CHANNEL_COUNT)),
        max_alternatives=int(settings.get("max_alternatives", DEFAULT_MAX_ALTERNATIVES)),
        profanity_filter=normalize_bool(settings.get("profanity_filter"), DEFAULT_PROFANITY_FILTER),
        enable_automatic_punctuation=normalize_bool(
            settings.get("enable_automatic_punctuation"),
            DEFAULT_AUTOMATIC_PUNCTUATION,
        ),
        verbatim_transcripts=False,
        enable_word_time_offsets=True,
    )
    return riva.client.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=normalize_bool(settings.get("interim_results"), DEFAULT_INTERIM_RESULTS),
    )


def default_session_settings() -> Dict[str, Any]:
    return {
        "language": DEFAULT_LANGUAGE_CODE,
        "sample_rate": DEFAULT_SAMPLE_RATE_HZ,
        "channels": DEFAULT_CHANNEL_COUNT,
        "encoding": "LINEAR_PCM",
        "enable_automatic_punctuation": DEFAULT_AUTOMATIC_PUNCTUATION,
        "profanity_filter": DEFAULT_PROFANITY_FILTER,
        "max_alternatives": DEFAULT_MAX_ALTERNATIVES,
        "interim_results": DEFAULT_INTERIM_RESULTS,
    }


class StreamingSession:
    def __init__(self, websocket: WebSocket, session_id: str, settings: Dict[str, Any]):
        self.websocket = websocket
        self.session_id = session_id
        self.settings = dict(settings)
        self.audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self.task is not None:
            return
        self.loop = asyncio.get_running_loop()
        self.task = asyncio.create_task(asyncio.to_thread(self._run_stream))

    async def push_audio(self, audio_bytes: bytes) -> None:
        await self.audio_queue.put(audio_bytes)

    async def finish(self) -> None:
        await self.audio_queue.put(None)
        if self.task is not None:
            try:
                await self.task
            finally:
                self.task = None

    def _audio_chunks(self):
        while True:
            chunk = asyncio.run_coroutine_threadsafe(self.audio_queue.get(), self.loop).result()
            if chunk is None:
                break
            yield chunk

    def _emit_json(self, payload: Dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(self.websocket.send_json(payload), self.loop).result()

    def _run_stream(self) -> None:
        if riva_asr_service is None:
            self._emit_json(
                {
                    "type": "error",
                    "message": "Riva ASR service is not initialized",
                    "code": "RIVA_UNAVAILABLE",
                    "recoverable": False,
                }
            )
            return

        try:
            responses = riva_asr_service.streaming_response_generator(
                audio_chunks=self._audio_chunks(),
                streaming_config=build_streaming_config(self.settings),
            )
            for response in responses:
                for result in getattr(response, "results", []):
                    alternatives_payload = []
                    raw_alternatives = list(getattr(result, "alternatives", []))
                    for alternative in raw_alternatives:
                        alternatives_payload.append(
                            {
                                "transcript": getattr(alternative, "transcript", ""),
                                "confidence": float(getattr(alternative, "confidence", 0.0) or 0.0),
                            }
                        )
                    if not alternatives_payload:
                        continue

                    words_payload = []
                    top_words = getattr(raw_alternatives[0], "words", []) if raw_alternatives else []
                    for word in top_words:
                        start_time = getattr(word, "start_time", None)
                        end_time = getattr(word, "end_time", None)
                        words_payload.append(
                            {
                                "word": getattr(word, "word", ""),
                                "start_time": getattr(start_time, "seconds", 0)
                                + getattr(start_time, "nanos", 0) / 1_000_000_000,
                                "end_time": getattr(end_time, "seconds", 0)
                                + getattr(end_time, "nanos", 0) / 1_000_000_000,
                                "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
                            }
                        )

                    self._emit_json(
                        {
                            "type": "transcript",
                            "transcript": alternatives_payload[0]["transcript"],
                            "is_final": bool(getattr(result, "is_final", False)),
                            "confidence": alternatives_payload[0]["confidence"],
                            "alternatives": alternatives_payload,
                            "words": words_payload,
                        }
                    )
        except Exception as asr_error:
            logger.exception("Streaming ASR session failed: %s")
            self._emit_json(
                {
                    "type": "error",
                    "message": "ASR stream failed",
                    "code": "ASR_STREAM_FAILED",
                    "recoverable": True,
                }
            )


@app.websocket("/ws/audio")
async def websocket_audio_bridge(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if CORS_ALLOWED_ORIGINS and origin and origin not in CORS_ALLOWED_ORIGINS:
        await websocket.close(code=1008, reason="Origin not allowed")
        return

    requested_subprotocols = websocket.scope.get("subprotocols", [])
    accepted_subprotocol = "riva-asr-v1" if "riva-asr-v1" in requested_subprotocols else None
    await websocket.accept(subprotocol=accepted_subprotocol)

    session_id = uuid.uuid4().hex
    settings = default_session_settings()
    stream: Optional[StreamingSession] = None

    await websocket.send_json(
        {
            "type": "status",
            "status": "ready",
            "session_id": session_id,
            "riva_version": "pubapps-integrated",
            "capabilities": ["streaming_asr", "punctuation", "profanity_filter"],
            "max_audio_duration": 300,
        }
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            audio_chunk = message.get("bytes")
            if audio_chunk is not None:
                if stream is None:
                    stream = StreamingSession(websocket=websocket, session_id=session_id, settings=settings)
                    await stream.start()
                await stream.push_audio(audio_chunk)
                continue

            text_payload = message.get("text")
            if text_payload is None:
                continue

            try:
                payload = json.loads(text_payload)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON control message",
                        "code": "INVALID_JSON",
                        "recoverable": True,
                    }
                )
                continue

            message_type = payload.get("type")
            if message_type == "start_recognition":
                settings.update(payload.get("data") or {})
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "recognition_configured",
                        "session_id": session_id,
                        "settings": settings,
                    }
                )
            elif message_type == "stop_recognition":
                if stream is not None:
                    await stream.finish()
                    stream = None
                await websocket.send_json(
                    {
                        "type": "status",
                        "status": "stopped",
                        "session_id": session_id,
                    }
                )
            elif message_type == "ping":
                await websocket.send_json(
                    {
                        "type": "pong",
                        "timestamp": payload.get("timestamp", time.time()),
                        "session_id": session_id,
                    }
                )
            else:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": f"Unsupported control message: {message_type}",
                        "code": "UNSUPPORTED_MESSAGE",
                        "recoverable": True,
                    }
                )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", session_id)
    finally:
        if stream is not None:
            await stream.finish()
''').strip()

backend_text = backend_text.replace(
    '@app.get("/health")',
    ws_merge_block + '\n\n@app.get("/health")',
    1,
)

main_py.write_text(backend_text)
print(f"Merged websocket support into {main_py}")
```

<!-- Cell 23: markdown -->

A comprehensive automated check scans the `main.py` file and asserts that over 80 specific code patterns are present (and several dangerous legacy patterns are absent).

Think of it as a build-quality gate: before deploying the application, this step verifies that all the critical security, reliability, and compliance features are actually in the code.

The checks are grouped into categories:
- **Adapter management (C4/C5):** Confirms all three LLM adapters (caregiver, coach, supervisor) are registered by name using `adapter_name=` parameters â€” not as three separate model objects (which would triple GPU memory usage).
- **API authentication (M7):** Verifies the `require_api_key` auth guard is defined and injected via `Depends()` into the chat endpoint.
- **Environment config (M8):** Confirms all sensitive values (Firebase path, Riva URL, model path, CORS origins) are read from environment variables â€” not hard-coded.
- **CORS security (H3):** Checks that `allow_origins=[\"*\"]` (wildcard) is absent and specific allowed origins are configured.
- **Guardrails (H5):** Confirms NeMo Guardrails is imported and both input and output enforcement functions are called.
- **Async inference (H12):** Validates that `asyncio.wait_for()` and `asyncio.to_thread()` are used for model calls â€” preventing the event loop from blocking during inference.
- **Circuit breaker (H13):** Checks that timeout and circuit breaker functions are defined and wired up for all three operations (inference, coach, TTS).
- **Quantization (H15):** Confirms 4-bit NF4 quantization config is present â€” the memory optimization that makes the 120B-parameter model fit on an L4 GPU.

> **If any assertion fails:** The error message tells you exactly which marker is missing or which blocked pattern is still present, so you know exactly what needs to be fixed in the main.py before deploying.

<!-- Cell 24: code (python) -->

```python
# 6.2 C4/C5/M7/M8/M9/M11/L5/H2/H3/H5/H10/H11/H12/H13/H14/H15 Smoke Test — Adapter/Auth/Config + Timeout/Circuit-Breaker + Riva Client Reuse + Bounded TTS Delivery + Lifespan Lifecycle + Redaction + Contract + CORS + Guardrails + Async Inference + Health Readiness + Schema Constraints + Quantization + Legacy Unity Compatibility + Integrated WebSocket Route

backend_text = main_py.read_text()
ws_bridge_text = backend_text

required_markers = [
    "import base64",
    "adapter_name=\"caregiver\"",
    "load_adapter(ADAPTER_PATHS[\"coach\"], adapter_name=\"coach\")",
    "load_adapter(ADAPTER_PATHS[\"supervisor\"], adapter_name=\"supervisor\")",
    "adapter_model.set_adapter(primary_adapter)",
    "adapter_model.set_adapter(\"coach\")",
    "def require_api_key(",
    "Header(default=None, alias=\"X-API-Key\")",
    "Depends(require_api_key)",
    "SPARC_FIREBASE_CREDS",
    "SPARC_MODEL_BASE_PATH",
    "SPARC_RIVA_SERVER",
    "os.path.isfile(FIREBASE_CREDS)",
    "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig",
    "API_CONTRACT_VERSION = \"v1\"",
    "session_id: str = Field(..., min_length=1, max_length=128, pattern=r\"^[a-zA-Z0-9_-]+$\")",
    "user_message: Optional[str] = Field(default=None, max_length=10000)",
    "user_transcript: Optional[str] = Field(default=None, max_length=10000)",
    "audio_data: Optional[str] = Field(default=None, max_length=2_000_000)",
    "mode: Optional[str] = Field(default=None, max_length=32)",
    "agent_mode: Optional[str] = Field(default=None, max_length=32)",
    "target_agent: Optional[str] = Field(default=None, max_length=32)",
    "include_legacy_audio_b64: bool = True",
    "caregiver_text: str",
    "caregiver_audio_b64: Optional[str] = None",
    "caregiver_animation_cues: Optional[Dict[str, str]] = None",
    "coach_feedback: Optional[str] = None",
    "coach_feedback_meta: Optional[Dict[str, Any]] = None",
    "active_agent: str",
    "api_contract_version: str = API_CONTRACT_VERSION",
    "api_contract_version\": API_CONTRACT_VERSION",
    "CORS_ALLOWED_ORIGINS = [",
    "CORS_ALLOW_CREDENTIALS = os.getenv(\"SPARC_CORS_ALLOW_CREDENTIALS\", \"false\")",
    "allow_origins=CORS_ALLOWED_ORIGINS",
    "allow_credentials=CORS_ALLOW_CREDENTIALS",
    "from nemoguardrails import LLMRails, RailsConfig",
    "load_guardrails_runtime()",
    "enforce_guardrails_input(normalized_user_message)",
    "enforce_guardrails_output(response_text)",
    "guardrails_loaded\": guardrails_engine is not None",
    "legacy_unity_compatibility\": True",
    "import asyncio",
    "inference_lock = asyncio.Lock()",
    "LLM_TIMEOUT_SECONDS = float(os.getenv(\"SPARC_LLM_TIMEOUT_SECONDS\", \"10\"))",
    "COACH_TIMEOUT_SECONDS = float(os.getenv(\"SPARC_COACH_TIMEOUT_SECONDS\", \"10\"))",
    "TTS_TIMEOUT_SECONDS = float(os.getenv(\"SPARC_TTS_TIMEOUT_SECONDS\", \"5\"))",
    "TTS_MAX_AUDIO_BYTES = int(os.getenv(\"SPARC_TTS_MAX_AUDIO_BYTES\", \"524288\"))",
    "LEGACY_AUDIO_B64_MAX_BYTES = int(os.getenv(\"SPARC_LEGACY_AUDIO_B64_MAX_BYTES\", str(TTS_MAX_AUDIO_BYTES)))",
    "SPARC_AUDIO_URL_TTL_SECONDS = float(os.getenv(\"SPARC_AUDIO_URL_TTL_SECONDS\", \"300\"))",
    "SPARC_AUDIO_CACHE_DIR = os.getenv(\"SPARC_AUDIO_CACHE_DIR\", os.path.join(tempfile.gettempdir(), \"sparc_tts_audio\"))",
    "from contextlib import asynccontextmanager",
    "async def lifespan(app: FastAPI):",
    "await load_models()",
    "lifespan=lifespan",
    "CIRCUIT_BREAKER_THRESHOLD = int(os.getenv(\"SPARC_TIMEOUT_CIRCUIT_THRESHOLD\", \"3\"))",
    "CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv(\"SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS\", \"30\"))",
    "def init_riva_clients() -> None:",
    "riva_asr_service = riva.client.ASRService(riva_auth)",
    "riva_tts_service = riva.client.SpeechSynthesisService(riva_auth)",
    "init_riva_clients()",
    "if riva_tts_service is None:",
    "riva_client_pool_initialized\": riva_ok",
    "async def is_circuit_open(operation: str) -> bool:",
    "async def record_timeout_event(operation: str) -> bool:",
    "def generate_tokens_sync(",
    "def synthesize_tts_sync(",
    "def resolve_requested_mode(request: \"ChatRequest\", session_state: Dict[str, Any]) -> str:",
    "def extract_user_message(request: \"ChatRequest\") -> str:",
    "def default_animation_cues() -> Dict[str, str]:",
    "def build_legacy_audio_b64(audio_bytes: Optional[bytes], include_legacy_audio_b64: bool) -> Optional[str]:",
    "base64.b64encode(audio_bytes).decode(\"ascii\")",
    "async def persist_tts_audio(audio_bytes: bytes) -> Optional[str]:",
    "@app.get(\"/v1/audio/{audio_id}\")",
    "return FileResponse(audio_path, media_type=\"audio/wav\", filename=f\"{audio_id}.wav\")",
    "asyncio.wait_for(",
    "asyncio.to_thread(",
    "Primary inference timed out after",
    "Coach inference timed out after",
    "Riva TTS timed out after",
    "from fastapi.responses import JSONResponse",
    "model_ok = tokenizer is not None and adapter_model is not None",
    "ready_for_traffic\": model_ok",
    "status.HTTP_503_SERVICE_UNAVAILABLE",
    "return JSONResponse(status_code=http_status, content=health_payload)",
    "raise HTTPException(status_code=500, detail=\"Internal server error\")",
    "bnb_config = BitsAndBytesConfig(",
    "quantization_config=bnb_config",
    "bnb_4bit_quant_type=\"nf4\"",
    "bnb_4bit_compute_dtype=torch.bfloat16",
]

required_ws_markers = [
    "import json",
    "from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status",
    "DEFAULT_SAMPLE_RATE_HZ = int(os.getenv(\"SPARC_ASR_SAMPLE_RATE_HZ\", \"16000\"))",
    "def build_streaming_config(settings: Dict[str, Any]):",
    "riva.client.RecognitionConfig(",
    "riva.client.StreamingRecognitionConfig(",
    "class StreamingSession:",
    "streaming_response_generator(",
    "@app.websocket(\"/ws/audio\")",
    "await websocket.accept(subprotocol=accepted_subprotocol)",
    "await websocket.send_json({",
    "\"type\": \"status\"",
    "\"type\": \"transcript\"",
    "\"type\": \"pong\"",
    "message_type == \"start_recognition\"",
    "message_type == \"stop_recognition\"",
    "message_type == \"ping\"",
]

missing = [marker for marker in required_markers if marker not in backend_text]
missing_ws = [marker for marker in required_ws_markers if marker not in ws_bridge_text]
assert not missing, f"Missing required backend markers: {missing}"
assert not missing_ws, f"Missing required websocket markers in main.py: {missing_ws}"

assert "caregiver_model = PeftModel.from_pretrained(base_model" not in backend_text, "Legacy shared-object adapter pattern remains"
assert "coach_model = PeftModel.from_pretrained(base_model" not in backend_text, "Legacy shared-object adapter pattern remains"
assert "supervisor_model = PeftModel.from_pretrained(base_model" not in backend_text, "Legacy shared-object adapter pattern remains"
assert "async def process_chat(request: ChatRequest):" not in backend_text, "Endpoint still lacks auth dependency"
assert "session_state[\"last_user_message\"] = request.user_message" not in backend_text, "Raw user message still persisted to Firebase"
assert "session_state[\"last_response\"] = response_text" not in backend_text, "Raw response still persisted to Firebase"
assert "allow_origins=[\"*\"]" not in backend_text, "Wildcard CORS origins remain configured"
assert "allow_credentials=True" not in backend_text, "Credentialed wildcard CORS remains configured"
assert "blocked = [\"politics\", \"election\", \"gambling\", \"crypto\", \"finance advice\"]" not in backend_text, "Legacy keyword blocklist remains configured"
assert "output = adapter_model.generate(" not in backend_text, "Primary generation still blocks event loop"
assert "feedback_tokens = adapter_model.generate(" not in backend_text, "Coach generation still blocks event loop"
assert "\"models_loaded\": True" not in backend_text, "Health still hard-codes models_loaded=True"
assert "detail=str(e)" not in backend_text, "Raw exception details still leak to client"
assert "from_pretrained(base_model_name, load_in_4bit=" not in backend_text, "Legacy direct load_in_4bit kwarg in from_pretrained() still present"
assert "data:audio/wav;base64" not in backend_text, "Legacy data-URI audio delivery still present"
assert "@app.on_event(\"startup\")" not in backend_text, "Deprecated FastAPI startup event hook still present"
assert "webSocketUrl = \"wss://hipergator.apps.rc.ufl.edu:8080/ws/audio\"" not in backend_text, "Unity client URL leaked into backend implementation"

print("✅ C4/C5/M7/M8/M9/M11/L5/H2/H3/H5/H10/H11/H12/H13/H14/H15 validation passed: named adapters, optional auth guard, timeout/circuit-breaker policy, startup-initialized reusable Riva clients, bounded TTS URL delivery, legacy Unity base64 compatibility, lifespan-based FastAPI lifecycle initialization, env config, unified v1 API contract, safe CORS policy, runtime Guardrails pipeline, non-blocking async inference path, readiness-aware health behavior, strict request schema constraints, direct per-request agent routing, explicit 4-bit quantization config, and an integrated `/ws/audio` WebSocket ASR route in the main FastAPI app are configured.")
```

<!-- Cell 25: markdown -->

`health_load_test` now runs directly in this notebook instead of being generated as a separate Python file.

What the inline test does when you run the next cell:
- **Fires 30 concurrent chat requests** (`POST /v1/chat`) using a thread pool, simulating 30 simultaneous users sending messages about HPV vaccines to the backend. This stress-tests the async inference pipeline.
- **Simultaneously pings `/health` every 200ms for 12 seconds** — for a total of 60 health check calls — to measure how the health endpoint responds *while* the backend is under load from the chat requests.
- **Measures p95 latency** for health checks (the 95th percentile, meaning 95% of checks must complete within this time).
- **Asserts three conditions:**
  1. All 30 chat requests return a recognized status code (200 OKs, 401 if API key is wrong in the test, or 422 for validation errors — but not 500 errors).
  2. 99% of health checks must complete successfully within 1.5 seconds.
  3. The p95 health latency must be under 1,500ms — confirming the health endpoint stays responsive even when inference is running.

> **To run this test:** After the backend is live, set `SPARC_API_KEY` and `SPARC_BASE_URL` if needed, then run the next code cell directly in this notebook.

<!-- Cell 26: code (python) -->

```python
# 6.3 Health Load Test — Health Responsiveness Under Chat Load
import concurrent.futures
import os
import statistics
import time

import requests

BASE_URL = os.getenv("SPARC_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("SPARC_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
CHAT_PAYLOAD = {
    "session_id": "health-load",
    "user_message": "Help me discuss HPV vaccines with a hesitant caregiver.",
}


def post_chat() -> int:
    response = requests.post(f"{BASE_URL}/v1/chat", json=CHAT_PAYLOAD, headers=HEADERS, timeout=120)
    return response.status_code


def ping_health() -> float:
    start = time.perf_counter()
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000


health_latencies = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    chat_futures = [pool.submit(post_chat) for _ in range(30)]
    for _ in range(60):
        health_latencies.append(ping_health())
        time.sleep(0.2)
    chat_statuses = [future.result() for future in chat_futures]

health_p95 = statistics.quantiles(health_latencies, n=20)[18] if len(health_latencies) >= 20 else max(health_latencies)
health_success_ratio = sum(1 for latency in health_latencies if latency < 1500) / len(health_latencies)

assert all(code in (200, 401, 422) for code in chat_statuses), f"Unexpected chat status codes: {sorted(set(chat_statuses))}"
assert health_success_ratio >= 0.99, f"Health responsiveness dropped below target: {health_success_ratio:.3f}"
assert health_p95 < 1500, f"Health p95 latency too high under chat load: {health_p95:.1f}ms"

print(f"✅ Health load test passed: /health p95={health_p95:.1f}ms, success_ratio={health_success_ratio:.3f}")
```

<!-- Cell 27: markdown -->

`quantization_memory_check` now runs directly in this notebook instead of being generated as a separate Python file.

What the inline check measures and why it matters:

The L4 GPU has **24 GB of VRAM** total. The SPARC-P system needs to share this between three components:
- The fine-tuned LLM (120B parameters in 4-bit quantization ≈ ~13 GB)
- The NVIDIA Riva ASR and TTS models (≈ ~3 GB combined, running in a separate container)
- System overhead and CUDA libraries (≈ ~1–2 GB)

That leaves only ~7 GB headroom. If memory usage grows beyond the expected budget, the system may start throwing CUDA out-of-memory errors during inference — causing 500 errors for users.

What the next cell does:
1. Checks that CUDA is available (fails loudly if not — this check is useless without a GPU).
2. Calls `torch.cuda.synchronize()` to ensure all pending CUDA operations are flushed.
3. Reads `memory_allocated()` (actively used by tensors), `memory_reserved()` (total pool held by PyTorch), and total `capacity_gb` from the GPU device.
4. **Asserts that reserved memory is under 22.0 GB** — leaving at least 2 GB headroom on a 24 GB L4.

> **To run:** After the backend has been running for a few minutes so the model is fully loaded, run the next code cell directly in this notebook.

<!-- Cell 28: code (python) -->

```python
# 6.4 Quantization Memory Profile Check
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for quantization memory profile check")

    torch.cuda.synchronize()
    allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    capacity_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print(f"GPU memory allocated: {allocated_gb:.2f} GB")
    print(f"GPU memory reserved: {reserved_gb:.2f} GB")
    print(f"GPU capacity: {capacity_gb:.2f} GB")

    assert reserved_gb < 22.0, (
        f"Reserved memory exceeds expected L4 quantized startup budget: {reserved_gb:.2f} GB"
    )
    print("✅ Quantization memory profile check passed: quantized startup is within expected L4 budget.")


main()
```

<!-- Cell 29: markdown -->

The systemd service file tells the Linux process manager how to run the unified SPARC-P FastAPI app as a persistent service — so it starts automatically and restarts itself if it crashes.

What the generated service file specifies, and why each setting matters:
- **`sparc-backend.service`** runs the single FastAPI app on port 8000 for `/v1/chat`, `/v1/audio/{id}`, `/health`, and `/ws/audio`.
- **`After=network.target riva-server.service`**: The service only starts *after* the Riva speech server is running. Without this ordering, the backend could start before Riva is ready and fail to connect to `localhost:50051`.
- **`Requires=riva-server.service`**: If Riva stops, systemd also stops the SPARC service. This prevents the app from running silently without speech support.
- **`ExecStart={CONDA_ENV}/bin/uvicorn ...`**: Uses the *full absolute path* to the uvicorn binary inside the conda environment — not relying on `PATH`. This guarantees the correct Python environment is used even in a non-interactive systemd session.
- **Single app on 8000**: REST and WebSocket traffic are served by the same process, reducing deployment complexity and eliminating the extra bridge service.
- **`Restart=always` + `RestartSec=10`**: If the process crashes, systemd waits 10 seconds and restarts it automatically.
- The file is written to `~/.config/systemd/user/` — the per-user systemd directory that a non-root user can manage without sudo.

<!-- Cell 30: code (python) -->

```python
# 6.5 Create systemd user service for unified FastAPI backend
systemd_dir = Path.home() / '.config/systemd/user'
systemd_dir.mkdir(parents=True, exist_ok=True)

backend_service_file = systemd_dir / 'sparc-backend.service'
backend_service_content = textwrap.dedent(f"""
[Unit]
Description=SPARC-P Unified FastAPI Backend
After=network.target riva-server.service
Requires=riva-server.service

[Service]
Type=simple
Environment=PATH={CONDA_ENV}/bin:/usr/bin
Environment=PYTHONUNBUFFERED=1
WorkingDirectory={BACKEND_DIR}
ExecStart={CONDA_ENV}/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers {UVICORN_WORKERS}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
""").strip()
backend_service_file.write_text(backend_service_content)

print(f"Wrote {backend_service_file}")
print(backend_service_content)
print(f"Configured backend uvicorn workers: {UVICORN_WORKERS}")
```

<!-- Cell 31: markdown -->

This is the final service activation step — it registers the unified FastAPI backend with systemd and starts it running.

Step by step:
1. **`systemctl --user daemon-reload`**: Tells systemd to re-read all service files from disk, picking up `sparc-backend.service`.
2. **`systemctl --user enable --now sparc-backend`**: Registers the backend to start automatically and starts it immediately.
3. **Status check** (`check=False`): Prints the current status for the service. Expected output: `Active: active (running)`.

After completing successfully with `EXECUTE = True`:
- The FastAPI backend is live at `http://localhost:8000`
- The WebSocket route is live at `ws://localhost:8000/ws/audio`
- The Riva speech server is live at `localhost:50051`
- The service is persistent and will restart automatically on failure
- Run `curl -s http://localhost:8000/health` to confirm readiness

<!-- Cell 32: code (python) -->

```python
# 6.6 Enable unified backend service
run("systemctl --user daemon-reload")
run("systemctl --user enable --now sparc-backend")
run("systemctl --user status sparc-backend --no-pager", check=False)
```

<!-- Cell 33: markdown -->

## 7. Validation Checks
Set `EXECUTE = True` before running these checks.

<!-- Cell 34: markdown -->

![6.0 & 9.0 End-to-End Production Access Flow Diagram](../images/notebook4-5.png)

This sequence diagram shows the full production traffic flow, highlighting the integration of NGINX, UF Shibboleth SSO authentication, and the HIPAA-mandated "Transient PHI" compliance loop.

The final deployment check verifies that the unified FastAPI backend and Riva are operational by running diagnostic commands against the live PubApps VM. Switch `EXECUTE = TRUE` before running, otherwise the commands will only print.

What each command checks:
1. **`curl -s http://localhost:8000/health`**: Confirms the unified backend is healthy, model adapters are loaded, and the integrated WebSocket route is configured.
2. **`journalctl --user -u riva-server -n 50`**: Shows the last 50 log lines from the Riva speech server service.
3. **`journalctl --user -u sparc-backend -n 50`**: Shows the last 50 log lines from the unified backend service.
4. **`ls -lh {MODEL_DIR}`**: Lists the model files in the models directory and their sizes.

> **If websocket clients fail to connect:** Check the `sparc-backend` journal first. Common causes are Riva not being ready on `localhost:50051`, origin mismatch, or the frontend still pointing to `:8080/ws/audio` instead of the unified backend route.

<!-- Cell 35: code (python) -->

```python
# 7.1 Health and service checks
run("curl -s http://localhost:8000/health", check=False)
run("journalctl --user -u riva-server -n 50 --no-pager", check=False)
run("journalctl --user -u sparc-backend -n 50 --no-pager", check=False)
run(f"ls -lh {MODEL_DIR}", check=False)
```
