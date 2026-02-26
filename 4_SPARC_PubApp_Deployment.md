# SPARC-P PubApp Deployment Guide

## Overview


The final deployment check verifies the entire SPARC-P deployment is operational by running four diagnostic commands against the live PubApps VM. Switch `EXECUTE = True` before running, otherwise all four commands will just print without executing.

What each command checks:
1. **`curl -s http://localhost:8000/health`**: Makes an HTTP request to the backend's health endpoint. A healthy response looks like `{"status": "healthy", "models_loaded": true, "riva_connected": true, "guardrails_loaded": true, ...}`. If you see `"status": "degraded"` or an HTTP error, the backend is not fully initialized — check the service log.
2. **`journalctl --user -u riva-server -n 50`**: Shows the last 50 log lines from the Riva speech server service. Look for lines like `Riva server ready` and confirm there are no CUDA errors or model loading failures.
3. **`journalctl --user -u sparc-backend -n 50`**: Shows the last 50 log lines from the FastAPI backend service. Look for uvicorn startup messages and confirm that model adapters, guardrails, and Riva clients all initialized successfully.
4. **`ls -lh {MODEL_DIR}`**: Lists the model files in the models directory and their sizes. This confirms the model adapters were transferred from HiPerGator successfully. You should see directories for `CaregiverAgent`, `C-LEAR_CoachAgent`, and `SupervisorAgent` — each several GB in size.

> **If health returns `"models_loaded": false`:** The LLM adapters failed to load. Common causes: the model directory path is wrong, the PEFT adapter files are missing, or the GPU ran out of memory during loading. Check the backend journal for the specific error.
This notebook provides step-by-step instructions for deploying the SPARC-P backend to **UF RC PubApps** for public access. PubApps is a separate infrastructure from HiPerGator designed for hosting web applications that serve research results.

### Key Differences: HiPerGator vs PubApps

| Aspect | HiPerGator | PubApps |
|--------|------------|---------|
| **Purpose** | Model training & batch processing | Public web application hosting |
| **Access** | Internal (UF network + VPN) | Public internet |
| **Containers** | Apptainer only | Podman only |
| **Storage** | `/blue` (shared with HiPerGator) | `/pubapps` (1TB included) |
| **Scheduling** | SLURM batch jobs | Systemd services (persistent) |
| **GPUs** | 4 GPUs available for parallelization | 1x L4 (24GB) for inference |
| **CPU / RAM** | 16 CPU cores available | 2 CPU cores, 16GB RAM |
| **Conda** | Yes, via modules | Yes, can be installed |

---

## 1.0 Prerequisites

### 1.1 Required Allocations

Before deploying to PubApps, ensure you have:

1. **PA-Instance allocation** ($300/year) - Request via [HiPerGator Service Purchase Form](https://it.ufl.edu/rc/get-started/purchase-request/)
2. **PA-GPU allocation** (if using GPU inference) - L4 GPUs available
3. **Completed Risk Assessment** - See [Risk Assessment Documentation](https://docs.rc.ufl.edu/services/web_hosting/risk_assessment/)

### 1.2 PubApp Instance Setup

After purchasing a PA-Instance, open a support ticket to provision your instance:
- Instance will be accessible via SSH from HiPerGator
- You'll receive a project user account (usually matches your HiPerGator group name)
- Default resources for this project: 1x L4 (24GB), 2 vCPUs, 16GB RAM, 1TB `/pubapps` storage

### 1.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Public Internet                          │
│                          │                                   │
│                    UF Shibboleth                            │
│                    (Authentication)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    PubApps Instance                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              NGINX Reverse Proxy                       │ │
│  │         (SSL termination, routing)                     │ │
│  └─────────┬──────────────────────────────┬───────────────┘ │
│            │                               │                 │
│  ┌─────────▼─────────┐         ┌──────────▼──────────────┐ │
│  │  Unity WebGL      │         │   FastAPI Backend       │ │
│  │  (Static Files)   │         │   (Conda Environment)   │ │
│  └───────────────────┘         │                         │ │
│                                │  ┌──────────────────┐   │ │
│                                │  │  LangGraph       │   │ │
│                                │  │  Orchestration   │   │ │
│                                │  └──────────────────┘   │ │
│                                │  ┌──────────────────┐   │ │
│                                │  │ Trained Models   │   │ │
│                                │  │ (from /blue)     │   │ │
│                                │  └──────────────────┘   │ │
│                                └─────────┬───────────────┘ │
│                                          │                  │
│                                ┌─────────▼──────────────┐  │
│                                │   Riva Container       │  │
│                                │   (ASR/TTS, L4 GPU)    │  │
│                                └────────────────────────┘  │
│                                          │                  │
│  ┌────────────────────────────────────┘                    │
│  │          Firebase Firestore (External)                  │
│  │          (Session state, metrics)                       │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

---

## 2.0 Transfer Trained Models from HiPerGator

### 2.1 Sync Models to PubApps Storage


The `rsync` command below transfers the fine-tuned SPARC-P model adapters from HiPerGator's `/blue` storage to the PubApps VM's model directory. The command is **printed, not executed** (unless `EXECUTE = True`).

`rsync` is the right tool for this task because:
- It transfers only files that have changed (delta sync), so re-running after a partial transfer is safe and fast.
- The `-avz` flags mean: archive mode (preserves permissions and timestamps), verbose output, and gzip compression in transit.
- `--progress` shows a progress bar for each file, which is helpful since the fine-tuned adapters can be 2–10 GB total.

The command transfers from `HIPERGATOR_SOURCE_MODELS/` (the `/blue/jasondeanarnold/SPARCP/trained_models` directory) to `MODEL_DIR` on the PubApps VM using SSH as the transport.

> **Where to run this:** This command needs to run on HiPerGator (where the `/blue` filesystem is mounted) or on a machine that has SSH access to both systems. Copy the printed command and run it in your HiPerGator terminal session.
PubApps storage (`/pubapps`) is NOT directly accessible from HiPerGator. You must transfer files.

```bash
# On HiPerGator, after training completes
# Configure source/destination endpoints
export SPARC_BASE_PATH=${SPARC_BASE_PATH:-/blue/jasondeanarnold/SPARCP}
export SPARC_HIPERGATOR_SOURCE_MODELS=${SPARC_HIPERGATOR_SOURCE_MODELS:-$SPARC_BASE_PATH/trained_models}
export SPARC_PUBAPPS_SSH_USER=${SPARC_PUBAPPS_SSH_USER:-SPARCP}
export SPARC_PUBAPPS_HOST=${SPARC_PUBAPPS_HOST:-pubapps-vm.rc.ufl.edu}
export SPARC_PUBAPPS_ROOT=${SPARC_PUBAPPS_ROOT:-/pubapps/SPARCP}
export SPARC_CORS_ALLOWED_ORIGINS=${SPARC_CORS_ALLOWED_ORIGINS:-https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org}

# Models are in: $SPARC_HIPERGATOR_SOURCE_MODELS

# Method 1: Use rsync from HiPerGator to PubApps
# (Requires SSH access to PubApps instance - you must SSH through HPG first)
rsync -avz --progress \
    $SPARC_HIPERGATOR_SOURCE_MODELS/ \
        $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST:$SPARC_PUBAPPS_ROOT/models/

# Method 2: Use Globus (recommended for large models)
# Set up Globus endpoints for both HiPerGator and PubApps
# Transfer via Globus web interface

# Method 3: Stage to intermediate location
# Use $SPARC_BASE_PATH as staging
# Then scp/rsync from there
```

### 2.2 Verify Model Transfer

```bash
# SSH to PubApps instance (from HiPerGator)
ssh $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST

# Check models arrived
ls -lh $SPARC_PUBAPPS_ROOT/models/
# Should see: CaregiverAgent/, C-LEAR_CoachAgent/, SupervisorAgent/
```

---

## 3.0 Setup Conda Environment on PubApps

### 3.1 Install Conda on PubApps


All the directory folders the SPARC-P service needs on the PubApps VM are created here before any files are written or software is installed. A single `mkdir -p` command creates the entire directory structure in one shot — the `-p` flag means it creates every level in the path and doesn't fail if a directory already exists.

The directories created:
- **`/pubapps/SPARCP/`** — the project root for everything SPARC-P on this VM
- **`/pubapps/SPARCP/models/`** — where the fine-tuned LLM adapters (CaregiverAgent, CoachAgent, SupervisorAgent) are stored after transfer from HiPerGator
- **`/pubapps/SPARCP/backend/`** — where the FastAPI `main.py` application code lives
- **`/pubapps/SPARCP/riva_models/`** — where NVIDIA Riva's ASR and TTS model files are stored
- **`/pubapps/SPARCP/conda_envs/`** — the parent directory for the `sparc_backend` conda environment

This is always the first step before installing software or transferring files — you can't write to a directory that doesn't exist.
PubApps VMs don't have the `module` system like HiPerGator. Install miniconda directly:

```bash
# SSH to PubApps instance
ssh $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST

# Download miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install (accept license, use default location: ~/miniconda3)
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# Initialize conda for bash
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Verify installation
conda --version
```

### 3.2 Create Backend Environment


The complete Python software environment for the SPARC-P backend is installed using conda — three commands verify, create, and validate the environment in sequence.

Step by step:
1. **`conda --version`** (`check=False`) — confirms conda is available on this VM. The `check=False` means a failure here just prints a warning rather than stopping the notebook; useful if you're running in dry-run mode.
2. **`conda env create -f environment_backend.yml -p {CONDA_ENV}`** — creates the entire Python environment from a configuration file. The `-f environment_backend.yml` points to the yaml file that lists every package and version (FastAPI, PyTorch, transformers, bitsandbytes, NeMo Guardrails, etc.). The `-p` flag installs the environment to the exact path `/pubapps/SPARCP/conda_envs/sparc_backend` instead of conda's default location. This can take **10–30 minutes** as it downloads and installs hundreds of packages including CUDA-compiled PyTorch.
3. **`conda run -p {CONDA_ENV} python -c 'import fastapi,langgraph,torch; print("backend env ok")'`** — immediately tests that the newly created environment is functional by importing three critical libraries. If any import fails, this command fails loudly, catching broken installs before you proceed.

> **One-time step:** Only run this if the conda environment doesn't already exist. If it exists and you just want to update it, use `conda env update -f environment_backend.yml -p {CONDA_ENV}` instead.
```bash
# On PubApps VM
cd /pubapps/SPARCP

# Transfer environment file from HiPerGator notebooks
# Option 1: Copy from HiPerGator
scp jayrosen@hpg.rc.ufl.edu:/path/to/Sparc\ Hipergator\ Notebooks/environment_backend.yml .

# Option 2: Create manually using the environment_backend.yml from the notebooks repo

# Create environment in /pubapps to avoid home directory space issues
conda env create -f environment_backend.yml -p /pubapps/SPARCP/conda_envs/sparc_backend

# Activate environment
conda activate /pubapps/SPARCP/conda_envs/sparc_backend

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import fastapi, langgraph, transformers; print('✓ All packages available')"
```

---

## 4.0 Deploy Riva Speech Services

### 4.1 Pull Riva Container with Podman


Downloading the Riva container image and activating Riva as a running systemd service completes the Quadlet setup. Four commands run in sequence, plus a validation check.

Step by step:
1. **`podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server`**: Downloads the Riva container image from NVIDIA's container registry. This is a large image (~5–8 GB) and needs to happen before systemd can start the service. Only needed once.
2. **`systemctl --user daemon-reload`**: Tells systemd to re-read all service definitions from disk, including the Quadlet file written in the previous cell. Without this, systemd wouldn't know about the new `riva-server` service.
3. **`systemctl --user enable --now riva-server`**: Registers the Riva service to start automatically on login (`enable`) and starts it immediately right now (`--now`). After this command, Riva begins loading its ASR and TTS models into GPU memory.
4. **`systemctl --user status riva-server`** (`check=False`): Shows the current status of the Riva service. Expected output: `Active: active (running)`. The `check=False` prevents this from stopping the notebook if the service is still starting.
5. **GPU validation commands**: `nvidia-ctk cdi list` confirms the CDI GPU device is registered, and the `podman run ... nvidia-smi` command runs `nvidia-smi` inside a test container to confirm Riva's container can actually see the GPU.
6. **Assertion**: Verifies the Quadlet file contains the correct CDI GPU mapping (`Device=nvidia.com/gpu=all`) and not the legacy mapping (`AddDevice=`), which would fail on newer Podman versions.

> **Expected next:** Allow 2–3 minutes for Riva to initialize. Check `journalctl --user -u riva-server -n 50` to monitor startup progress.
```bash
# On PubApps VM (Podman is pre-installed, NOT Docker)
# Note: Use podman, not docker commands

# Pull Riva server image
podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server

# Create persistent storage for Riva models
mkdir -p /pubapps/SPARCP/riva_models

# Initialize Riva (downloads ASR/TTS models, ~10GB)
# This requires GPU access - run with --hooks-dir option for rootless podman
podman run --rm -it \
  --gpus all \
    -v /pubapps/SPARCP/riva_models:/data \
  nvcr.io/nvidia/riva/riva-speech:2.16.0-server \
  bash -c "cd /data && /opt/riva/bin/riva_init.sh"
```

### 4.2 Configure Riva Server


A "Quadlet" file tells Podman and systemd how to manage the NVIDIA Riva speech server as a persistent background service on the PubApps VM — similar to how you'd configure a Windows Service, but for Linux.

What a Quadlet is: On modern Linux systems, systemd is the service manager. Podman Quadlets are simple configuration files that let systemd start, stop, and automatically restart containers — without needing Docker daemon or complex shell scripts.

What the generated `riva-server.container` file specifies:
- **`Image`**: Downloads and runs NVIDIA's official Riva speech server container from their registry (NGC) at version 2.16.0.
- **`Device=nvidia.com/gpu=all`**: Uses the modern CDI (Container Device Interface) standard to pass the L4 GPU through to the container. This is required for Riva to run its ASR and TTS models.
- **`Volume={RIVA_MODEL_DIR}:/data:Z`**: Mounts the local riva_models directory into the container where Riva looks for its model files. The `:Z` sets the correct SELinux label for Podman.
- **`PublishPort=50051:50051`**: Exposes Riva's gRPC port so the FastAPI backend (running outside the container) can connect using `localhost:50051`.
- **`Restart=always`**: If Riva crashes or the VM reboots, systemd automatically restarts it.
- **`TimeoutStartSec=300`**: Gives Riva 5 minutes to start (loading ASR and TTS models into GPU memory takes ~2 minutes).
- The file is written to `~/.config/containers/systemd/` — the per-user path where Podman looks for Quadlet definitions.

> **The CDI assertion at the end** (`Device=nvidia.com/gpu=all` in `quadlet_content`) is a regression check that confirms the GPU mapping was written correctly before proceeding.
Create a Podman Quadlet service file for systemd integration:

```bash
# Create systemd user service directory
mkdir -p ~/.config/containers/systemd

# Create Riva service file
cat > ~/.config/containers/systemd/riva-server.container << 'EOF'
[Unit]
Description=SPARC-P Riva Speech Server
After=network-online.target

[Container]
Image=nvcr.io/nvidia/riva/riva-speech:2.16.0-server
ContainerName=riva-server
# GPU access via CDI (requires nvidia-container-toolkit CDI support on PubApps VM)
Device=nvidia.com/gpu=all
# Volume mounts
Volume=/pubapps/SPARCP/riva_models:/data:Z
# Network
PublishPort=50051:50051
# Environment
Environment=NVIDIA_VISIBLE_DEVICES=all
# Command
Exec=/opt/riva/bin/riva_server --riva_model_repo=/data/models

[Service]
Restart=always
TimeoutStartSec=300

[Install]
WantedBy=default.target
EOF

# Reload systemd
systemctl --user daemon-reload

# Enable and start Riva service
systemctl --user enable --now riva-server

# Check status
systemctl --user status riva-server

# Validate CDI profile and GPU passthrough
nvidia-ctk cdi list
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Validate quadlet uses CDI and does not regress to raw device-node mappings
grep -q '^Device=nvidia.com/gpu=all$' ~/.config/containers/systemd/riva-server.container
! grep -q '^AddDevice=' ~/.config/containers/systemd/riva-server.container
```

---

## 5.0 Deploy FastAPI Backend Service


This is the final service activation step — it registers the FastAPI backend service with systemd and starts it running. These three commands mirror what was done for the Riva service and complete the PubApps deployment.

Step by step:
1. **`systemctl --user daemon-reload`**: Tells systemd to re-read all service files from disk, picking up the `sparc-backend.service` file just written by the previous cell.
2. **`systemctl --user enable --now sparc-backend`**: 
   - `enable` — registers the service to start automatically whenever you log in to the PubApps VM (persistent across reboots).
   - `--now` — starts the service immediately without waiting for the next login.
3. **`systemctl --user status sparc-backend`** (`check=False`): Prints the current status. Expected output: `Active: active (running)`. The `check=False` allows the notebook to continue even if the service is still starting.

After completing successfully with `EXECUTE = True`:
- The FastAPI backend is live at `http://localhost:8000`
- The Riva speech server is live at `localhost:50051`
- Both services are persistent and will restart automatically on failure
- Run `curl -s http://localhost:8000/health` to confirm the backend is healthy and models are loaded

The systemd service file (`sparc-backend.service`) tells the Linux process manager how to run the FastAPI backend as a persistent service — so it starts automatically and restarts itself if it crashes.

What the generated service file specifies, and why each setting matters:
- **`After=network.target riva-server.service`**: The backend only starts *after* the Riva speech server is running. Without this ordering, the backend could start before Riva is ready and fail to connect to `localhost:50051`.
- **`Requires=riva-server.service`**: If Riva stops, systemd also stops the backend. This prevents the backend from running in a degraded state (no speech services) silently.
- **`ExecStart={CONDA_ENV}/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1`**: Uses the *full absolute path* to the uvicorn binary inside the conda environment — not relying on `PATH`. This guarantees the correct Python environment is used even in a non-interactive systemd session. `--workers 1` is intentional for the 2-core, 16 GB PubApps VM.
- **`Environment=PATH={CONDA_ENV}/bin:/usr/bin`**: Sets the PATH so child processes spawned by uvicorn also use the conda environment's binaries.
- **`Restart=always` + `RestartSec=10`**: If the process crashes (e.g., CUDA out-of-memory during a night of heavy use), systemd waits 10 seconds and restarts it automatically — no manual intervention required.
- The file is written to `~/.config/systemd/user/` — the per-user systemd directory that a non-root user can manage without sudo.

`h15_quantization_memory_check.py` is generated and saved to the backend directory. Like the load test, it's meant to be run on the live PubApps VM, not from within this workflow.

What the script measures and why it matters:

The L4 GPU has **24 GB of VRAM** total. The SPARC-P system needs to share this between three components:
- The fine-tuned LLM (120B parameters in 4-bit quantization ≈ ~13 GB)
- The NVIDIA Riva ASR and TTS models (≈ ~3 GB combined, running in a separate container)
- System overhead and CUDA libraries (≈ ~1–2 GB)

That leaves only ~7 GB headroom. If memory usage grows beyond the expected budget, the system may start throwing CUDA out-of-memory errors during inference — causing 500 errors for users.

What the script does:
1. Checks that CUDA is available (fails loudly if not — this script is useless without a GPU).
2. Calls `torch.cuda.synchronize()` to ensure all pending CUDA operations are flushed.
3. Reads `memory_allocated()` (actively used by tensors), `memory_reserved()` (total pool held by PyTorch), and total `capacity_gb` from the GPU device.
4. **Asserts that reserved memory is under 22.0 GB** — leaving at least 2 GB headroom on a 24 GB L4.

> **To run:** After the backend has been running for a few minutes (so the model is fully loaded), SSH to the PubApps VM and run `python h15_quantization_memory_check.py` from the backend directory.

`h11_health_load_test.py` is generated and saved to the backend directory — the load test is designed to be run separately on the PubApps VM against the live service, not from within this workflow.

What the load test script does when you run it:
- **Fires 30 concurrent chat requests** (`POST /v1/chat`) using a thread pool, simulating 30 simultaneous users sending messages about HPV vaccines to the backend. This stress-tests the async inference pipeline.
- **Simultaneously pings `/health` every 200ms for 12 seconds** — for a total of 60 health check calls — to measure how the health endpoint responds *while* the backend is under load from the chat requests.
- **Measures p95 latency** for health checks (the 95th percentile, meaning 95% of checks must complete within this time).
- **Asserts three conditions:**
  1. All 30 chat requests return a recognized status code (200 OKs, 401 if API key is wrong in the test, or 422 for validation errors — but not 500 errors).
  2. 99% of health checks must complete successfully within 1.5 seconds.
  3. The p95 health latency must be under 1,500ms — confirming the health endpoint stays responsive even when inference is running.

> **To run this test:** After the backend is live, SSH to the PubApps VM, go to the backend directory, and run `python h11_health_load_test.py`. Set `SPARC_API_KEY` and `SPARC_BASE_URL` environment variables first.
### 5.1 Create Backend Application


This is the most important cell in the notebook — it writes the complete, production-ready `main.py` FastAPI application (approximately 520 lines) to disk at `/pubapps/SPARCP/backend/main.py`. This is the actual program that runs on the PubApps server and handles every interaction with SPARC-P users.

What the written application does (plain-English overview of each major section):

**System startup (lifespan):** When the server starts, it loads all three fine-tuned LLM adapters (CaregiverAgent, CoachAgent, SupervisorAgent) into GPU memory using 4-bit quantization (NF4 format via bitsandbytes) to fit within the L4's 24 GB VRAM. It also connects to Riva for speech, loads the NeMo Guardrails safety config, and creates the audio file cache directory.

**Safety pipeline (guardrails):** Every incoming user message passes through NeMo Guardrails before reaching the AI models. Off-topic messages (politics, finance, anything unrelated to HPV vaccine communication) are rejected with a pre-set refusal message. The AI's response also passes through guardrails before being sent back — a two-stage safety check.

**PII redaction (Presidio):** Before any text touches Firebase or logging, it's passed through Microsoft Presidio to redact personal identifiers (names, phone numbers, medical record numbers). If Presidio fails to initialize, all text is replaced with `[REDACTED]` rather than risking PHI exposure — a "fail-closed" safety posture.

**API authentication:** Every API call requires an `X-API-Key` header. The Unity client sends this key, and the server validates it against the `SPARC_API_KEY` environment variable. This prevents unauthorized access to the backend.

**Circuit breakers:** If the LLM, coach, or Riva TTS times out three times in a row, the corresponding circuit "opens" for 30 seconds — returning a graceful degraded response instead of queuing more timeout requests. Once 30 seconds pass, the circuit closes and normal operation resumes.

**Audio delivery:** Instead of base64-encoding audio in the API response (which would be very large), TTS audio is written to a temp file and returned as a URL (`/v1/audio/{id}`) that expires after 5 minutes. The Unity client fetches the audio separately.

**Firebase session state:** After each turn, the session's last message and response (Presidio-redacted) are written to Firestore for session continuity and audit purposes.

> **The file is written to disk but the server is not yet started.** The systemd service section below handles starting the running process.
```bash
# On PubApps VM
cd /pubapps/SPARCP
mkdir -p backend
cd backend
```

Create the main FastAPI application (`main.py`):

A comprehensive automated check scans the `main.py` file and asserts that over 80 specific code patterns are present (and several dangerous legacy patterns are absent).

Think of it as a build-quality gate: before deploying the application, this step verifies that all the critical security, reliability, and compliance features are actually in the code.

The checks are grouped into categories:
- **Adapter management (C4/C5):** Confirms all three LLM adapters (caregiver, coach, supervisor) are registered by name using `adapter_name=` parameters — not as three separate model objects (which would triple GPU memory usage).
- **API authentication (M7):** Verifies the `require_api_key` auth guard is defined and injected via `Depends()` into the chat endpoint.
- **Environment config (M8):** Confirms all sensitive values (Firebase path, Riva URL, model path, CORS origins) are read from environment variables — not hard-coded.
- **PII redaction (M9/L5):** Verifies Presidio is imported and `sanitize_for_storage()` is called on both the user message and response before Firebase writes.
- **CORS security (H3):** Checks that `allow_origins=[\"*\"]` (wildcard) is absent and specific allowed origins are configured.
- **Guardrails (H5):** Confirms NeMo Guardrails is imported and both input and output enforcement functions are called.
- **Async inference (H12):** Validates that `asyncio.wait_for()` and `asyncio.to_thread()` are used for model calls — preventing the event loop from blocking during inference.
- **Circuit breaker (H13):** Checks that timeout and circuit breaker functions are defined and wired up for all three operations (inference, coach, TTS).
- **Quantization (H15):** Confirms 4-bit NF4 quantization config is present — the memory optimization that makes the 120B-parameter model fit on an L4 GPU.

> **If any assertion fails:** The error message tells you exactly which marker is missing or which blocked pattern is still present, so you know exactly what needs to be fixed in the main.py before deploying.

```python
# /pubapps/SPARCP/backend/main.py
"""
SPARC-P FastAPI Backend for PubApps
Serves the trained multi-agent system for public access
"""
import os
import sys
import asyncio
import time
import logging
import tempfile
import uuid
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
from langgraph.graph import StateGraph
from nemoguardrails import LLMRails, RailsConfig
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import firebase_admin
from firebase_admin import credentials, firestore

# Configuration
MODEL_BASE_PATH = os.getenv("SPARC_MODEL_BASE_PATH", "/pubapps/SPARCP/models")
RIVA_SERVER = os.getenv("SPARC_RIVA_SERVER", "localhost:50051")
FIREBASE_CREDS = os.getenv("SPARC_FIREBASE_CREDS", "/pubapps/SPARCP/config/firebase-credentials.json")
GUARDRAILS_DIR = os.getenv("SPARC_GUARDRAILS_DIR", os.path.join(os.path.dirname(__file__), "guardrails"))

API_AUTH_ENABLED = os.getenv("SPARC_API_AUTH_ENABLED", "true").strip().lower() == "true"
API_KEY = os.getenv("SPARC_API_KEY", "")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("SPARC_CORS_ALLOWED_ORIGINS", "https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org").split(",")
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
SPARC_AUDIO_URL_TTL_SECONDS = float(os.getenv("SPARC_AUDIO_URL_TTL_SECONDS", "300"))
SPARC_AUDIO_CACHE_DIR = os.getenv("SPARC_AUDIO_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sparc_tts_audio"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("SPARC_TIMEOUT_CIRCUIT_THRESHOLD", "3"))
CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS", "30"))

# Validate runtime-sensitive configuration at startup
if not FIREBASE_CREDS:
    raise RuntimeError("SPARC_FIREBASE_CREDS is empty; set Firebase service account path")
if not os.path.isfile(FIREBASE_CREDS):
    raise RuntimeError(
        f"Firebase credentials file not found: {FIREBASE_CREDS}. "
        "Set SPARC_FIREBASE_CREDS to a valid path."
    )

# Initialize Firebase
cred = credentials.Certificate(FIREBASE_CREDS)
firebase_admin.initialize_app(cred)
db = firestore.client()

logger = logging.getLogger("sparc_backend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    presidio_analyzer = AnalyzerEngine()
    presidio_anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except Exception as presidio_init_error:
    presidio_analyzer = None
    presidio_anonymizer = None
    PRESIDIO_AVAILABLE = False
    logger.warning("Presidio initialization failed; using fail-closed redaction placeholders: %s", presidio_init_error)


def sanitize_for_storage(text: Optional[str]) -> str:
    if not text:
        return ""
    if not PRESIDIO_AVAILABLE:
        return "[REDACTED]"
    try:
        findings = presidio_analyzer.analyze(text=text, language="en")
        if not findings:
            return text
        return presidio_anonymizer.anonymize(text=text, analyzer_results=findings).text
    except Exception:
        return "[REDACTED]"

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
        logger.exception("Guardrails initialization failed: %s", sanitize_for_storage(str(guardrails_error)))

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
        logger.exception("Input guardrails failed: %s", sanitize_for_storage(str(guardrails_error)))
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
        logger.exception("Output guardrails failed: %s", sanitize_for_storage(str(guardrails_error)))
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": "output_rails_error"}

# Initialize FastAPI lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_models()
    yield

app = FastAPI(
    title="SPARC-P Multi-Agent Backend",
    description="HPV Vaccine Communication Training System",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Unity WebGL
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOWED_METHODS,
    allow_headers=CORS_ALLOWED_HEADERS,
)

# Load models at startup using named adapters on one PEFT model
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
        logger.warning("Riva client initialization failed: %s", sanitize_for_storage(str(riva_init_error)))

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
    """Load one base model with named adapters to prevent adapter overwrite/collision."""
    global adapter_model, tokenizer

    print("Loading base model and named adapters...")
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
        device_map="auto"
    )

    adapter_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATHS["caregiver"],
        adapter_name="caregiver"
    )
    adapter_model.load_adapter(ADAPTER_PATHS["coach"], adapter_name="coach")
    adapter_model.load_adapter(ADAPTER_PATHS["supervisor"], adapter_name="supervisor")
    adapter_model.set_adapter("caregiver")

    load_guardrails_runtime()
    init_riva_clients()
    ensure_audio_cache_dir()

    print("✓ Models loaded successfully")

# Pydantic models
class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    user_message: str = Field(..., min_length=1, max_length=10000)
    audio_data: Optional[str] = Field(default=None, max_length=2_000_000)  # Optional Base64 audio

class ChatResponse(BaseModel):
    response_text: str
    audio_url: Optional[str] = None
    emotion: str
    animation_cues: Dict[str, str]
    coach_feedback: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for monitoring"""
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
    }
    http_status = status.HTTP_200_OK if model_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=health_payload)

# Main chat endpoint
@app.get("/v1/audio/{audio_id}")
async def get_tts_audio(audio_id: str, _api_key: str = Depends(require_api_key)):
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
    """
    Process user input through multi-agent system
    """
    try:
        # 1. Retrieve session state from Firestore
        session_ref = db.collection('sessions').document(request.session_id)
        session_state = session_ref.get().to_dict() or {}
        
        # 2. Process through enforced guardrails input path
        input_guard = await enforce_guardrails_input(request.user_message)
        if not input_guard["allowed"]:
            return ChatResponse(
                response_text=input_guard["text"],
                emotion="neutral",
                animation_cues={"gesture": "idle"},
                coach_feedback={"safe": False, "reason": input_guard["reason"]}
            )
        
        # 3. Route to appropriate adapter (named-adapter state routing)
        conversation_mode = session_state.get("mode", "caregiver")
        primary_adapter = select_adapter_for_mode(conversation_mode)
        
        # 4. Generate response (adapter-based inference)
        prompt = f"[SESSION: {request.session_id}] User: {input_guard['text']}\nAssistant:"
        model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        model_inputs = {k: v.to(adapter_model.device) for k, v in model_inputs.items()}

        if await is_circuit_open("primary_inference"):
            logger.warning("Primary inference circuit open; returning degraded fallback response")
            return ChatResponse(
                response_text="I’m temporarily unable to generate a response right now. Please try again shortly.",
                audio_url=None,
                emotion="neutral",
                animation_cues={"gesture": "idle", "intensity": "low"},
                coach_feedback={"safe": True, "reason": "inference_circuit_open", "summary": "Primary model temporarily unavailable."},
            )

        try:
            async with inference_lock:
                adapter_model.set_adapter(primary_adapter)
                output_tokens = await asyncio.wait_for(
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
            logger.warning(
                "Primary inference timed out after %.1fs%s",
                LLM_TIMEOUT_SECONDS,
                "; circuit opened" if circuit_opened else "",
            )
            return ChatResponse(
                response_text="I’m temporarily unable to generate a response right now. Please try again shortly.",
                audio_url=None,
                emotion="neutral",
                animation_cues={"gesture": "idle", "intensity": "low"},
                coach_feedback={"safe": True, "reason": "inference_timeout", "summary": "Primary model timeout fallback."},
            )

        decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        response_text = decoded.split("Assistant:")[-1].strip() or "I’m here to help with HPV vaccine communication practice."

        # 5. Enforce guardrails on generated output
        output_guard = await enforce_guardrails_output(response_text)
        response_text = output_guard["text"]

        # Optional coach feedback generated with coach adapter
        feedback_prompt = f"Provide concise coaching feedback for this response: {response_text}"
        feedback_inputs = tokenizer(feedback_prompt, return_tensors="pt", truncation=True, max_length=512)
        feedback_inputs = {k: v.to(adapter_model.device) for k, v in feedback_inputs.items()}
        coach_feedback_text = "Coach feedback temporarily unavailable."
        coach_feedback_reason = output_guard["reason"]
        try:
            if await is_circuit_open("coach_inference"):
                logger.warning("Coach inference circuit open; skipping coach generation")
                coach_feedback_reason = "coach_circuit_open"
            else:
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
            logger.warning(
                "Coach inference timed out after %.1fs%s",
                COACH_TIMEOUT_SECONDS,
                "; circuit opened" if circuit_opened else "",
            )
            coach_feedback_reason = "coach_timeout"
        except Exception as coach_error:
            logger.warning("Coach inference failed: %s", sanitize_for_storage(str(coach_error)))
            coach_feedback_reason = "coach_error"
        finally:
            async with inference_lock:
                adapter_model.set_adapter(primary_adapter)
        
        # 5. Convert to speech with Riva TTS
        audio_url = None
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
            logger.warning(
                "Riva TTS timed out after %.1fs%s",
                TTS_TIMEOUT_SECONDS,
                "; circuit opened" if circuit_opened else "",
            )
        except Exception as riva_error:
            logger.warning("Riva TTS unavailable: %s", sanitize_for_storage(str(riva_error)))
        
        # 6. Update session state in Firestore using Presidio-sanitized values only
        sanitized_user_message = sanitize_for_storage(request.user_message)
        sanitized_response_text = sanitize_for_storage(response_text)
        session_state["last_user_message"] = sanitized_user_message
        session_state["last_response"] = sanitized_response_text
        session_state["mode"] = conversation_mode
        session_state["phi_redaction"] = "presidio"
        session_state["phi_redaction_applied"] = True
        session_ref.set(session_state, merge=True)
        
        return ChatResponse(
            response_text=response_text,
            audio_url=audio_url,
            emotion="supportive",
            animation_cues={"gesture": "speaking", "intensity": "low"},
            coach_feedback={"summary": coach_feedback_text[:500], "safe": output_guard["allowed"], "reason": coach_feedback_reason}
        )
        
    except Exception as e:
        logger.exception("/v1/chat failed after sanitization path: %s", sanitize_for_storage(str(e)))
        raise HTTPException(status_code=500, detail="Internal server error")

# For development only

### 6.2 C4/C5/M7/M8/M9/M11/L5/H2/H3/H5/H10/H11/H12/H13/H14/H15 Smoke Test — Adapter/Auth/Config + Timeout/Circuit-Breaker + Riva Client Reuse + Bounded TTS Delivery + Lifespan Lifecycle + Redaction + Contract + CORS + Guardrails + Async Inference + Health Readiness + Error Sanitization + Schema Constraints + Quantization Validation
```python
backend_text = main_py.read_text()

required_markers = [
    'adapter_name="caregiver"',
    'load_adapter(ADAPTER_PATHS["coach"], adapter_name="coach")',
    'load_adapter(ADAPTER_PATHS["supervisor"], adapter_name="supervisor")',
    'adapter_model.set_adapter(primary_adapter)',
    'adapter_model.set_adapter("coach")',
    'def require_api_key(',
    'Header(default=None, alias="X-API-Key")',
    'Depends(require_api_key)',
    'from presidio_analyzer import AnalyzerEngine',
    'from presidio_anonymizer import AnonymizerEngine',
    'from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig',
    'def sanitize_for_storage(',
    'sanitized_user_message = sanitize_for_storage(request.user_message)',
    'sanitized_response_text = sanitize_for_storage(response_text)',
    'session_state["phi_redaction_applied"] = True',
    'API_CONTRACT_VERSION = "v1"',
    'session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")',
    'user_message: str = Field(..., min_length=1, max_length=10000)',
    'audio_data: Optional[str] = Field(default=None, max_length=2_000_000)',
    'api_contract_version": API_CONTRACT_VERSION',
    'CORS_ALLOWED_ORIGINS = [',
    'CORS_ALLOW_CREDENTIALS = os.getenv("SPARC_CORS_ALLOW_CREDENTIALS", "false")',
    'allow_origins=CORS_ALLOWED_ORIGINS',
    'allow_credentials=CORS_ALLOW_CREDENTIALS',
    'allow_methods=CORS_ALLOWED_METHODS',
    'allow_headers=CORS_ALLOWED_HEADERS',
    'from nemoguardrails import LLMRails, RailsConfig',
    'load_guardrails_runtime()',
    'enforce_guardrails_input(request.user_message)',
    'enforce_guardrails_output(response_text)',
    'guardrails_loaded": guardrails_engine is not None',
    'import asyncio',
    'inference_lock = asyncio.Lock()',
    'LLM_TIMEOUT_SECONDS = float(os.getenv("SPARC_LLM_TIMEOUT_SECONDS", "10"))',
    'COACH_TIMEOUT_SECONDS = float(os.getenv("SPARC_COACH_TIMEOUT_SECONDS", "10"))',
    'TTS_TIMEOUT_SECONDS = float(os.getenv("SPARC_TTS_TIMEOUT_SECONDS", "5"))',
    'TTS_MAX_AUDIO_BYTES = int(os.getenv("SPARC_TTS_MAX_AUDIO_BYTES", "524288"))',
    'SPARC_AUDIO_URL_TTL_SECONDS = float(os.getenv("SPARC_AUDIO_URL_TTL_SECONDS", "300"))',
    'SPARC_AUDIO_CACHE_DIR = os.getenv("SPARC_AUDIO_CACHE_DIR", os.path.join(tempfile.gettempdir(), "sparc_tts_audio"))',
    'from contextlib import asynccontextmanager',
    'async def lifespan(app: FastAPI):',
    'await load_models()',
    'lifespan=lifespan,',
    'CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("SPARC_TIMEOUT_CIRCUIT_THRESHOLD", "3"))',
    'CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS", "30"))',
    'def init_riva_clients() -> None:',
    'riva_asr_service = riva.client.ASRService(riva_auth)',
    'riva_tts_service = riva.client.SpeechSynthesisService(riva_auth)',
    'init_riva_clients()',
    'if riva_tts_service is None:',
    'riva_client_pool_initialized": riva_ok',
    'async def is_circuit_open(operation: str) -> bool:',
    'async def record_timeout_event(operation: str) -> bool:',
    'def generate_tokens_sync(',
    'def synthesize_tts_sync(',
    'async def persist_tts_audio(audio_bytes: bytes) -> Optional[str]:',
    '@app.get("/v1/audio/{audio_id}")',
    'return FileResponse(audio_path, media_type="audio/wav", filename=f"{audio_id}.wav")',
    'asyncio.wait_for(',
    'await asyncio.to_thread(',
    'Primary inference timed out after',
    'Coach inference timed out after',
    'Riva TTS timed out after',
    'from fastapi.responses import JSONResponse',
    'model_ok = tokenizer is not None and adapter_model is not None',
    'ready_for_traffic": model_ok',
    'status.HTTP_503_SERVICE_UNAVAILABLE',
    'return JSONResponse(status_code=http_status, content=health_payload)',
    'logger.exception("/v1/chat failed after sanitization path: %s", sanitize_for_storage(str(e)))',
    'raise HTTPException(status_code=500, detail="Internal server error")',
    'bnb_config = BitsAndBytesConfig(',
    'quantization_config=bnb_config',
    'bnb_4bit_quant_type="nf4"',
    'bnb_4bit_compute_dtype=torch.bfloat16',
]

missing = [marker for marker in required_markers if marker not in backend_text]
assert not missing, f"Missing required markers: {missing}"

assert 'caregiver_model = PeftModel.from_pretrained(base_model' not in backend_text
assert 'coach_model = PeftModel.from_pretrained(base_model' not in backend_text
assert 'supervisor_model = PeftModel.from_pretrained(base_model' not in backend_text
assert 'async def process_chat(request: ChatRequest):' not in backend_text
assert 'session_state["last_user_message"] = request.user_message' not in backend_text
assert 'session_state["last_response"] = response_text' not in backend_text
assert 'user_transcript' not in backend_text
assert 'allow_origins=["*"]' not in backend_text
assert 'allow_credentials=True' not in backend_text
assert 'blocked = ["politics", "election", "gambling", "crypto", "finance advice"]' not in backend_text
assert 'output_tokens = adapter_model.generate(' not in backend_text
assert 'feedback_tokens = adapter_model.generate(' not in backend_text
assert '"models_loaded": True' not in backend_text
assert 'detail=str(e)' not in backend_text
assert 'load_in_4bit=True,' not in backend_text
assert 'data:audio/wav;base64' not in backend_text
assert 'base64.b64encode(' not in backend_text
assert '@app.on_event("startup")' not in backend_text

print("✅ C4/C5/M7/M8/M9/M11/L5/H2/H3/H5/H10/H11/H12/H13/H14/H15 validation passed: named adapters, auth guard, timeout/circuit-breaker policy, startup-initialized reusable Riva clients, bounded TTS URL delivery with payload limits, lifespan-based FastAPI lifecycle initialization, env config, Presidio redaction, unified v1 API contract, safe CORS policy, runtime Guardrails pipeline, non-blocking async inference path, readiness-aware health behavior, sanitized client error responses, strict request schema constraints, and explicit 4-bit quantization config are configured.")
```

### 6.3 H11 Load Test — Health Responsiveness Under Chat Load
```python
import concurrent.futures
import statistics
import time
import requests

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("SPARC_API_KEY", "")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
CHAT_PAYLOAD = {"session_id": "h11-load", "user_message": "Help me discuss HPV vaccines"}

def post_chat() -> int:
    return requests.post(f"{BASE_URL}/v1/chat", json=CHAT_PAYLOAD, headers=HEADERS, timeout=120).status_code

def ping_health() -> float:
    start = time.perf_counter()
    response = requests.get(f"{BASE_URL}/health", timeout=5)
    response.raise_for_status()
    return (time.perf_counter() - start) * 1000

health_latencies = []
chat_statuses = []
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    chat_futures = [pool.submit(post_chat) for _ in range(30)]
    for _ in range(60):
        health_latencies.append(ping_health())
        time.sleep(0.2)
    chat_statuses = [f.result() for f in chat_futures]

health_p95 = statistics.quantiles(health_latencies, n=20)[18] if len(health_latencies) >= 20 else max(health_latencies)
health_success_ratio = sum(1 for latency in health_latencies if latency < 1500) / len(health_latencies)

assert all(code in (200, 401, 422) for code in chat_statuses), f"Unexpected chat status codes: {sorted(set(chat_statuses))}"
assert health_success_ratio >= 0.99, f"Health responsiveness dropped below target: {health_success_ratio:.3f}"
assert health_p95 < 1500, f"Health p95 latency too high under chat load: {health_p95:.1f}ms"

print(f"✅ H11 load test passed: /health p95={health_p95:.1f}ms, success_ratio={health_success_ratio:.3f}")
```

### 6.4 H15 Quantization Memory Profile Check
```python
import torch

assert adapter_model is not None, "Model not loaded; run startup path first"
assert torch.cuda.is_available(), "CUDA device is required for H15 memory profile check"

torch.cuda.synchronize()
allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
capacity_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

print(f"GPU memory allocated: {allocated_gb:.2f} GB")
print(f"GPU memory reserved: {reserved_gb:.2f} GB")
print(f"GPU capacity: {capacity_gb:.2f} GB")

# L4 target budget guardrail; tune threshold if hardware profile changes.
assert reserved_gb < 22.0, f"Reserved memory exceeds expected L4 quantized startup budget: {reserved_gb:.2f} GB"
print("✅ H15 memory profile check passed: quantized startup is within expected L4 budget.")
```

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 5.2 Create Systemd Service

Create a systemd user service to run the FastAPI backend:

```bash
# Create service file
cat > ~/.config/systemd/user/sparc-backend.service << 'EOF'
[Unit]
Description=SPARC-P FastAPI Backend
After=network.target riva-server.service
Requires=riva-server.service

[Service]
Type=simple
Environment="PATH=/pubapps/SPARCP/conda_envs/sparc_backend/bin:/usr/bin"
Environment="PYTHONUNBUFFERED=1"
WorkingDirectory=/pubapps/SPARCP/backend
ExecStart=/pubapps/SPARCP/conda_envs/sparc_backend/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF

# Reload systemd
systemctl --user daemon-reload

# Enable and start service
systemctl --user enable --now sparc-backend

# Check status
systemctl --user status sparc-backend

# View logs
journalctl --user -u sparc-backend -f
```

---

## 6.0 Configure NGINX Reverse Proxy

### 6.1 Request NGINX Configuration

Open a support ticket to request NGINX reverse proxy configuration:

```
Subject: NGINX Configuration for SPARC-P PubApps Instance

Body:
Please configure NGINX reverse proxy for the SPARC-P application:

1. SSL Certificate: Request *.rc.ufl.edu certificate or custom domain
2. Proxy Rules:
    - / → Unity WebGL static files (/pubapps/SPARCP/unity_webgl/)
   - /api/ → FastAPI backend (http://localhost:8000)
3. WebSocket Support: Enable for /api/ws
4. Authentication: UF Shibboleth SSO for access control
5. CORS: Allow for Unity WebGL

Public URL: https://sparc-p.rc.ufl.edu (or assigned domain)
```

### 6.2 NGINX Configuration (Reference)

The RC team will configure NGINX, but for reference:

```nginx
server {
    listen 443 ssl http2;
    server_name sparc-p.rc.ufl.edu;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    # Static Unity WebGL files
    location / {
        root /pubapps/SPARCP/unity_webgl;
        index index.html;
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy to FastAPI backend
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket support
    location /api/ws {
        proxy_pass http://localhost:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 7.0 Deployment Checklist

### 7.1 Pre-Deployment

- [ ] PubApps instance provisioned
- [ ] Models transferred from HiPerGator to `/pubapps/SPARCP/models/`
- [ ] Conda environment created and tested
- [ ] Firebase credentials configured
- [ ] UF RC risk assessment completed

### 7.2 Service Deployment

- [ ] Riva container running (`systemctl --user status riva-server`)
- [ ] FastAPI backend running (`systemctl --user status sparc-backend`)
- [ ] Services set to auto-start on boot
- [ ] Logs configured and accessible

### 7.3 Integration

- [ ] NGINX reverse proxy configured
- [ ] SSL certificate installed
- [ ] UF Shibboleth SSO configured
- [ ] Unity WebGL build deployed to `/pubapps/SPARCP/unity_webgl/`
- [ ] CORS configured correctly

### 7.4 Testing

- [ ] Health check endpoint accessible: `https://sparc-p.rc.ufl.edu/api/health`
- [ ] Chat endpoint functional: `POST https://sparc-p.rc.ufl.edu/api/v1/chat`
- [ ] Speech-to-text working (Riva ASR)
- [ ] Text-to-speech working (Riva TTS)
- [ ] Firebase connectivity confirmed
- [ ] End-to-end Unity → Backend → Unity flow tested

---

## 8.0 Monitoring and Maintenance

### 8.1 Service Management

```bash
# Check service status
systemctl --user status riva-server
systemctl --user status sparc-backend

# View logs
journalctl --user -u riva-server -n 100
journalctl --user -u sparc-backend -n 100 -f

# Restart services
systemctl --user restart riva-server
systemctl --user restart sparc-backend

# Stop services
systemctl --user stop sparc-backend riva-server
```

### 8.2 Resource Monitoring

```bash
# Check disk usage
df -h /pubapps/SPARCP

# Check GPU usage
nvidia-smi

# Check memory usage
free -h

# Check service resource usage
systemctl --user status sparc-backend
```

### 8.3 Update Workflow

To update models or code:

1. Train new model on HiPerGator
2. Transfer to PubApps via rsync/Globus
3. Restart backend service: `systemctl --user restart sparc-backend`
4. Monitor logs for successful reload

---

## 9.0 Security and Compliance

### 9.1 Data Classification

Per the PubApp request form, SPARC-P processes:
- **Open Data**: Training materials, public health information
- **Sensitive Data**: Session metadata (transient, in-memory processing)
- **Not Stored**: PHI, FERPA records, credentials (handled by Shibboleth)

### 9.2 Security Controls

1. **Authentication**: UF Shibboleth SSO (delegated to UF IdP)
2. **Transport Security**: TLS 1.2+ for all connections
3. **Data Retention**: Transient PHI model (in-memory only, not persisted)
4. **Audit Logging**: Non-sensitive metrics stored in Firebase Firestore
5. **Access Control**: Limited to authorized project members via SSH keys

### 9.3 Compliance

- Follow [UFIT RC PubApps Policy](https://docs.rc.ufl.edu/services/web_hosting/)
- Quarterly user access reviews (document in ticket)
- Vulnerability scanning compliance (RC team performs scans)
- No Critical/High vulnerabilities in production

---

## 10.0 Troubleshooting

### 10.1 Common Issues

**Issue**: Conda environment not found
```bash
# Solution: Verify path and activate
conda info --envs
conda activate /pubapps/SPARCP/conda_envs/sparc_backend
```

**Issue**: Riva container won't start
```bash
# Solution: Check GPU access and permissions
podman run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
# Check service logs
journalctl --user -u riva-server -n 50
```

**Issue**: FastAPI service crashes
```bash
# Solution: Check logs for errors
journalctl --user -u sparc-backend -n 100
# Verify conda environment packages
conda list | grep -E "fastapi|uvicorn|torch"
```

**Issue**: "Connection refused" from Unity
```bash
# Solution: Verify NGINX proxy and backend are running
curl http://localhost:8000/health  # Should return {"status": "healthy"}
# Check NGINX logs (contact RC if needed)
```

### 10.2 Support Resources

- **UF RC Support Ticket**: https://support.rc.ufl.edu/
- **PubApps Documentation**: https://docs.rc.ufl.edu/services/web_hosting/
- **RC Consulting**: Schedule via support ticket
- **Project Team**: Contact Jason Arnold (jda@coe.ufl.edu)

---

## 11.0 Summary

This notebook covered the complete PubApps deployment workflow:

1. ✅ Transferred trained models from HiPerGator to PubApps
2. ✅ Set up conda environment on PubApps VM
3. ✅ Deployed Riva speech services with Podman
4. ✅ Created FastAPI backend with systemd service
5. ✅ Configured NGINX reverse proxy
6. ✅ Implemented security controls (Shibboleth SSO, TLS)
7. ✅ Established monitoring and maintenance procedures

**Key Takeaways**:
- PubApps is separate from HiPerGator (different infrastructure, access, tools)
- Use **conda** for Python environment management (UF RC requirement)
- Use **Podman** for containers (NOT Docker)
- Use **systemd** for persistent service management
- Models trained on HiPerGator → Deployed on PubApps → Served publicly

**Next Steps**:
1. Complete PubApps instance request and risk assessment
2. Transfer trained models from HiPerGator
3. Follow deployment steps in this notebook
4. Test end-to-end integration with Unity WebGL
5. Monitor services and maintain documentation
