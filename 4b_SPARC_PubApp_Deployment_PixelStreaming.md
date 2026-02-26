# SPARC-P PubApp Deployment Guide (Pixel Streaming / Server-Side Rendering)

## Overview

This notebook provides step-by-step instructions for deploying SPARC-P on **UF RC PubApps** using **server-side rendering** for thin clients. Instead of serving Unity WebGL assets to the browser, Unity runs on the PubApps GPU and streams rendered video/audio to the browser through WebRTC.

### Key Changes from the WebGL Deployment

1. **Audio2Face removed** to free GPU memory (~2GB VRAM reclaimed).
2. **Unity Linux Server Build** runs in a container on PubApps and performs server-side rendering.
3. **Signaling Server** is added as a lightweight Node.js container for WebRTC negotiation.
4. **Single L4 VRAM budget** is enforced across all runtime services.

---

## 1.0 Prerequisites

### 1.1 Required Allocations

Before deploying to PubApps, ensure you have:

1. **PA-Instance allocation** ($300/year) - Request via [HiPerGator Service Purchase Form](https://it.ufl.edu/rc/get-started/purchase-request/)
2. **PA-GPU allocation** - Single NVIDIA L4 (24GB VRAM)
3. **Completed Risk Assessment** - See [Risk Assessment Documentation](https://docs.rc.ufl.edu/services/web_hosting/risk_assessment/)

### 1.2 PubApp Instance Setup

After purchasing a PA-Instance, open a support ticket to provision your instance:
- Instance accessible via SSH from HiPerGator
- Project account provisioned (commonly `SPARCP`)
- Default resources: 1x L4 (24GB), 2 vCPUs, 16GB RAM, 1TB `/pubapps` storage

### 1.3 Strict L4 VRAM Budget
This is the configuration cell for the Pixel Streaming deployment variant — it sets up all paths and environment variables, and also defines the VRAM allocation plan that makes the entire multi-service setup fit on a single 24 GB L4 GPU.

What's unique to this notebook vs. Notebook 4:
- **`QUADLET_DIR`**: Points to `~/.config/containers/systemd/` — the directory where Podman Quadlet unit files are stored. This notebook writes five Quadlet files (pod + four containers), all in this directory.
- **`VRAM_PLAN` dictionary**: This is a budgeting document baked into code. It documents how the 24 GB L4 VRAM is allocated across the four GPU-using services:
  - **LLM (4-bit quantized) ≈ 13 GB** — the fine-tuned language model adapter
  - **Unity render server ≈ 3.5 GB** — rendering the 3D avatar at 1080p for WebRTC streaming
  - **Riva embedded (ASR + TTS) ≈ 3 GB** — speech recognition and synthesis models
  - **Buffer ≈ 2–3 GB** — for CUDA overhead, frame buffers, and peak demand headroom
  
  Total: approximately 21.5–22.5 GB, staying within the 24 GB limit with modest headroom. If you add more features or increase model size, revisit this plan first.

- **`PUBAPP_ALLOWED_ORIGINS`**: Same CORS restriction as Notebook 4 — only `hpvcommunicationtraining.com` and `.org` can make API calls.
 (24GB)

Use this budget to prevent OOM and unstable services:

- **LLM (vLLM, `gpt-oss-20b`, 4-bit quantized)**: ~13GB
- **Unity Linux Server Rendering**: ~3.5GB
- **Riva TTS/ASR (Embedded config)**: ~3GB
- **System/driver/runtime overhead buffer**: ~2-3GB

**Enforcement guidance**:
- Start services in this order: Riva -> LLM -> Signaling -> Unity
- Verify with `nvidia-smi` after each startup
- If usage exceeds budget, reduce Unity quality profile and/or LLM KV cache limits

### 1.4 Architecture Overview (Server-Side Rendering)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             Public Browser                              │
│                 (thin client, no local Unity/WebGL)                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ HTTPS/WSS
┌───────────────────────────────▼──────────────────────────────────────────┐
│                              NGINX                                       │
│              TLS termination + reverse proxy routing                     │
└───────────────┬───────────────────────────────────────────────┬──────────┘
                │                                               │
         /signal /ws                                     /api/*
                │                                               │
┌───────────────▼──────────────┐                    ┌──────────▼───────────┐
│ Node.js Signaling Container  │                    │ FastAPI + LLM (vLLM) │
│ WebRTC session negotiation   │                    │ gpt-oss-20b 4-bit    │
└───────────────┬──────────────┘                    └──────────┬───────────┘
                │                                               │
                │ WebRTC setup                                  │ localhost
                │                                               │
┌───────────────▼───────────────────────────────────────────────▼───────────┐
│                    Unity Linux Server Container                            │
│         (server-side rendering + Render Streaming package)                 │
│                local calls to Riva + backend services                      │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │
                         ┌──────▼──────┐
                         │ Riva Server │
                         │ ASR + TTS   │
                         └─────────────┘
```

Data flow: Browser connects to the **Signaling Server** -> signaling negotiates WebRTC -> browser receives A/V stream rendered by **Unity container** -> Unity exchanges local requests with **FastAPI/LLM** and **Riva**.

---

## 2.0 Transfer Trained Models from HiPerGator

### 2.1 Sync Models to PubApps Storage

```bash
# On HiPerGator
export SPARC_BASE_PATH=${SPARC_BASE_PATH:-/blue/jasondeanarnold/SPARCP}
export SPARC_HIPERGATOR_SOURCE_MODELS=${SPARC_HIPERGATOR_SOURCE_MODELS:-$SPARC_BASE_PATH/trained_models}
export SPARC_PUBAPPS_SSH_USER=${SPARC_PUBAPPS_SSH_USER:-SPARCP}
export SPARC_PUBAPPS_HOST=${SPARC_PUBAPPS_HOST:-pubapps-vm.rc.ufl.edu}
export SPARC_PUBAPPS_ROOT=${SPARC_PUBAPPS_ROOT:-/pubapps/SPARCP}
export SPARC_CORS_ALLOWED_ORIGINS=${SPARC_CORS_ALLOWED_ORIGINS:-https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org}

rsync -avz --progress \
    $SPARC_HIPERGATOR_SOURCE_MODELS/ \
    $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST:$SPARC_PUBAPPS_ROOT/models/
```

### 2.2 Verify Model Transfer

```bash
ssh $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST
ls -lh $SPARC_PUBAPPS_ROOT/models/
# Expected: CaregiverAgent/, C-LEAR_CoachAgent/, SupervisorAgent/
```

---

## 3.0 Setup Conda Environment (Backend + vLLM)

### 3.1 Install Conda

```bash
ssh $SPARC_PUBAPPS_SSH_USER@$SPARC_PUBAPPS_HOST
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version
```

### 3.2 Create Runtime Environment

```bash
cd $SPARC_PUBAPPS_ROOT
conda env create -f environment_backend.yml -p $SPARC_PUBAPPS_ROOT/conda_envs/sparc_backend
conda activate $SPARC_PUBAPPS_ROOT/conda_envs/sparc_backend
python -c "import torch, fastapi; print(torch.__version__)"
```

---

## 4.0 Build and Stage Containers

### 4.1 Required Runtime Images

Build or pull these images before service enablement:

1. `sparc/llm-backend:latest` (FastAPI + vLLM)
2. `sparc/unity-server:latest` (Unity Linux server build with Render Streaming)
3. `sparc/signaling-server:latest` (Unity Render Streaming signaling server)
4. `nvcr.io/nvidia/riva/riva-speech:2.16.0-server`

### 4.2 Reference Build Commands


The complete sequence of commands needed to build and pull all four container images required by the Pixel Streaming deployment is printed below. These are reference commands — copy and run them in your terminal on the PubApps VM (or wherever you're building Docker images).

The commands in order:
1. **`mkdir -p artifacts/unity/LinuxServer artifacts/signaling`** — creates the directories where you place your Unity Linux Server build files and signaling server package files before building images.
2. **Two comment lines** — reminders to manually copy your Unity build output and Node.js signaling files into these directories before the build steps.
3. **`podman build -f Dockerfile.mas`** → builds `sparc/llm-backend:latest` — the Python FastAPI + LLM backend image from `Dockerfile.mas`.
4. **`podman build -f Dockerfile.unity-server`** → builds `sparc/unity-server:latest` — the NVIDIA OpenGL Unity Linux Server rendering image from `Dockerfile.unity-server`.
5. **`podman build -f Dockerfile.signaling`** → builds `sparc/signaling-server:latest` — the Node.js WebRTC signaling server image from `Dockerfile.signaling`.
6. **`podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server`** — downloads the Riva speech server image from NVIDIA's container registry. This doesn't need a local Dockerfile — you pull it directly from the official source.

> **Prerequisite:** The Dockerfiles referenced here (`Dockerfile.mas`, `Dockerfile.unity-server`, `Dockerfile.signaling`) are generated by Notebook 2b. Run those cells first, then copy or transfer the files to this VM before running these build commands.
```bash
# Stage required build-context artifacts first:
mkdir -p artifacts/unity/LinuxServer artifacts/signaling
# Copy Unity Linux server build output into artifacts/unity/LinuxServer/
# Copy signaling package files (e.g., package.json, server.js) into artifacts/signaling/

podman build -f Dockerfile.mas -t sparc/llm-backend:latest .
podman build -f Dockerfile.unity-server -t sparc/unity-server:latest .
podman build -f Dockerfile.signaling -t sparc/signaling-server:latest .
podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server
```

---

## 5.0 Podman + Systemd Orchestration


The `EXECUTE` flag and `run()` helper function used throughout this deployment follow the same dry-run safety pattern as Notebook 4.

When `EXECUTE = False` (the default), calling `run("some command")` just prints the command prefixed with `$` and notes it was not executed. Change to `EXECUTE = True` to run commands for real on the PubApps VM.

What makes this version different from Notebook 4's `run()`:
- This version uses `os.system()` instead of `subprocess.run()` — it runs commands in a shell without capturing stdout/stderr separately. This is simpler but means you see output directly in the notebook output area rather than in a captured result.
- There's no `check=True` parameter; failed commands don't automatically stop the notebook. Monitor output carefully when running with `EXECUTE = True` to catch any failures.

> **Before running any cell with `EXECUTE = True`:** Confirm you're SSH'd into the correct PubApps VM and that the paths printed in the configuration cell above are correct for your project.
### 5.1 Create Runtime Directories


All necessary directories on the PubApps VM are created here before any files are written or containers are configured. Two `mkdir -p` commands run — one for the model and data directories, one for the Quadlet configuration directory.

Directories created:
- **`/pubapps/SPARCP/`** — the project root for all SPARC-P files on this VM
- **`/pubapps/SPARCP/models/`** — where the transferred fine-tuned LLM adapter files from HiPerGator are stored
- **`/pubapps/SPARCP/riva_models/`** — where NVIDIA Riva's pre-initialized ASR and TTS model files are stored (populated separately via Riva's `riva_init.sh`)
- **`/pubapps/SPARCP/logs/`** — application log output directory
- **`~/.config/containers/systemd/`** — the per-user Podman Quadlet directory where systemd looks for container service definitions

Like in Notebook 4, these commands go through the `run()` helper — they only execute when `EXECUTE = True`. In dry-run mode, the `mkdir` commands are just printed.
```bash
mkdir -p $SPARC_PUBAPPS_ROOT/{models,riva_models,logs}
mkdir -p ~/.config/containers/systemd
```

### 5.2 Define `avatar.pod`
`avatar.pod` is the Podman Pod definition that creates the shared network namespace for all SPARC-P Pixel Streaming services. This is the first of five Quadlet files written in sequence.

What a Podman Pod is: A Pod is a group of containers that share the same network namspace, meaning they can all reach each other using `localhost`. This is the same concept as a Kubernetes Pod. Instead of configuring inter-container networking (which requires custom bridge networks and container hostnames), every service in the pod just talks to `localhost:PORT`.

The port mappings in this pod definition expose specific ports from inside the pod to the outside world (the PubApps VM's network):
- **`8000:8000`** — FastAPI backend API (used by the Unity client to call `/v1/chat`)
- **`8080:8080`** — WebRTC signaling server (used by browsers to establish the video stream connection)
- **`3478:3478/udp`** — STUN server port for WebRTC NAT traversal (helps browsers behind firewalls connect)
- **`49152-49200:49152-49200/udp` and `/tcp`** — The WebRTC media UDP port range. WebRTC uses these ephemeral ports to transmit the actual video stream data between Unity and each connected browser. The range must be large enough for concurrent users (one port pair per active stream).

The pod file is written to `QUADLET_DIR` (`~/.config/containers/systemd/avatar.pod`). When `systemctl --user daemon-reload` runs, systemd discovers this file and registers `avatar-pod` as a manageable service.
 (Ports + GPU-aware services)

```ini
# ~/.config/containers/systemd/avatar.pod
[Unit]
Description=SPARC-P Avatar Pod (Pixel Streaming)
After=network-online.target

[Pod]
PodName=sparc-avatar
PublishPort=8000:8000
PublishPort=8080:8080
PublishPort=3478:3478/udp
PublishPort=49152-49200:49152-49200/udp
PublishPort=49152-49200:49152-49200/tcp

[Install]
WantedBy=default.target
```

- `8000`: FastAPI backend
- `8080`: Signaling HTTP/WebSocket
- `3478/udp`: STUN baseline (if used)
- `49152-49200`: WebRTC media candidate range (adjust per RC network policy)

### 5.3 Riva Container Service


`riva-server.container` is the Podman Quadlet configuration that runs the NVIDIA Riva speech server (ASR + TTS) as part of the Pixel Streaming pod.

Key differences from the standard Notebook 4 Riva service:
- **`Pod=sparc-avatar.pod`**: This is the critical integration point. Instead of running as a standalone container, Riva joins the `sparc-avatar` pod. This means Riva and all other services (backend, Unity, signaling) share a single network namespace — they communicate using `localhost` rather than container hostnames.
- **`After=avatar-pod.service` + `Requires=avatar-pod.service`**: The Riva container only starts after the pod itself is running. The pod must be created first for `Pod=` to work.
- **`Device=nvidia.com/gpu=all`**: Grants Riva access to the L4 GPU for running its ASR and TTS neural network models. In the Pixel Streaming configuration, the GPU is shared between Riva (~3 GB VRAM), the backend LLM (~13 GB VRAM), and Unity's render server (~3.5 GB VRAM).
- **`Volume={RIVA_MODEL_DIR}:/data:Z`**: Mounts the host's riva_models directory inside the container where Riva expects to find its pre-initialized model files.
- **`TimeoutStartSec=300`**: Allows up to 5 minutes for Riva to start — loading ASR and TTS models into VRAM takes time, especially when competing with the LLM for GPU memory at startup.
```ini
# ~/.config/containers/systemd/riva-server.container
[Unit]
Description=SPARC-P Riva Speech Server
After=avatar-pod.service
Requires=avatar-pod.service

[Container]
Pod=sparc-avatar.pod
Image=nvcr.io/nvidia/riva/riva-speech:2.16.0-server
ContainerName=riva-server
Volume=${SPARC_PUBAPPS_ROOT}/riva_models:/data:Z
PublishPort=50051:50051
Device=nvidia.com/gpu=all
Environment=NVIDIA_VISIBLE_DEVICES=all
Exec=/opt/riva/bin/riva_server --riva_model_repo=/data/models

[Service]
Restart=always
TimeoutStartSec=300

[Install]
WantedBy=default.target
```

### 5.4 Backend/LLM Container Service


`sparc-backend.container` is the Quadlet configuration for the FastAPI + vLLM AI backend service inside the Pixel Streaming pod. This is the container that hosts the SPARC-P language models and handles all `/v1/chat` API calls.

What makes this container's configuration notable:
- **GPU sharing**: `Device=nvidia.com/gpu=all` gives the backend container access to the full L4 GPU. In practice, PyTorch and the quantized LLM will use ~13 GB of the 24 GB VRAM, leaving the remainder for Riva and Unity (all sharing the same physical GPU via the pod).
- **`MODEL_ID=gpt-oss-20b` and `QUANTIZATION=4bit`**: These environment variables tell the backend which model to load and that it should use 4-bit quantization (NF4 format). 4-bit quantization reduces the model's VRAM footprint by ~75%, making it feasible to run a 20B-parameter model on a 24 GB GPU alongside other services.
- **`RIVA_SERVER=localhost:50051`**: Because the backend and Riva share the pod's localhost, the backend connects to Riva at this address — no container-to-container networking complexity needed.
- **`SPARC_CORS_ALLOW_CREDENTIALS=false`** + `SPARC_CORS_ALLOWED_ORIGINS` set to the two official domains: Enforces secure CORS policy to prevent unauthorized browser origins from making API calls.
- **`After=riva-server.service`** + **`Requires=riva-server.service`**: The backend only starts after Riva is ready, so it can connect during its startup sequence without retrying.
- **`Exec=uvicorn main:app --workers 1`**: Single-worker mode for the 2-core PubApps VM constraint.
```ini
# ~/.config/containers/systemd/sparc-backend.container
[Unit]
Description=SPARC-P FastAPI + vLLM Backend
After=avatar-pod.service riva-server.service
Requires=avatar-pod.service riva-server.service

[Container]
Pod=sparc-avatar.pod
Image=sparc/llm-backend:latest
ContainerName=sparc-backend
Volume=${SPARC_PUBAPPS_ROOT}/models:${SPARC_PUBAPPS_ROOT}/models:Z
Environment=MODEL_ID=gpt-oss-20b
Environment=QUANTIZATION=4bit
Environment=RIVA_SERVER=localhost:50051
Environment=SPARC_CORS_ALLOWED_ORIGINS=${SPARC_CORS_ALLOWED_ORIGINS}
Environment=SPARC_CORS_ALLOW_CREDENTIALS=false
Device=nvidia.com/gpu=all
Exec=uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

[Service]
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

### 5.5 Signaling Server Container Service


`signaling-server.container` is the Quadlet configuration for the WebRTC signaling server, the "matchmaker" service that helps browsers establish a direct video stream connection to the Unity renderer.

How WebRTC signaling works (plain-English): Before a browser can receive the Unity avatar's live video stream, the browser and the Unity server need to exchange connection details (their respective IP addresses, network capabilities, and encryption keys). The signaling server is a lightweight intermediary that passes these details back and forth — once the connection is established, the signaling server is no longer needed and the video streams directly between Unity and the browser.

Key configuration details:
- **`Pod=sparc-avatar.pod`**: Joins the shared pod so the signaling server can communicate with Unity's render server via `localhost:8080` — no external networking needed.
- **`HTTP_PORT=8080`** and **`Exec=node server.js --httpPort 8080`**: The signaling server listens on port 8080. The pod definition (written in a previous cell) maps this port to the host, so browsers can reach it.
- **No GPU needed**: This container doesn't need `Device=nvidia.com/gpu=all` — it's pure Node.js and runs on CPU.
- **`Restart=always` with `RestartSec=5`**: If the signaling server crashes (which can happen if too many clients connect simultaneously), systemd restarts it within 5 seconds. Active video streams aren't affected by a signaling server restart — only new connection attempts would briefly fail.
- This is the `sparc/signaling-server:latest` image built from the `Dockerfile.signaling` created in Notebook 2b.
```ini
# ~/.config/containers/systemd/signaling-server.container
[Unit]
Description=SPARC-P Unity Render Streaming Signaling Server
After=avatar-pod.service
Requires=avatar-pod.service

[Container]
Pod=sparc-avatar.pod
Image=sparc/signaling-server:latest
ContainerName=signaling-server
Environment=HTTP_PORT=8080
Environment=PUBLIC_HOST=sparc-p.rc.ufl.edu
Exec=node server.js --httpPort 8080

[Service]
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

### 5.6 Unity Server Container Service
`unity-server.container` is the Quadlet configuration for the Unity Linux Server, which does the actual 3D rendering of the SPARC-P digital human avatar on the GPU and streams that video to users’ browsers over WebRTC.

This is the most complex service in the Pixel Streaming architecture. Key points:

**Dependencies (last to start):** `After=` and `Requires=` list all four other services — the pod, signaling server, backend, and Riva. Unity must start last because it needs to connect to all of them at launch. If any required service is missing, systemd won't start this container.

**GPU access (`Device=nvidia.com/gpu=all`):** Unlike the standard deployment where Unity runs as a WebGL app in the user's browser, here Unity runs server-side and uses the L4 GPU for rendering (~3.5 GB VRAM). The GPU renders the 3D scene frames and Unity's Render Streaming plugin encodes them as a video stream for delivery via WebRTC.

**Environment variables:** 
- `SIGNALING_URL=ws://localhost:8080` — connects Unity's Render Streaming plugin to the signaling server within the pod to coordinate WebRTC connection setup.
- `BACKEND_URL=http://localhost:8000` — the FastAPI backend address where Unity sends caregiver responses to trigger mouth animation, gestures, and other avatar behaviors.
- `RIVA_URL=localhost:50051` — connects Unity to Riva for audio playback (Unity receives TTS audio from the backend and may use Riva's audio pipeline directly for lip-sync timing).

**Launch flags** (`-batchmode -force-vulkan`): `-batchmode` runs Unity without a display window (headless server mode); `-force-vulkan` uses NVIDIA's Vulkan renderer which is required for GPU-accelerated rendering on Linux without a display.
 (GPU via CDI)

```ini
# ~/.config/containers/systemd/unity-server.container
[Unit]
Description=SPARC-P Unity Linux Server (Render Streaming)
After=avatar-pod.service signaling-server.service sparc-backend.service riva-server.service
Requires=avatar-pod.service signaling-server.service sparc-backend.service riva-server.service

[Container]
Pod=sparc-avatar.pod
Image=sparc/unity-server:latest
ContainerName=unity-server
Device=nvidia.com/gpu=all
Environment=SIGNALING_URL=ws://localhost:8080
Environment=BACKEND_URL=http://localhost:8000
Environment=RIVA_URL=localhost:50051
Exec=/app/SPARC-P.x86_64 -logFile /dev/stdout -batchmode -force-vulkan

[Service]
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

### 5.7 Enable Services


The final sequence of shell commands needed to activate all five SPARC-P Pixel Streaming services is printed below. Copy and run these in your PubApps VM terminal.

What each command does and why order matters:
1. **`systemctl --user daemon-reload`**: Tells systemd to re-read all Quadlet definitions written in the previous cells. Must run first, otherwise systemd doesn't know about any of the new services.
2. **`systemctl --user enable --now avatar-pod`**: Creates and starts the shared network pod (`sparc-avatar`) that all containers join. Must start before any container, because containers with `Pod=sparc-avatar.pod` can't start until the pod exists.
3. **`systemctl --user enable --now riva-server`**: Starts Riva (ASR + TTS). Takes ~2 minutes to load models into GPU.
4. **`systemctl --user enable --now sparc-backend`**: Starts the FastAPI backend (requires Riva to be up — systemd handles ordering via `After=`).
5. **`systemctl --user enable --now signaling-server`**: Starts the WebRTC signaling Node.js process.
6. **`systemctl --user enable --now unity-server`**: Starts the Unity renderer last — it depends on all other services being ready.
7. **`nvidia-smi`**: Runs the NVIDIA GPU monitor to confirm all three GPU-using services (Riva, backend, Unity) are visible in the process list and total VRAM usage is within the 24 GB L4 budget.

The `enable` flag (in `enable --now`) registers each service to auto-start on future logins — so the entire stack comes up automatically whenever the PubApps VM restarts.
```bash
systemctl --user daemon-reload
systemctl --user enable --now avatar-pod
systemctl --user enable --now riva-server
systemctl --user enable --now sparc-backend
systemctl --user enable --now signaling-server
systemctl --user enable --now unity-server
```

---

## 6.0 Configure NGINX Reverse Proxy

### 6.1 Ticket Request Template

```
Subject: NGINX Configuration for SPARC-P Pixel Streaming (PubApps)

Body:
Please configure reverse proxy for SPARC-P server-side rendering deployment:

1. SSL certificate for sparc-p.rc.ufl.edu
2. Proxy routes:
   - /api/ -> http://localhost:8000/
   - /signal/ -> http://localhost:8080/
3. WebSocket upgrade enabled for /signal/
4. UF Shibboleth SSO enabled for access control
5. Preserve headers for WebRTC signaling endpoints
```

### 6.2 Reference NGINX Config

```nginx
server {
    listen 443 ssl http2;
    server_name sparc-p.rc.ufl.edu;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /signal/ {
        proxy_pass http://localhost:8080/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 7.0 Deployment Checklist

### 7.1 Pre-Deployment
- [ ] PubApps instance provisioned
- [ ] Models transferred to `${SPARC_PUBAPPS_ROOT}/models/`
- [ ] Conda backend environment created
- [ ] Runtime images available (backend, unity-server, signaling, riva)
- [ ] UF RC risk assessment completed

### 7.2 Service Deployment
- [ ] `avatar-pod` active
- [ ] `riva-server` active
- [ ] `sparc-backend` active
- [ ] `signaling-server` active
- [ ] `unity-server` active
- [ ] All services configured for auto-start

### 7.3 Validation
- [ ] `https://sparc-p.rc.ufl.edu/api/health` responds
- [ ] `/signal/` endpoint negotiates WebRTC sessions
- [ ] Browser receives Unity-rendered stream
- [ ] Unity can call backend and Riva on localhost
- [ ] `nvidia-smi` confirms runtime fits 24GB budget

---

## 8.0 Monitoring and Maintenance

### 8.1 Service Commands

```bash
systemctl --user status avatar-pod riva-server sparc-backend signaling-server unity-server
journalctl --user -u unity-server -n 100 -f
journalctl --user -u signaling-server -n 100 -f
journalctl --user -u sparc-backend -n 100 -f
journalctl --user -u riva-server -n 100 -f
```

### 8.2 Resource Monitoring

```bash
nvidia-smi
free -h
df -h $SPARC_PUBAPPS_ROOT
```

### 8.3 VRAM Guardrail Checks

```bash
# Ensure aggregate stays within single L4 budget
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 9.0 Security and Compliance

1. **Authentication**: UF Shibboleth SSO
2. **Transport Security**: TLS 1.2+
3. **Data Handling**: Session state managed through approved services, no PHI persistence in container logs
4. **Access Control**: SSH key-based project access for operators
5. **Compliance**: Follow [UFIT RC PubApps Policy](https://docs.rc.ufl.edu/services/web_hosting/)

---

## 10.0 Troubleshooting

### 10.1 Common Issues

**Issue**: Unity stream connects but no video
```bash
journalctl --user -u unity-server -n 100
journalctl --user -u signaling-server -n 100
# Verify signaling URL and WebRTC ports are open per RC policy
```

**Issue**: GPU memory exceeded
```bash
nvidia-smi
# Reduce Unity quality settings / frame rate
# Lower LLM KV cache or max context configuration
```

**Issue**: Browser cannot negotiate WebRTC
```bash
curl -i https://sparc-p.rc.ufl.edu/signal/
# Verify NGINX WebSocket upgrade config and signaling server health
```

**Issue**: Backend unavailable from Unity container
```bash
curl http://localhost:8000/health
# Check sparc-backend service logs and model load status
```

---

## 11.0 Summary

This deployment variant replaces client-side WebGL with server-side rendering optimized for thin clients:

1. Unity runs on PubApps GPU in a Linux server container.
2. Browser sessions connect through a Node signaling service for WebRTC negotiation.
3. FastAPI + LLM + Riva remain local services for dialog, inference, and speech.
4. Audio2Face is removed to maintain single-L4 VRAM stability.
5. Runtime is orchestrated with Podman + systemd user services on PubApps.

Next steps:
1. Build and publish the Unity Linux server and signaling images.
2. Enable services in order and validate WebRTC signaling and stream quality.
3. Tune Unity and model runtime settings to keep total VRAM within the 24GB budget.
