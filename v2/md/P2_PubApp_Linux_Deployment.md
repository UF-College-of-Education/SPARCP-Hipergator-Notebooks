# P2_PubApp_Linux_Deployment

> Auto-generated markdown counterpart from notebook cells.

# SPARC-P PubApps Deployment (Pixel Streaming)

This notebook deploys SPARC-P using server-side Unity rendering for thin clients on a single L4 GPU (24GB).

## Architectural Changes
- Remove Audio2Face
- Run Unity Linux Server container with Render Streaming
- Add Node signaling container for WebRTC negotiation
- Keep FastAPI + LLM + Riva as local services

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

```python
import os
import textwrap
from pathlib import Path

PROJECT = os.environ.get("SPARC_PUBAPPS_PROJECT", "SPARCP")
PUBAPPS_ROOT = Path(f"/pubapps/{PROJECT}")
MODEL_DIR = PUBAPPS_ROOT / "models"
RIVA_MODEL_DIR = PUBAPPS_ROOT / "riva_models"
QUADLET_DIR = Path.home() / ".config/containers/systemd"
PUBAPP_ALLOWED_ORIGINS = os.environ.get(
    "SPARC_CORS_ALLOWED_ORIGINS",
    "https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org",
)

VRAM_PLAN = {
    "llm_vllm_4bit": "~13GB",
    "unity_render": "~3.5GB",
    "riva_embedded": "~3GB",
    "buffer": "~2-3GB"
}

print(f"Project: {PROJECT}")
print(f"Root: {PUBAPPS_ROOT}")
print(f"Allowed CORS origins: {PUBAPP_ALLOWED_ORIGINS}")
print(f"VRAM plan: {VRAM_PLAN}")
```

The `EXECUTE` flag and `run()` helper function used throughout this deployment follow the same dry-run safety pattern as Notebook 4.

When `EXECUTE = False` (the default), calling `run("some command")` just prints the command prefixed with `$` and notes it was not executed. Change to `EXECUTE = True` to run commands for real on the PubApps VM.

What makes this version different from Notebook 4's `run()`:
- This version uses `os.system()` instead of `subprocess.run()` — it runs commands in a shell without capturing stdout/stderr separately. This is simpler but means you see output directly in the notebook output area rather than in a captured result.
- There's no `check=True` parameter; failed commands don't automatically stop the notebook. Monitor output carefully when running with `EXECUTE = True` to catch any failures.

> **Before running any cell with `EXECUTE = True`:** Confirm you're SSH'd into the correct PubApps VM and that the paths printed in the configuration cell above are correct for your project.

```python
EXECUTE = False

def run(cmd: str):
    print(f"$ {cmd}")
    if not EXECUTE:
        print("(dry-run) command not executed\n")
        return
    os.system(cmd)
```

## 1) Prepare directories and base resources

All necessary directories on the PubApps VM are created here before any files are written or containers are configured. Two `mkdir -p` commands run — one for the model and data directories, one for the Quadlet configuration directory.

Directories created:
- **`/pubapps/SPARCP/`** — the project root for all SPARC-P files on this VM
- **`/pubapps/SPARCP/models/`** — where the transferred fine-tuned LLM adapter files from HiPerGator are stored
- **`/pubapps/SPARCP/riva_models/`** — where NVIDIA Riva's pre-initialized ASR and TTS model files are stored (populated separately via Riva's `riva_init.sh`)
- **`/pubapps/SPARCP/logs/`** — application log output directory
- **`~/.config/containers/systemd/`** — the per-user Podman Quadlet directory where systemd looks for container service definitions

Like in Notebook 4, these commands go through the `run()` helper — they only execute when `EXECUTE = True`. In dry-run mode, the `mkdir` commands are just printed.

```python
run(f"mkdir -p {PUBAPPS_ROOT} {MODEL_DIR} {RIVA_MODEL_DIR} {PUBAPPS_ROOT / 'logs'}")
run(f"mkdir -p {QUADLET_DIR}")
```

## 2) Build/pull required images

The complete sequence of commands needed to build and pull all four container images required by the Pixel Streaming deployment is printed below. These are reference commands — copy and run them in your terminal on the PubApps VM (or wherever you're building Docker images).

The commands in order:
1. **`mkdir -p artifacts/unity/LinuxServer artifacts/signaling`** — creates the directories where you place your Unity Linux Server build files and signaling server package files before building images.
2. **Two comment lines** — reminders to manually copy your Unity build output and Node.js signaling files into these directories before the build steps.
3. **`podman build -f Dockerfile.mas`** → builds `sparc/llm-backend:latest` — the Python FastAPI + LLM backend image from `Dockerfile.mas`.
4. **`podman build -f Dockerfile.unity-server`** → builds `sparc/unity-server:latest` — the NVIDIA OpenGL Unity Linux Server rendering image from `Dockerfile.unity-server`.
5. **`podman build -f Dockerfile.signaling`** → builds `sparc/signaling-server:latest` — the Node.js WebRTC signaling server image from `Dockerfile.signaling`.
6. **`podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server`** — downloads the Riva speech server image from NVIDIA's container registry. This doesn't need a local Dockerfile — you pull it directly from the official source.

> **Prerequisite:** The Dockerfiles referenced here (`Dockerfile.mas`, `Dockerfile.unity-server`, `Dockerfile.signaling`) are generated by Notebook 2b. Run those cells first, then copy or transfer the files to this VM before running these build commands.

```python
build_cmds = textwrap.dedent("""
mkdir -p artifacts/unity/LinuxServer artifacts/signaling
# Copy Unity Linux server build output into artifacts/unity/LinuxServer/
# Copy signaling package files (e.g., package.json, server.js) into artifacts/signaling/
podman build -f Dockerfile.mas -t sparc/llm-backend:latest .
podman build -f Dockerfile.unity-server -t sparc/unity-server:latest .
podman build -f Dockerfile.signaling -t sparc/signaling-server:latest .
podman pull nvcr.io/nvidia/riva/riva-speech:2.16.0-server
""").strip()
print(build_cmds)
```

## 3) Write Podman Quadlet units

`avatar.pod` is the Podman Pod definition that creates the shared network namespace for all SPARC-P Pixel Streaming services. This is the first of five Quadlet files written in sequence.

What a Podman Pod is: A Pod is a group of containers that share the same network namspace, meaning they can all reach each other using `localhost`. This is the same concept as a Kubernetes Pod. Instead of configuring inter-container networking (which requires custom bridge networks and container hostnames), every service in the pod just talks to `localhost:PORT`.

The port mappings in this pod definition expose specific ports from inside the pod to the outside world (the PubApps VM's network):
- **`8000:8000`** — FastAPI backend API (used by the Unity client to call `/v1/chat`)
- **`8080:8080`** — WebRTC signaling server (used by browsers to establish the video stream connection)
- **`3478:3478/udp`** — STUN server port for WebRTC NAT traversal (helps browsers behind firewalls connect)
- **`49152-49200:49152-49200/udp` and `/tcp`** — The WebRTC media UDP port range. WebRTC uses these ephemeral ports to transmit the actual video stream data between Unity and each connected browser. The range must be large enough for concurrent users (one port pair per active stream).

The pod file is written to `QUADLET_DIR` (`~/.config/containers/systemd/avatar.pod`). When `systemctl --user daemon-reload` runs, systemd discovers this file and registers `avatar-pod` as a manageable service.

```python
pod_content = textwrap.dedent("""
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
""").strip()
(QUADLET_DIR / "avatar.pod").write_text(pod_content)
print("Wrote avatar.pod")
```

`riva-server.container` is the Podman Quadlet configuration that runs the NVIDIA Riva speech server (ASR + TTS) as part of the Pixel Streaming pod.

Key differences from the standard Notebook 4 Riva service:
- **`Pod=sparc-avatar.pod`**: This is the critical integration point. Instead of running as a standalone container, Riva joins the `sparc-avatar` pod. This means Riva and all other services (backend, Unity, signaling) share a single network namespace — they communicate using `localhost` rather than container hostnames.
- **`After=avatar-pod.service` + `Requires=avatar-pod.service`**: The Riva container only starts after the pod itself is running. The pod must be created first for `Pod=` to work.
- **`Device=nvidia.com/gpu=all`**: Grants Riva access to the L4 GPU for running its ASR and TTS neural network models. In the Pixel Streaming configuration, the GPU is shared between Riva (~3 GB VRAM), the backend LLM (~13 GB VRAM), and Unity's render server (~3.5 GB VRAM).
- **`Volume={RIVA_MODEL_DIR}:/data:Z`**: Mounts the host's riva_models directory inside the container where Riva expects to find its pre-initialized model files.
- **`TimeoutStartSec=300`**: Allows up to 5 minutes for Riva to start — loading ASR and TTS models into VRAM takes time, especially when competing with the LLM for GPU memory at startup.

```python
riva_container = textwrap.dedent(f"""
[Unit]
Description=SPARC-P Riva Speech Server
After=avatar-pod.service
Requires=avatar-pod.service

[Container]
Pod=sparc-avatar.pod
Image=nvcr.io/nvidia/riva/riva-speech:2.16.0-server
ContainerName=riva-server
Volume={RIVA_MODEL_DIR}:/data:Z
Device=nvidia.com/gpu=all
Environment=NVIDIA_VISIBLE_DEVICES=all
Exec=/opt/riva/bin/riva_server --riva_model_repo=/data/models

[Service]
Restart=always
TimeoutStartSec=300

[Install]
WantedBy=default.target
""").strip()
(QUADLET_DIR / "riva-server.container").write_text(riva_container)
print("Wrote riva-server.container")
```

`sparc-backend.container` is the Quadlet configuration for the FastAPI + vLLM AI backend service inside the Pixel Streaming pod. This is the container that hosts the SPARC-P language models and handles all `/v1/chat` API calls.

What makes this container's configuration notable:
- **GPU sharing**: `Device=nvidia.com/gpu=all` gives the backend container access to the full L4 GPU. In practice, PyTorch and the quantized LLM will use ~13 GB of the 24 GB VRAM, leaving the remainder for Riva and Unity (all sharing the same physical GPU via the pod).
- **`MODEL_ID=gpt-oss-20b` and `QUANTIZATION=4bit`**: These environment variables tell the backend which model to load and that it should use 4-bit quantization (NF4 format). 4-bit quantization reduces the model's VRAM footprint by ~75%, making it feasible to run a 20B-parameter model on a 24 GB GPU alongside other services.
- **`RIVA_SERVER=localhost:50051`**: Because the backend and Riva share the pod's localhost, the backend connects to Riva at this address — no container-to-container networking complexity needed.
- **`SPARC_CORS_ALLOW_CREDENTIALS=false`** + `SPARC_CORS_ALLOWED_ORIGINS` set to the two official domains: Enforces secure CORS policy to prevent unauthorized browser origins from making API calls.
- **`After=riva-server.service`** + **`Requires=riva-server.service`**: The backend only starts after Riva is ready, so it can connect during its startup sequence without retrying.
- **`Exec=uvicorn main:app --workers 1`**: Single-worker mode for the 2-core PubApps VM constraint.

```python
backend_container = textwrap.dedent(f"""
[Unit]
Description=SPARC-P FastAPI + vLLM Backend
After=avatar-pod.service riva-server.service
Requires=avatar-pod.service riva-server.service

[Container]
Pod=sparc-avatar.pod
Image=sparc/llm-backend:latest
ContainerName=sparc-backend
Volume={MODEL_DIR}:{MODEL_DIR}:Z
Environment=MODEL_ID=gpt-oss-20b
Environment=QUANTIZATION=4bit
Environment=RIVA_SERVER=localhost:50051
Environment=SPARC_MODEL_BASE_PATH={MODEL_DIR}
Environment=SPARC_FIREBASE_CREDS={PUBAPPS_ROOT / 'config' / 'firebase-credentials.json'}
Environment=SPARC_CORS_ALLOWED_ORIGINS={PUBAPP_ALLOWED_ORIGINS}
Environment=SPARC_CORS_ALLOW_CREDENTIALS=false
Device=nvidia.com/gpu=all
Exec=uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

[Service]
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
""").strip()
(QUADLET_DIR / "sparc-backend.container").write_text(backend_container)
print("Wrote sparc-backend.container")
```

`signaling-server.container` is the Quadlet configuration for the WebRTC signaling server, the "matchmaker" service that helps browsers establish a direct video stream connection to the Unity renderer.

How WebRTC signaling works (plain-English): Before a browser can receive the Unity avatar's live video stream, the browser and the Unity server need to exchange connection details (their respective IP addresses, network capabilities, and encryption keys). The signaling server is a lightweight intermediary that passes these details back and forth — once the connection is established, the signaling server is no longer needed and the video streams directly between Unity and the browser.

Key configuration details:
- **`Pod=sparc-avatar.pod`**: Joins the shared pod so the signaling server can communicate with Unity's render server via `localhost:8080` — no external networking needed.
- **`HTTP_PORT=8080`** and **`Exec=node server.js --httpPort 8080`**: The signaling server listens on port 8080. The pod definition (written in a previous cell) maps this port to the host, so browsers can reach it.
- **No GPU needed**: This container doesn't need `Device=nvidia.com/gpu=all` — it's pure Node.js and runs on CPU.
- **`Restart=always` with `RestartSec=5`**: If the signaling server crashes (which can happen if too many clients connect simultaneously), systemd restarts it within 5 seconds. Active video streams aren't affected by a signaling server restart — only new connection attempts would briefly fail.
- This is the `sparc/signaling-server:latest` image built from the `Dockerfile.signaling` created in Notebook 2b.

```python
signaling_container = textwrap.dedent("""
[Unit]
Description=SPARC-P Unity Render Streaming Signaling Server
After=avatar-pod.service
Requires=avatar-pod.service

[Container]
Pod=sparc-avatar.pod
Image=sparc/signaling-server:latest
ContainerName=signaling-server
Environment=HTTP_PORT=8080
Exec=node server.js --httpPort 8080

[Service]
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
""").strip()
(QUADLET_DIR / "signaling-server.container").write_text(signaling_container)
print("Wrote signaling-server.container")
```

`unity-server.container` is the Quadlet configuration for the Unity Linux Server, which does the actual 3D rendering of the SPARC-P digital human avatar on the GPU and streams that video to users’ browsers over WebRTC.

This is the most complex service in the Pixel Streaming architecture. Key points:

**Dependencies (last to start):** `After=` and `Requires=` list all four other services — the pod, signaling server, backend, and Riva. Unity must start last because it needs to connect to all of them at launch. If any required service is missing, systemd won't start this container.

**GPU access (`Device=nvidia.com/gpu=all`):** Unlike the standard deployment where Unity runs as a WebGL app in the user's browser, here Unity runs server-side and uses the L4 GPU for rendering (~3.5 GB VRAM). The GPU renders the 3D scene frames and Unity's Render Streaming plugin encodes them as a video stream for delivery via WebRTC.

**Environment variables:** 
- `SIGNALING_URL=ws://localhost:8080` — connects Unity's Render Streaming plugin to the signaling server within the pod to coordinate WebRTC connection setup.
- `BACKEND_URL=http://localhost:8000` — the FastAPI backend address where Unity sends caregiver responses to trigger mouth animation, gestures, and other avatar behaviors.
- `RIVA_URL=localhost:50051` — connects Unity to Riva for audio playback (Unity receives TTS audio from the backend and may use Riva's audio pipeline directly for lip-sync timing).

**Launch flags** (`-batchmode -force-vulkan`): `-batchmode` runs Unity without a display window (headless server mode); `-force-vulkan` uses NVIDIA's Vulkan renderer which is required for GPU-accelerated rendering on Linux without a display.

```python
unity_container = textwrap.dedent("""
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
""").strip()
(QUADLET_DIR / "unity-server.container").write_text(unity_container)
print("Wrote unity-server.container")
```

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

```python
enable_cmds = textwrap.dedent("""
systemctl --user daemon-reload
systemctl --user enable --now avatar-pod
systemctl --user enable --now riva-server
systemctl --user enable --now sparc-backend
systemctl --user enable --now signaling-server
systemctl --user enable --now unity-server
nvidia-smi
""").strip()
print(enable_cmds)
```
