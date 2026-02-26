# SPARC Containerization and Deployment

## 1.0 Introduction
This notebook covers packaging SPARC services as portable containers and preparing both local Podman validation and HiPerGator production deployment artifacts.

### 1.1 Objectives
1. **Containerize**: Build images for MAS backend, Unity Linux server runtime, and Signaling Server.
2. **Orchestrate**: Validate local Podman pod networking for server-side rendering (Pixel Streaming).
3. **Deploy**: Generate production SLURM artifacts for HiPerGator-compatible backend workflows.

### 1.2 Introduction Diagram
![Introduction](./images/notebook_2_-_section_1.png)

Introduction: This section defines the containerization objectives for the SPARC stack. We now support server-side rendering for thin clients by introducing a Unity Linux server runtime and a signaling container in addition to the backend services.

---

## 2.0 Containerization (Docker/Podman -> Apptainer)
We develop with Docker/Podman and deploy with Apptainer on HPC when needed.

### 2.0 Container Build Strategy Diagram
![Container Build Strategy](./images/notebook_2_-_section_2.png)

Container Build Strategy: The flow uses secure, minimal runtime images. Build steps compile dependencies in dedicated stages, then copy only required artifacts into runtime images.

### 2.1 Dockerfile Definition

This section provides image definitions for three build targets:
1. **MAS Backend** (`Dockerfile.mas`)
2. **Unity Linux Server Build** (`Dockerfile.unity-server`)
3. **WebRTC Signaling Server** (`Dockerfile.signaling`)

Canonical build-context artifacts used by these Dockerfiles:
- `requirements.txt` (MAS Python dependencies)
- `artifacts/unity/LinuxServer/` (Unity Linux server build output)
- `artifacts/signaling/` (Render Streaming signaling server source)

### 2.2 Dockerfile for Multi-Agent System (MAS)

**Note on Conda vs Containers**: On HiPerGator and PubApps you can deploy with conda environments, but containers are useful for portability and repeatability.

Two files are produced — `requirements.txt` (the Python dependency list for the MAS backend) and `Dockerfile.mas` (the container recipe) — and a validation check confirms that the Unity and signaling build artifacts are present before continuing.

- `create_requirements_file()` writes all required Python libraries (AI models, speech, PII redaction, vector search, etc.) to a plain text file that Docker will use during the image build.
- `create_mas_dockerfile()` writes a two-stage Dockerfile: the "builder" stage installs all heavy dependencies; the "runtime" stage copies only the final packages into a small, clean image that deploys faster and has a smaller attack surface.
- `validate_container_artifacts()` checks that the `artifacts/unity/LinuxServer/` and `artifacts/signaling/` directories exist before proceeding. If either is missing, it raises an error so you know exactly what's needed before attempting a build.
- The `Dockerfile.mas` exposes port `8000` and launches the backend with `uvicorn`, the high-performance async web server used in production.

```python
# 2.2 Dockerfile for Multi-Agent System (MAS)
from pathlib import Path

UNITY_BUILD_DIR = Path("artifacts/unity/LinuxServer")
SIGNALING_DIR = Path("artifacts/signaling")

def create_requirements_file():
    requirements = """
fastapi
uvicorn[standard]
pydantic>=2.5.0
numpy>=1.24.0
aiofiles
websockets
python-multipart
transformers>=4.36.0
accelerate>=0.25.0
tokenizers>=0.15.0
bitsandbytes>=0.41.0
peft>=0.7.0
langchain>=0.1.0
langchain-community>=0.0.13
langchain-openai>=0.0.5
langchain-chroma>=0.1.0
langgraph>=0.0.26
nvidia-riva-client>=2.14.0
nemoguardrails>=0.5.0
chromadb>=0.4.22
presidio-analyzer>=2.2.33
presidio-anonymizer>=2.2.33
firebase-admin>=6.2.0
python-jose[cryptography]
python-dotenv
grpcio
grpcio-tools
""".strip()
    Path("requirements.txt").write_text(requirements + "\n", encoding="utf-8")
    print("Created requirements.txt")

def validate_container_artifacts():
    missing = []
    if not Path("requirements.txt").exists():
        missing.append("requirements.txt")
    if not UNITY_BUILD_DIR.exists():
        missing.append(str(UNITY_BUILD_DIR))
    if not SIGNALING_DIR.exists():
        missing.append(str(SIGNALING_DIR))
    if missing:
        raise FileNotFoundError(f"Missing build artifacts: {missing}")

def create_mas_dockerfile():
    dockerfile_content = """
# --- Build Stage ---
FROM python:3.11-slim as builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# --- Runtime Stage ---
FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    with open("Dockerfile.mas", "w", encoding="utf-8") as f:
        f.write(dockerfile_content.strip())
    print("Created Dockerfile.mas")

create_requirements_file()
create_mas_dockerfile()
```

### 2.3 Dockerfile for Unity Linux Server (Render Streaming)

This image packages a Unity Linux player build for server-side rendering.

`Dockerfile.unity-server` is a container recipe that packages the compiled Unity Linux Server application so it can run headlessly on a GPU-equipped server and stream its rendered output over WebRTC to users’ browsers.

Key choices in this Dockerfile:
- **Base image `nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04`:** Unity's Linux player needs OpenGL libraries to render graphics, and it needs access to the NVIDIA GPU. This base image provides both — without it, Unity would crash immediately on a headless server.
- **System dependencies:** Libraries like `libasound2` (audio), `libglu1-mesa` (3D graphics utilities), and `libxi6` (input) are required by the Unity runtime even in server mode.
- **`COPY artifacts/unity/LinuxServer/ /app/`:** Copies your compiled Unity build (produced by Unity Editor → Build Settings → Linux → Server Build) into the image. You must place these files in `artifacts/unity/LinuxServer/` before building.
- **`-batchmode -force-vulkan`:** Runs Unity without a display window (headless) and forces the Vulkan graphics API, which is required for GPU-accelerated server-side rendering on Linux.
- Port `8080` is where Unity's Render Streaming plugin will serve WebRTC signaling traffic.

```python
# 2.3 Dockerfile for Unity Linux Server Build
def create_unity_server_dockerfile():
    dockerfile_content = """
FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu20.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libnss3 \\
    libxss1 \\
    libasound2 \\
    libglu1-mesa \\
    libxi6 \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Copy Linux server build output from canonical artifact directory
COPY artifacts/unity/LinuxServer/ /app/

RUN chmod +x /app/SPARC-P.x86_64

# Typical Render Streaming launch flags.
# Use -batchmode/-nographics for headless workflows, or -force-vulkan when package requires GPU rendering context.
CMD ["/app/SPARC-P.x86_64", "-logFile", "/dev/stdout", "-batchmode", "-force-vulkan"]
    """
    with open("Dockerfile.unity-server", "w") as f:
        f.write(dockerfile_content.strip())
    print("Created Dockerfile.unity-server")

create_unity_server_dockerfile()
```

### 2.4 Dockerfile for WebRTC Signaling Server (Node.js)

Use the signaling server from the Unity Render Streaming package.

`Dockerfile.signaling` is a container recipe for a lightweight Node.js server that acts as the matchmaker between users’ browsers and the Unity GPU renderer, enabling the WebRTC peer connection that carries the live video stream.

How it works:
- **Base image `node:20-alpine`:** Alpine Linux is a minimal OS (~5 MB), keeping this container very small since a signaling server doesn't need a GPU or heavy libraries — just a Node.js runtime.
- **`COPY artifacts/signaling/ /app/`:** Copies the signaling server source files (typically from the Unity Render Streaming package — `package.json`, `server.js`, etc.) into the container. Place these in `artifacts/signaling/` before building.
- **`npm ci --omit=dev`:** Installs only production dependencies (no test tools or dev utilities), making the image as small as possible.
- **Ports 8080 and 8888:** Port 8080 is the main HTTP/WebSocket endpoint where browsers connect to negotiate WebRTC. Port 8888 is an optional metrics or admin endpoint provided by some versions of the Unity signaling package.
- After writing the Dockerfile, `validate_container_artifacts()` runs immediately to confirm all three required artifact directories (`requirements.txt`, `artifacts/unity/LinuxServer/`, `artifacts/signaling/`) exist — halting with a clear error if anything is missing.

```python
# 2.4 Dockerfile for Signaling Server
def create_signaling_dockerfile():
    dockerfile_content = """
FROM node:20-alpine

WORKDIR /app

# Copy signaling source from canonical artifact directory
COPY artifacts/signaling/ /app/

RUN npm ci --omit=dev

EXPOSE 8080 8888

# 8080: HTTP/WebSocket signaling endpoint
# 8888: Optional metrics/admin endpoint if enabled by your signaling package
CMD ["node", "server.js", "--httpPort", "8080"]
    """
    with open("Dockerfile.signaling", "w") as f:
        f.write(dockerfile_content.strip())
    print("Created Dockerfile.signaling")

create_signaling_dockerfile()
validate_container_artifacts()
```

### 2.5 Build Commands (Reference)

Three `podman build` commands build the three container images from the Dockerfiles created above. Run these in your terminal (not as Python — they are shell commands) after the Dockerfiles and artifacts are in place.

- `podman build -f Dockerfile.mas` → builds the AI backend image (Python 3.11, all ML libraries)
- `podman build -f Dockerfile.unity-server` → builds the Unity renderer image (Ubuntu + NVIDIA OpenGL)  
- `podman build -f Dockerfile.signaling` → builds the signaling server image (Node.js Alpine)

Each image is tagged with the `sparc/` namespace and `:latest` so they can be referenced by the Podman pod commands in the next sections. Building all three can take **5–20 minutes** depending on your internet connection and CPU speed, as the ML libraries alone are several gigabytes.

> **Prerequisite:** Run these only after the `artifacts/unity/LinuxServer/` and `artifacts/signaling/` directories have been populated (see the staging step below).

```bash
podman build -f Dockerfile.mas -t sparc/mas-server:latest .
podman build -f Dockerfile.unity-server -t sparc/unity-server:latest .
podman build -f Dockerfile.signaling -t sparc/signaling-server:latest .
```

Artifact staging reference (run before `podman build`):
Two empty directories — `artifacts/unity/LinuxServer/` and `artifacts/signaling/` — are created here for the Dockerfiles to reference when building images. The `mkdir -p` flag means "make the full path and don't fail if it already exists."

After this step, you need to **manually copy files into these directories** before running `podman build`:
- Copy your **Unity Linux Server build output** (the `SPARC-P.x86_64` binary and its accompanying `_Data/` directory) into `artifacts/unity/LinuxServer/`
- Copy the **signaling server package files** (typically `package.json`, `package-lock.json`, and `server.js` from the Unity Render Streaming npm package) into `artifacts/signaling/`

The `validate_container_artifacts()` function (called at the end of the signaling Dockerfile cell) will check these directories exist before allowing the build to proceed.

```bash
mkdir -p artifacts/unity/LinuxServer artifacts/signaling
# Place Unity Linux server build output in artifacts/unity/LinuxServer/
# Place signaling package files (e.g., package.json, server.js) in artifacts/signaling/
```

---

## 3.0 Local Development with Podman
Podman pods allow local validation of the same localhost service mesh used in deployment.

### 3.0 Local Development Pod Diagram
![Local Development Pod](./images/notebook_2_-_section_3.png)

Local Development Pod (Podman): The local pod now includes MAS backend, Riva, Unity Linux server renderer, and signaling server. The browser receives a live video stream over WebRTC rather than loading a local Unity WebGL build. Audio2Face is removed from this architecture.

### 3.1 Podman Local Workflow

For local development, run all core services in one pod so they share localhost routing:
- Unity renderer -> localhost services (Riva + backend APIs)
- Signaling server -> browser negotiation path
- Browser -> receives WebRTC stream from Unity runtime

### 3.2 Podman Workflow (Reference Commands)
The complete set of Podman commands needed to launch all five SPARC-P components simultaneously is printed below, sharing a common network so they can communicate. Nothing is executed automatically — copy and paste these into your terminal.

What gets started and why:
1. **`podman pod create`** — creates a virtual network namespace called `sparc-avatar` where everything shares `localhost`. The port mappings expose the API (`8000`), signaling (`8080`), STUN/TURN (`3478` UDP), and a block of UDP ports (`49152–49200`) that WebRTC uses for actual video streaming.
2. **MAS server** — the AI orchestration backend (all three SPARC-P agents + FastAPI endpoints).
3. **Riva server** — NVIDIA's speech AI service that converts spoken audio to text (ASR) and text back to speech (TTS). The `--device nvidia.com/gpu=all` flag passes the local GPU through to the container.
4. **Unity server** — the 3D avatar renderer. It receives a `SIGNALING_URL` environment variable pointing to the signaling server so it knows where to connect for WebRTC negotiation. Also needs GPU access.
5. **Signaling server** — the WebRTC rendezvous point. `PUBLIC_HOST=localhost` tells it to advertise `localhost` as the ICE candidate address (appropriate for local testing; change to your public hostname for remote access).

> **Expected result:** After all containers start, open your browser to `http://localhost:8080` to view the live-rendered digital human avatar stream.

```python
# 3.2 Podman Workflow (Reference Commands)
podman_commands = """
# 1. Create Pod with signaling + API ports and WebRTC UDP range
podman pod create --name sparc-avatar \\
  -p 8000:8000 \\
  -p 8080:8080 \\
  -p 3478:3478/udp \\
  -p 49152-49200:49152-49200/udp

# 2. Run MAS Server
podman run -d --pod sparc-avatar --name mas-server sparc/mas-server:latest

# 3. Run Riva Server
podman run -d --pod sparc-avatar --name riva-server \\
  --device nvidia.com/gpu=all \\
  nvcr.io/nvidia/riva/riva-speech:2.16.0-server

# 4. Run Unity Linux Render Streaming Server
podman run -d --pod sparc-avatar --name unity-server \\
  --device nvidia.com/gpu=all \\
  -e SIGNALING_URL=ws://localhost:8080 \\
  sparc/unity-server:latest

# 5. Run Signaling Server
podman run -d --pod sparc-avatar --name signaling-server \\
  -e PUBLIC_HOST=localhost \\
  sparc/signaling-server:latest
"""
print(podman_commands)
```

---

## 4.0 Production Deployment on HiPerGator
Deploy persistent backend services using SLURM and Apptainer where required.

### 4.0 Production Deployment Diagram
![Production Deployment](./images/notebook_2_-_section_4.png)

Production Deployment (SLURM): This diagram represents the HiPerGator scheduling path for backend services. PubApps runtime orchestration is handled by Podman + systemd user services.

### 4.1 Building SIF Images

HiPerGator uses Apptainer, which requires Singularity Image Format (`.sif`) files.

### 4.2 Build SIF Images
A documentation placeholder prints a reminder that `.sif` (Singularity Image Format) files must be built before deploying to HiPerGator. The actual build commands are commented out because they need to run in a HiPerGator login node terminal with the Apptainer module loaded.

Why this step is necessary: HiPerGator's compute nodes cannot use Docker or Podman directly. They require Apptainer (formerly Singularity), which uses self-contained `.sif` image files. You build these once from your Docker images on a login node and store them in `/blue/` — then any compute node can run them without an internet connection.

> **To actually build the images:** SSH into a HiPerGator login node, run `module load apptainer`, uncomment the three `apptainer build` lines, and execute them. Each build can take 10–30 minutes depending on image size.

```python
# 4.2 Build SIF Images
# !module load apptainer
# !apptainer build mas_server.sif docker-daemon://sparc/mas-server:latest
print("Build SIF images from Docker sources before production deployment on HPG.")
```

### 4.3 Production Service Launch

This function generates a baseline `sparc_production.slurm` script for backend execution on HiPerGator.

### 4.4 Production SLURM Script Generator
`sparc_production.slurm` is the SLURM batch script for the Pixel Streaming variant of SPARC-P. This is the most resource-intensive configuration because it must simultaneously run the AI backend, Riva speech AI, and the Unity GPU renderer on a single node.

What makes this different from the standard deployment SLURM script:
- **4 GPUs, 32 CPU cores, 256 GB RAM** — Unity's server-side rendering requires substantial GPU memory alongside the language models. This resource request reflects the combined needs of all services.
- **MAS server only** — this script launches just the `mas_server.sif` container with `uvicorn`. The Riva and Unity servers are typically managed separately via Podman Quadlet units (see Notebooks 4 and 4b), as SLURM job termination would kill all co-located services.
- **14-day time limit** — extended from the 7-day standard to reduce how often the job needs to be resubmitted.
- **Background launch + `wait`** — starts the MAS server asynchronously with `&` and `wait` keeps the SLURM job alive until the process exits or the time limit is reached.

> **After running:** Transfer `sparc_production.slurm` to HiPerGator and deploy with `sbatch sparc_production.slurm`. Monitor with `squeue -u $USER`.

```python
# 4.4 Production SLURM Script Generator
def generate_production_script():
    script_content = """
#!/bin/bash
#SBATCH --job-name=sparc-production-service
#SBATCH --partition=hpg-ai
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb
#SBATCH --time=14-00:00:00
#SBATCH --output=sparc_service_%j.log

module purge
module load apptainer

MAS_SIF="/blue/jasondeanarnold/SPARCP/containers/mas_server.sif"

echo "Starting MAS..."
apptainer exec --nv ${MAS_SIF} uvicorn main:app --host 0.0.0.0 --port 8000 &

wait
    """
    with open("sparc_production.slurm", "w") as f:
        f.write(script_content.strip())
    print("Generated sparc_production.slurm")

generate_production_script()
```

---

## Summary

This notebook now covers both classic backend containerization and the new server-side rendering container artifacts:

1. **MAS Backend Container** for API/model orchestration.
2. **Unity Linux Server Container** for GPU-rendered server-side scene streaming.
3. **Node Signaling Container** for WebRTC negotiation.
4. **Podman Pod Topology** for localhost service routing and browser video streaming.
5. **HiPerGator Production Script** for backend workflows where Apptainer + SLURM are required.

The architecture removes Audio2Face from the deployment path and aligns local container artifacts with the PubApps Pixel Streaming runtime.
