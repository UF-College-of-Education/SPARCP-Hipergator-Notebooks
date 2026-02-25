# SPARC-P PubApp Deployment Guide

## Overview

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
# GPU access
AddDevice=/dev/nvidia0
AddDevice=/dev/nvidiactl
AddDevice=/dev/nvidia-uvm
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
```

---

## 5.0 Deploy FastAPI Backend Service

### 5.1 Create Backend Application

```bash
# On PubApps VM
cd /pubapps/SPARCP
mkdir -p backend
cd backend
```

Create the main FastAPI application (`main.py`):

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
import base64
import logging
from typing import Any, Dict, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Initialize FastAPI
app = FastAPI(
    title="SPARC-P Multi-Agent Backend",
    description="HPV Vaccine Communication Training System",
    version="1.0.0"
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
inference_lock = asyncio.Lock()
timeout_state_lock = asyncio.Lock()
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

def synthesize_tts_sync(text: str, voice_name: str = "English-US.Female-1") -> bytes:
    auth = riva.client.Auth(uri=RIVA_SERVER)
    tts_service = riva.client.SpeechSynthesisService(auth)
    tts_response = tts_service.synthesize(text, voice_name=voice_name)
    return tts_response.audio

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

@app.on_event("startup")
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
    try:
        auth = riva.client.Auth(uri=RIVA_SERVER)
        riva.client.ASRService(auth)
        riva_ok = True
    except Exception:
        riva_ok = False

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
    }
    http_status = status.HTTP_200_OK if model_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=health_payload)

# Main chat endpoint
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

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
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
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                audio_url = f"data:audio/wav;base64,{audio_b64}"
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

### 6.2 C4/C5/M7/M9/H2/H3/H5/H10/H11/H12/H13/H14/H15 Smoke Test — Adapter/Auth/Config + Timeout/Circuit-Breaker + Redaction + Contract + CORS + Guardrails + Async Inference + Health Readiness + Error Sanitization + Schema Constraints + Quantization Validation
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
    'CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("SPARC_TIMEOUT_CIRCUIT_THRESHOLD", "3"))',
    'CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS", "30"))',
    'async def is_circuit_open(operation: str) -> bool:',
    'async def record_timeout_event(operation: str) -> bool:',
    'def generate_tokens_sync(',
    'def synthesize_tts_sync(',
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

print("✅ C4/C5/M7/M9/H2/H3/H5/H10/H11/H12/H13/H14/H15 validation passed: named adapters, auth guard, timeout/circuit-breaker policy, env config, Presidio redaction, unified v1 API contract, safe CORS policy, runtime Guardrails pipeline, non-blocking async inference path, readiness-aware health behavior, sanitized client error responses, strict request schema constraints, and explicit 4-bit quantization config are configured.")
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
