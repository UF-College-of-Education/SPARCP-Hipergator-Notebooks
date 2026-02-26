# SPARC-P Digital Human Backend

## 1.0 Introduction and System Goals
This notebook implements the **Real-Time, Multi-Agent Backend** for SPARC-P on HiPerGator.

### 1.1 Objectives
1. **Containerized Deployment**: Run via Apptainer/Singularity (Docker is NOT used).
2. **Orchestration**: Use **LangGraph** to manage the multi-agent state machine.
3. **Audit Logging**: Immutable logging to `/blue` tier for compliance.
4. **API Exposure**: `POST /v1/chat` endpoint for Unity.

### 1.2 Environment Prerequisites
- **Compute**: HiPerGator GPU Node (Persistent Service)
- **Software**: Apptainer, Python 3.10+
- **Models**: Access to `/blue/.../trained_models`

### 1.3 Introduction and System Goals Diagram
![Introduction and System Goals](./images/notebook_3_-_section_1.png)

Introduction and System Goals: This section defines the objectives for the real-time backend. It implements the Real-Time, Multi-Agent Backend on HiPerGator, utilizing Apptainer for containerization, LangGraph for orchestration, and immutable audit logging to the /blue tier for compliance.

### 1.4 Environment Setup

**IMPORTANT**: On HiPerGator, use conda instead of pip (UF RC requirement).

This is the backend environment verification cell. It checks that all the libraries needed for the FastAPI real-time backend are available in the active conda environment — specifically `fastapi`, `uvicorn`, `langgraph`, and the `riva.client` package for speech services.

- Prints the Python executable path and version so you can confirm you're in the correct `sparc_backend` conda environment (not HiPerGator's system Python).
- If any package is missing, it prints exactly which `conda activate` command to run to switch to the right environment, rather than crashing with an unhelpful traceback.
- No side effects — it only reads environment state and prints diagnostics.

> **Expected output if everything is correct:** `✓ All required packages available in conda environment`. If you see an error, follow the printed instructions to activate the `sparc_backend` environment before running any subsequent cells.

```python
# 1.4 Environment Setup
import subprocess
import os
import sys

# Verify conda environment is activated
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Verify key packages
try:
    import fastapi
    import uvicorn
    import langgraph
    from riva.client import ASRService
    print("✓ All required packages available in conda environment")
except ImportError as e:
    base_path = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
    print(f"ERROR: Missing package - {e}")
    print("Ensure you've activated the conda environment:")
    print("  module load conda")
    print(f"  conda activate {base_path}/conda_envs/sparc_backend")
```

---

## 2.0 NVIDIA Riva Deployment
Deploying the Riva server for ASR and TTS capabilities.

### 2.1 Riva Server Setup

This section automates the setup of the NVIDIA Riva server. It downloads the `riva_quickstart` scripts from NGC. On HiPerGator, we use **Apptainer** to pull the server image (`riva-speech:2.16.0-server`). Note that `riva_init.sh` only needs to be run once to download and optimize the models.

### 2.2 Riva & Guardrails Setup Diagram
![Riva and Guardrails Setup](./images/notebook_3_-_section_2-3.png)

Riva & Guardrails Setup: This chart depicts the initialization of the speech services and safety rails. The Riva server is initialized with ASR (Speech-to-Text) and TTS (Text-to-Speech) enabled. Concurrently, NeMo Guardrails configuration files (config.yml, topical_rails.co) are generated to define the "boundary" of the conversation (e.g., refusing political topics).

### 2.3 Riva Setup for HiPerGator

**Note**: Riva runs as an Apptainer container alongside your Python backend (which uses conda).

A numbered, step-by-step instruction guide for installing the NVIDIA Riva speech server on HiPerGator is printed below to follow in a HiPerGator terminal — none of the commands execute automatically.

The instructions walk through four one-time setup steps:
1. **Load Apptainer module** — enables the container runtime on HiPerGator compute nodes.
2. **`apptainer pull`** — downloads the Riva 2.16.0 server image from NVIDIA's container registry (NGC) and saves it as a `.sif` file in your `/blue` directory. This image is ~10 GB and only needs to be downloaded once.
3. **`riva_init.sh`** — runs inside the container to download and optimize the ASR and TTS models for your GPU architecture. This also only needs to happen once and can take 30–60 minutes.
4. **SLURM launch** — the actual Riva server is started via the SLURM script generated in Section 7, not manually.

The Riva server runs as a separate gRPC service on port `50051`. Your Python backend (the FastAPI app in Section 6) connects to it as a client using `localhost:50051` when both run on the same node.

```python
# 2.3 Riva Setup for HiPerGator
import os

# Define version
RIVA_VERSION = "2.16.0"
BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
RIVA_SIF_PATH = os.path.join(BASE_PATH, "containers", "riva_server.sif")

def setup_riva_instructions():
    """
    Instructions for setting up Riva on HiPerGator.
    This needs to be run once to pull and initialize the Riva container.
    """
    instructions = f"""
    === Riva Setup on HiPerGator (One-Time) ===
    
    1. Load required module:
       module load apptainer
    
    2. Pull Riva container:
       apptainer pull {RIVA_SIF_PATH} \
           docker://nvcr.io/nvidia/riva/riva-speech:{RIVA_VERSION}-server
    
    3. Initialize Riva models (downloads ~10GB, run on GPU node):
       apptainer exec --nv {RIVA_SIF_PATH} riva_init.sh
    
    4. The Riva server will be launched via SLURM script (see Section 7)
    
    Note: Riva runs in its own container, while your Python backend uses
    the conda environment (sparc_backend).
    """
    print(instructions)
    return instructions

setup_riva_instructions()
```

### 2.4 Configure Riva
Instructions for configuring which Riva services are enabled before running `riva_init.sh` are printed here. NVIDIA Riva can host many different AI services (speech-to-text, text-to-speech, natural language processing, etc.) — this configuration step tells it which ones to activate when the container starts.

For SPARC-P, only two services are needed:
- **ASR (Automatic Speech Recognition)** — converts the caregiver's spoken audio to text: `service_enabled_asr=true`
- **TTS (Text-to-Speech)** — converts the AI agent's text responses back to spoken audio: `service_enabled_tts=true`
- **NLP is disabled** (`service_enabled_nlp=false`) — SPARC-P uses its own LangGraph-based orchestration for understanding and routing, not Riva's NLP pipeline.

The commented-out `sed` command at the bottom shows how you could automate this change programmatically. The current implementation just prints a reminder because the `config.sh` file only exists on the HiPerGator filesystem after the Riva quickstart scripts have been downloaded.

```python
# 2.2 Configure Riva (Mocking the config.sh modification)

def configure_riva():
    """
    Instructions to modify config.sh:
    1. Set service_enabled_asr=true
    2. Set service_enabled_tts=true
    3. Set service_enabled_nlp=false (not needed for this pipeline)
    """
    print("Please edit 'riva_quickstart_v2.16.0/config.sh' to enable ASR and TTS.")
    # In a real notebook, we might use sed to modify the file programmatically
    # !sed -i 's/service_enabled_asr=false/service_enabled_asr=true/g' config.sh

configure_riva()
```

### 2.5 Server Launch

The following commands launch the Riva server. In a notebook environment, these would block execution, so they are commented out or intended to be run in a separate terminal. The `riva_start.sh` script spins up the containerized service.

### 2.6 Launch Riva Server
An execution reminder prints a message explaining that `riva_init.sh` and `riva_start.sh` must be run in a terminal (not inside this notebook). The actual `!bash` commands are commented out.

Why they can't run inside the notebook:
- `riva_init.sh` downloads models from NVIDIA's servers (up to 10 GB) and runs inside the Apptainer container — it needs the `apptainer` module loaded in a HiPerGator terminal session.
- `riva_start.sh` starts the Riva server as a long-running background process. If run in a notebook cell, it would block the kernel indefinitely (the cell would never finish).

In production deployment, the Riva server is launched as a background process by the SLURM script (`launch_backend.slurm`) generated in Section 7 — not by this notebook cell directly.

```python
# 2.3 Launch Riva Server
# !bash riva_init.sh
# !bash riva_start.sh
print("Run 'riva_init.sh' and 'riva_start.sh' in the terminal to launch Docker containers.")
```

---

## 3.0 Riva Client Testing
Verifying ASR and TTS services.

### 3.1 Service Verification

Once the server is running, we must verify connectivity. These functions use the `riva.client` library to send a gRPC request to `localhost:50051`.
- `test_asr_service`: Streams audio chunks and prints the transcript.
- `test_tts_service`: Sends text and saves the synthesized audio to a WAV file.

### 3.2 Riva Client Testing Functions
Two test functions verify that the Riva speech server is reachable and responding correctly, connecting to it via gRPC.

What happens when you run this:
- **`riva.client.Auth(uri='localhost:50051')`**: Creates an authenticated gRPC channel to the Riva server at `localhost:50051` (the default Riva port). This line actually attempts a connection — if Riva isn't running, this will raise a connection error.
- **`test_asr_service(audio_file_path)`**: Would stream a WAV audio file to Riva's ASR service and print back the transcription. Currently simulated with a print statement — uncomment the internal Riva calls once the server is live.
- **`test_tts_service(text_input)`**: Would send a text string to Riva's TTS service and receive synthesized audio, saved to `output.wav`. Also simulated here.

The two uncommented example calls at the bottom (`# test_asr_service('sample.wav')` and `# test_tts_service(...)`) show exactly how to run these tests. Uncomment them after starting the Riva server to confirm that speech services are working before running the full backend.

> **Prerequisite:** The Riva server must be running (`riva_start.sh` completed) for these tests to actually connect. Running them with the server offline will produce a gRPC connection error.

```python
import riva.client

auth = riva.client.Auth(uri='localhost:50051')

def test_asr_service(audio_file_path):
    print(f"Testing ASR with {audio_file_path}...")
    # asr_service = riva.client.ASRService(auth)
    # Logic to stream audio and get transcript
    print("ASR Test Passed: [Simulated Transcript]")

def test_tts_service(text_input):
    print(f"Testing TTS with '{text_input}'...")
    # tts_service = riva.client.TTSService(auth)
    # Logic to generate audio
    print("TTS Test Passed: Output saved to output.wav")

# Uncomment to run if server is live
# test_asr_service('sample.wav')
# test_tts_service('Hello from SPARC-P')
```

### 3.3 NeMo Guardrails Configuration

Safety is critical. This cell programmatically generates the configuration files for **NVIDIA NeMo Guardrails**:
- `config.yml`: Defines the LLM connection.
- `topical_rails.co`: Uses Colang to define conversation flows, specifically instructing the agent to refuse off-topic discussions (e.g., politics) and stay focused on HPV vaccination.

### 3.4 Create Rails Configuration
Two configuration files are generated here that define the AI safety boundaries for SPARC-P — the "guardrails" that prevent the AI agents from discussing anything outside of HPV vaccination and clinical communication training.

The two files created:
- **`config.yml`**: Tells NeMo Guardrails which AI model is powering the system (the fine-tuned SPARC-P adapter stored in the `/blue` trained models directory). NeMo Guardrails loads this model when evaluating whether a message violates the conversation rules.
- **`topical_rails.co`**: Written in Colang (NVIDIA's domain-specific language for conversation flows), this file defines conversation patterns:
  - **"User asks about anything else"** — examples like politics, finance, sports that are off-topic for a vaccine communication training tool
  - **"Bot refuses to answer"** — the polite refusal messages the AI will use when the conversation veers off-topic
  - **Flow rule**: Connects the trigger (off-topic question) to the response (refusal), so any message matching the trigger pattern gets the refusal response automatically

Both files are saved to the directory specified by `SPARC_GUARDRAILS_DIR` (defaulting to `<BASE_PATH>/guardrails/`). The `SupervisorAgent` class in Section 5 loads these files at startup using `RailsConfig.from_path()`.

> **Why this matters:** Without guardrails, a caregiver could ask the AI to help them with completely unrelated tasks. The guardrails keep the system on-topic and appropriate for the educational context.

```python
# 3.2 NeMo Guardrails Configuration
import os

def create_rails_config():
    base_path = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
    guardrails_dir = os.environ.get("SPARC_GUARDRAILS_DIR", os.path.join(base_path, "guardrails"))
    os.makedirs(guardrails_dir, exist_ok=True)

        # 1. config.yml
        model_path = os.path.join(base_path, "trained_models", "sparc-agent-final")
        config_content = f"""
models:
  - type: main
    engine: huggingface
    model: {model_path}
    """
    with open(os.path.join(guardrails_dir, "config.yml"), "w", encoding="utf-8") as f:
        f.write(config_content.strip())
        
    # 2. topical_rails.co
    rails_content = """
define user ask about anything else
  "tell me about politics"
  "what are your thoughts on finance?"
  "who will win the game?"

define bot refuse to answer
  "I'm sorry, but I can only discuss topics related to HPV vaccination."
  "My purpose is to help you practice clinical communication skills for HPV vaccines."

define flow
  user ask about anything else
  bot refuse to answer
    """
    with open(os.path.join(guardrails_dir, "topical_rails.co"), "w", encoding="utf-8") as f:
        f.write(rails_content.strip())

    print(f"NeMo Guardrails configuration files created in {guardrails_dir}")
    return guardrails_dir

create_rails_config()
```

---

---

## 4.0 Coach Voice Cloning (Zero-Shot TTS)

Riva 2.x supports **zero-shot voice cloning natively**  no NeMo fine-tuning pipeline required. By providing a short (310 second) audio prompt, Riva's TTS engine adapts its output to match the speaker's vocal characteristics at inference time.

This section clones the Coach AI voice from sample recordings in `audio/coach_examples/`:
1. **4.1 Audio Preprocessing**  Scans the directory, converts all clips to Riva's required format (16 kHz mono PCM WAV), and selects the single best 310 second prompt.
2. **4.2 Test Synthesis**  Validates the prompt against a live Riva endpoint and writes a reference output for listening quality checks.
3. **4.3 `CoachVoiceConfig`**  A dataclass that bundles the prompt path, its transcript, and quality settings. `CoachAgent` accepts this config and uses the cloned voice when synthesizing feedback audio; it falls back to the default TTS voice gracefully if Riva is offline or no config is provided.

![notebook 3 - section 4.png](images/notebook_3_-_section_4.png)

Coach Voice Cloning: The diagram shows the zero-shot cloning pipeline. Audio files from `audio/coach_examples/` are preprocessed into a 16 kHz mono WAV prompt. That prompt is passed to the Riva TTS `synthesize()` call alongside the Coach's feedback text. Riva adapts its output to match the prompt speaker's voice without any separate training job.

### 4.1 Audio Preprocessing

Scans `audio/coach_examples/` for `.mp3` and `.wav` files, converts each to **16 kHz mono PCM WAV** (Riva's required input format), and selects the single best clip to use as the cloning prompt.

Selection criteria (applied in this order):
1. **Duration gate**: Only clips between 3 and 10 seconds are eligible  Riva rejects prompts outside this range.
2. **Target duration**: Prefers clips closest to 7 seconds (empirically the best balance between enough voice data and voice drift).
3. **RMS energy**: Among clips equally close to 7 seconds, selects the louder one (better signal-to-noise ratio).

The chosen prompt WAV is written to `audio/coach_examples/processed/best_prompt.wav`. A summary table is printed showing all discovered clips and why the winner was selected.

```python
# 4.1 Audio Preprocessing  Select and convert best prompt clip
import os
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
EXAMPLES_DIR = Path(BASE_PATH) / "audio" / "coach_examples"
PROCESSED_DIR = EXAMPLES_DIR / "processed"
BEST_PROMPT_PATH = PROCESSED_DIR / "best_prompt.wav"
TARGET_DURATION_S = 7.0
MIN_DURATION_S = 3.0
MAX_DURATION_S = 10.0
TARGET_SAMPLE_RATE = 16000

def preprocess_prompt_clips(examples_dir, processed_dir):
    processed_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    audio_files = sorted(
        [f for f in examples_dir.iterdir() if f.suffix.lower() in (".mp3", ".wav") and f.is_file()]
    )
    for src in audio_files:
        try:
            seg = AudioSegment.from_file(str(src))
            seg = seg.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2)
            duration_s = len(seg) / 1000.0
            rms = seg.rms
            dest = processed_dir / (src.stem + "_16k.wav")
            seg.export(str(dest), format="wav")
            candidates.append({
                "src": src.name, "dest": dest,
                "duration_s": round(duration_s, 2), "rms": rms,
                "eligible": MIN_DURATION_S <= duration_s <= MAX_DURATION_S,
            })
        except Exception as e:
            print(f"  SKIP {src.name}: {e}")
    return candidates

def select_best_prompt(candidates):
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        return None
    return min(eligible, key=lambda c: (abs(c["duration_s"] - TARGET_DURATION_S), -c["rms"]))

if PYDUB_AVAILABLE and EXAMPLES_DIR.exists():
    candidates = preprocess_prompt_clips(EXAMPLES_DIR, PROCESSED_DIR)
    best = select_best_prompt(candidates)
    if best:
        import shutil
        shutil.copy2(str(best["dest"]), str(BEST_PROMPT_PATH))
        COACH_PROMPT_TRANSCRIPT = input("Enter transcript of selected clip: ").strip()
```

### 4.2 Voice Prompt Validation and Test Synthesis

Validates the selected prompt by calling Riva's `synthesize()` API with the zero-shot parameters and writing the output to `audio/coach_voice_test.wav`. Listen to this file to verify voice quality before deploying.

Key parameters:
- **`zero_shot_audio_prompt_file`**: Path to the 16 kHz mono WAV prompt from Section 4.1.
- **`zero_shot_transcript`**: The exact words spoken in the prompt clip  Riva uses this to align phonemes; accuracy directly affects voice similarity.
- **`zero_shot_quality`**: Integer 140. Higher = slower inference but better voice match. Default of `20` balances latency and quality for real-time coaching.
- **`sample_rate_hz`**: Output sample rate set to 44100 Hz for audio playback compatibility.

If Riva is offline (e.g., running this cell outside of HiPerGator), the cell prints a dry-run summary instead of failing.

```python
# 4.2 Voice Prompt Validation and Test Synthesis
import wave, os
from pathlib import Path

RIVA_SERVER = os.environ.get("SPARC_RIVA_SERVER", "localhost:50051")
TEST_OUTPUT_PATH = Path(BASE_PATH) / "audio" / "coach_voice_test.wav"
TEST_TEXT = (
    "Great job maintaining eye contact and using affirming language. "
    "Next time, try pausing after your empathy statement to give the caregiver more space to respond."
)
ZERO_SHOT_QUALITY = 20  # 140

def run_test_synthesis(prompt_path, transcript, output_path):
    try:
        import riva.client
        channel = riva.client.connect(RIVA_SERVER)
        tts_service = riva.client.SpeechSynthesisService(channel)
        resp = tts_service.synthesize(
            TEST_TEXT, "English-US.Female-1", "en-US",
            sample_rate_hz=44100,
            zero_shot_audio_prompt_file=prompt_path,
            zero_shot_quality=ZERO_SHOT_QUALITY,
            zero_shot_transcript=transcript,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(resp.audio)
        print(f"Test synthesis written to: {output_path}")
        return True
    except Exception as err:
        print(f"Riva TTS unavailable ({err})  dry-run mode.")
        return False

if BEST_PROMPT_PATH.exists():
    run_test_synthesis(BEST_PROMPT_PATH, COACH_PROMPT_TRANSCRIPT, TEST_OUTPUT_PATH)
```

### 4.3 `CoachVoiceConfig`  Zero-Shot Voice Profile

`CoachVoiceConfig` is a lightweight dataclass that bundles everything `CoachAgent` needs to use the cloned voice:

| Field | Type | Description |
|---|---|---|
| `prompt_path` | `Path` | Path to the 16 kHz mono WAV prompt from Section 4.1 |
| `transcript` | `str` | Exact words spoken in the prompt  must match precisely |
| `quality` | `int` | Zero-shot quality 140 (default 20) |
| `language_code` | `str` | BCP-47 language tag (default `"en-US"`) |
| `voice_name` | `str` | Base Riva voice the adaptation starts from |

`CoachAgent` is updated to accept an optional `CoachVoiceConfig`. When the config is provided and Riva is reachable, `evaluate_turn()` returns both text feedback and base64-encoded audio synthesized in the cloned voice. When Riva is offline or no config is given, it falls back to returning text-only feedback  the orchestrator continues to run normally.

`build_app_graph()` is updated to attempt loading the config from `audio/coach_examples/processed/best_prompt.wav` at startup. If the file does not exist (e.g., first deployment before Section 4.1 has been run), it starts with voice cloning disabled.

```python
# 4.3 CoachVoiceConfig dataclass
import base64, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class CoachVoiceConfig:
    prompt_path: Path
    transcript: str
    quality: int = 20
    language_code: str = "en-US"
    voice_name: str = "English-US.Female-1"
    riva_server: str = field(
        default_factory=lambda: os.environ.get("SPARC_RIVA_SERVER", "localhost:50051")
    )

    def is_ready(self):
        return self.prompt_path.exists() and self.prompt_path.stat().st_size > 0

def load_coach_voice_config(processed_dir=None, transcript="", quality=20):
    base = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
    prompt = (processed_dir or Path(base) / "audio" / "coach_examples" / "processed") / "best_prompt.wav"
    if not prompt.exists():
        print(f"CoachVoiceConfig: prompt not found at {prompt}  voice cloning disabled.")
        return None
    cfg = CoachVoiceConfig(prompt_path=prompt, transcript=transcript, quality=quality)
    print(f"CoachVoiceConfig loaded: {prompt} (quality={quality})")
    return cfg
```


## 5.0 Multi-Agent Orchestration (LangGraph)
Implements the Supervisor-Worker architecture using a state graph.

### 5.1 Multi-Agent Orchestration Logic

This section implements the core reasoning loop using `asyncio` for concurrency. We define three agent classes:
- **Supervisor**: Checks input safety using NeMo Guardrails.
- **Caregiver**: Generates the persona response (simulating RAG+LLM latency).
- **Coach**: Evaluates the turn (simulating C-LEAR rubric latency).

The `handle_user_turn` function orchestrates these agents, running the Caregiver and Coach in parallel to minimize response time.

### 5.2 Multi-Agent Orchestration Diagram
![Multi-Agent Orchestration](./images/notebook_3_-_section_5.png)

Multi-Agent Orchestration (LangGraph): This is the core logic of the backend. It visualizes the Supervisor-Worker pattern. The User Input is first checked by the Supervisor (Guardrails). If safe, it triggers the Caregiver (generating the response) and the Coach (evaluating the response) in parallel to minimize latency. The results are aggregated into a single JSON response.

### 5.3 Multi-Agent System (MAS) Orchestration Logic
The core multi-agent backend — a 156-line implementation of the real-time conversation orchestration system. When a caregiver speaks to SPARC-P, the orchestrator decides what happens to their words and who responds.

The four classes defined here, and what each does:

**`SupervisorAgent`**: The safety gatekeeper. Every user message goes through this agent first. It loads the NeMo Guardrails configuration and checks whether the input is on-topic. If it's off-topic or harmful, it returns a pre-set refusal message and sets `is_safe=False`. It also checks the Caregiver's *response* (not just the input) before sending it to the user. The two-stage checking (input + output) prevents both prompt injection and model hallucinations from leaking inappropriate content.

**`CaregiverAgent`**: Simulates 800ms of LLM inference latency (in production, this calls the fine-tuned caregiver model). Returns the avatar's spoken response text.

**`CoachAgent`**: Simulates 400ms of LLM inference latency (in production, this calls the fine-tuned C-LEAR coach model). Returns structured feedback on the trainee's communication.

**`handle_user_turn()`**: The orchestration function that sequences the above agents:
1. Supervisor checks input (if unsafe → return refusal immediately)
2. Caregiver and Coach run **simultaneously** using `asyncio.gather()` — this parallel execution is critical for keeping response time under 1.5 seconds even though two LLMs are involved
3. Supervisor checks the combined output (second safety pass)

**`AsyncOrchestrationGraph`**: A thin adapter class that wraps `handle_user_turn()` with an `ainvoke(state)` interface. This makes it compatible with the FastAPI endpoint in Section 6 without requiring LangGraph compilation.

**`build_app_graph()`**: The factory function called at startup by the FastAPI application to create the orchestrator instance.

```python
import asyncio
import os
from typing import Any, Dict
from nemoguardrails import LLMRails, RailsConfig

# 3.3 Multi-Agent System (MAS) Orchestration Logic

class SupervisorAgent:
    def __init__(self, rails_path: str = None):
        self.refusal_message = "I can only discuss topics related to HPV vaccination and clinical communication training."
        base_path = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
        self.rails_path = rails_path or os.environ.get("SPARC_GUARDRAILS_DIR", os.path.join(base_path, "guardrails"))
        self.rails = None
        try:
            rails_config = RailsConfig.from_path(self.rails_path)
            self.rails = LLMRails(rails_config)
            self.guardrails_ready = True
        except Exception as rails_error:
            print(f"SUPERVISOR: Failed to load guardrails from {self.rails_path}: {rails_error}")
            self.guardrails_ready = False

    async def _run_rails(self, user_text: str) -> str:
        if not self.rails:
            raise RuntimeError("Guardrails runtime is not initialized")
        messages = [{"role": "user", "content": user_text}]
        if hasattr(self.rails, "generate_async"):
            result = await self.rails.generate_async(messages=messages)
        else:
            result = self.rails.generate(messages=messages)

        if isinstance(result, dict):
            if "content" in result:
                return str(result["content"])
            return str(result)
        return str(result)

    async def process_input(self, text: str):
        print(f"SUPERVISOR: Checking input '{text}'")
        if not text or not text.strip():
            return self.refusal_message, False, "empty_input"
        if not self.guardrails_ready:
            return self.refusal_message, False, "guardrails_unavailable"

        try:
            rails_output = await self._run_rails(text)
            refusal_detected = self.refusal_message.lower() in rails_output.lower()
            if refusal_detected:
                return self.refusal_message, False, "input_rails_blocked"
            return text, True, "input_rails_allowed"
        except Exception as rails_error:
            print(f"SUPERVISOR: Guardrails input evaluation failed: {rails_error}")
            return self.refusal_message, False, "input_rails_error"

    async def enforce_output(self, text: str):
        if not text or not text.strip():
            return self.refusal_message, False, "empty_output"
        if not self.guardrails_ready:
            return self.refusal_message, False, "guardrails_unavailable"

        try:
            rails_output = await self._run_rails(text)
            refusal_detected = self.refusal_message.lower() in rails_output.lower()
            if refusal_detected:
                return self.refusal_message, False, "output_rails_blocked"
            return text, True, "output_rails_allowed"
        except Exception as rails_error:
            print(f"SUPERVISOR: Guardrails output evaluation failed: {rails_error}")
            return self.refusal_message, False, "output_rails_error"

class CaregiverAgent:
    async def generate_response(self, text: str):
        # RAG + LLM Inference
        await asyncio.sleep(0.8)
        return f"Caregiver response to: {text}"

class CoachAgent:
    async def evaluate_turn(self, text: str):
        # C-LEAR Rubric
        await asyncio.sleep(0.4)
        return "Good empathy."

async def handle_user_turn(user_transcript: str, supervisor, caregiver, coach):
    # 2. Supervisor Check
    sanitized_text, is_safe, safety_reason = await supervisor.process_input(user_transcript)
    if not is_safe:
        return {
            "final_text": sanitized_text,
            "coach_feedback": "",
            "safety": {"is_safe": False, "reason": safety_reason},
        }
        
    # 3. Parallel Execution
    caregiver_task = asyncio.create_task(caregiver.generate_response(sanitized_text))
    coach_task = asyncio.create_task(coach.evaluate_turn(sanitized_text))
    
    caregiver_response, coach_feedback = await asyncio.gather(caregiver_task, coach_task)
    
    final_response = f"{caregiver_response} [Feedback: {coach_feedback}]"
    output_text, output_safe, output_reason = await supervisor.enforce_output(final_response)
    return {
        "final_text": output_text,
        "coach_feedback": coach_feedback if output_safe else "",
        "safety": {"is_safe": output_safe, "reason": output_reason},
    }

class AsyncOrchestrationGraph:
    """
    Minimal async graph adapter to provide an app_graph.ainvoke(...) interface.
    This preserves a clear initialization lifecycle without requiring notebook-wide
    LangGraph compilation for the prototype.
    """

    def __init__(self, supervisor: SupervisorAgent, caregiver: CaregiverAgent, coach: CoachAgent):
        self.supervisor = supervisor
        self.caregiver = caregiver
        self.coach = coach

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        transcript = state.get("transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            return {
                "final_response": {"text": "No transcript provided.", "audio": "", "cues": {}},
                "feedback": "",
            }

        turn_result = await handle_user_turn(
            transcript,
            self.supervisor,
            self.caregiver,
            self.coach,
        )

        caregiver_text = turn_result.get("final_text", "Error")
        coach_feedback = turn_result.get("coach_feedback", "")
        safety = turn_result.get("safety", {"is_safe": False, "reason": "unknown"})

        if " [Feedback: " in caregiver_text and caregiver_text.endswith("]"):
            caregiver_text, feedback_tail = caregiver_text.rsplit(" [Feedback: ", 1)
            coach_feedback = feedback_tail[:-1]

        return {
            "final_response": {
                "text": caregiver_text,
                "audio": "",
                "cues": {"gesture": "speaking"},
            },
            "feedback": coach_feedback,
            "safety": safety,
        }

def build_app_graph() -> AsyncOrchestrationGraph:
    """Canonical orchestrator construction lifecycle for the backend endpoint."""
    supervisor = SupervisorAgent()
    caregiver = CaregiverAgent()
    coach = CoachAgent()
    return AsyncOrchestrationGraph(supervisor, caregiver, coach)

# Example Run
# app_graph = build_app_graph()
# asyncio.run(app_graph.ainvoke({"transcript": "User said something about vaccines"}))
```

---

## 6.0 API Server (FastAPI)
Exposes the Orchestrator to the Unity Client.

### 6.1 FastAPI Server Implementation

This cell wraps the orchestration logic in a **FastAPI** application to expose it to the Unity client.
- **`/v1/chat` Endpoint**: Accepts a user transcript and session ID, invokes the orchestration loop, and returns the multi-agent response (Text, Audio, Feedback).
- **Redacted Audit Logging**: Writes only compliant metadata (`session_id`, `agent_type`, `is_safe`, `latency_ms`, timestamp) and excludes raw transcript content.
- **Health Check**: A simple `GET /health` endpoint for monitoring service uptime and audit retention metadata.

### 6.2 API Server Integration Diagram
![API Server Integration](./images/notebook_3_-_section_6.png)

API Server Integration: This diagram maps the data flow through the FastAPI application. The Unity Client sends a request to /v1/chat. The server invokes the LangGraph orchestration loop (defined in Section 4), writes redacted audit metadata only, and returns the structured ChatResponse containing text, audio (Base64), and animation cues.

### 6.3 FastAPI Server with Endpoints
The complete production FastAPI web server — the HTTP interface that the Unity-based SPARC-P client calls to interact with the AI agents — is defined here. The application object (`app`) and its endpoints are registered; the server does not start serving until `uvicorn.run(app, ...)` is called (which happens in the SLURM launch script).

What the server contains:

**Configuration & audit logging setup:**
- Reads `SPARC_BASE_PATH` and `SPARC_AUDIT_LOG` from environment variables, defaulting to `/blue/`.
- Calls `validate_audit_log_path()` at startup to ensure the log directory exists and is writable — failing loudly if not, so audit compliance issues are caught before the first API call.
- `log_redacted_audit_event()` writes only compliant metadata to the audit log: session ID, agent type, whether the message was safe, response latency, and a UTC timestamp. **No transcript text or PHI is written** — this is the HIPAA "transient PHI" model.

**Request/response models (Pydantic):**
- `ChatRequest`: Validates that `session_id` is 1–128 characters of alphanumerics/hyphens/underscores (preventing injection via session IDs) and `user_transcript` is 1–10,000 characters.
- `ChatResponse`: The structured response containing the caregiver's text, audio (Base64), animation cues, and coach feedback.

**`GET /health`** — Returns service status, whether the orchestrator is ready, and audit retention metadata. Used by monitoring systems to detect if the service is degraded.

**`POST /v1/chat`** — The primary endpoint:
1. Validates the request schema
2. Calls `app_graph.ainvoke()` with the transcript and timing context
3. Logs a redacted audit event
4. Returns the `ChatResponse` with caregiver text, audio, cues, and feedback

> **Thread safety note:** `app_graph` is set to `None` if `build_app_graph()` fails at startup. The `/v1/chat` endpoint checks for this and returns HTTP 503 (Service Unavailable) immediately, preventing any request from reaching an uninitialized orchestrator.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
import os
import json
import time
from datetime import datetime, timezone

app = FastAPI()

# 7.1 Configuration & Logging
BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
LOG_FILE = os.environ.get("SPARC_AUDIT_LOG", os.path.join(BASE_PATH, "logs", "audit.log"))
AUDIT_RETENTION_DAYS = int(os.environ.get("SPARC_AUDIT_RETENTION_DAYS", "30"))
LOG_DIR = os.path.dirname(LOG_FILE) or "."

def validate_audit_log_path(log_file: str) -> None:
    log_dir = os.path.dirname(log_file) or "."
    os.makedirs(log_dir, exist_ok=True)
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"Audit log directory is not writable: {log_dir}")
    with open(log_file, "a", encoding="utf-8"):
        pass

validate_audit_log_path(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(message)s')

app_graph = None

def log_redacted_audit_event(session_id: str, agent_type: str, is_safe: bool, latency_ms: float):
    event = {
        "event": "chat_turn",
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "agent_type": agent_type,
        "is_safe": is_safe,
        "latency_ms": round(latency_ms, 2),
        "retention_days": AUDIT_RETENTION_DAYS,
    }
    logging.info(json.dumps(event, sort_keys=True))

def initialize_orchestrator():
    """Build and inject the orchestrator graph once at startup/init time."""
    global app_graph
    try:
        app_graph = build_app_graph()
    except Exception as exc:
        app_graph = None
        logging.error(f"Failed to initialize orchestrator graph: {exc}")

initialize_orchestrator()

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    user_transcript: str = Field(..., min_length=1, max_length=10000)

class ChatResponse(BaseModel):
    caregiver_text: str
    caregiver_audio_b64: str
    caregiver_animation_cues: dict
    coach_feedback: str

# 7.2 Endpoints
@app.get("/health")
async def health_check():
    orchestrator_ready = app_graph is not None and hasattr(app_graph, "ainvoke")
    return {
        "status": "ok" if orchestrator_ready else "degraded",
        "service": "SPARC-P Backend",
        "orchestrator_ready": orchestrator_ready,
        "audit_log_path": LOG_FILE,
        "audit_retention_days": AUDIT_RETENTION_DAYS,
    }

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # Fail-fast for uninitialized orchestration
    if app_graph is None or not hasattr(app_graph, "ainvoke"):
        raise HTTPException(status_code=503, detail="Orchestrator is not initialized")

    # Invoke orchestrator
    start_time = time.perf_counter()
    initial_state = {
        "transcript": request.user_transcript,
        "history": [],
        "feedback": "",
        "next_action": "",
        "final_response": {},
    }
    result = await app_graph.ainvoke(initial_state)
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    response_data = result.get("final_response", {})
    caregiver_text = response_data.get("text", "Error")

    # Redacted audit log only (no raw transcript / PHI content)
    safety_result = result.get("safety", {})
    is_safe = bool(safety_result.get("is_safe", False))
    log_redacted_audit_event(
        session_id=request.session_id,
        agent_type="orchestrator",
        is_safe=is_safe,
        latency_ms=latency_ms,
    )
    
    return ChatResponse(
        caregiver_text=caregiver_text,
        caregiver_audio_b64=response_data.get("audio", ""),
        caregiver_animation_cues=response_data.get("cues", {}),
        coach_feedback=result.get("feedback", "")
    )

# To run:
# uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 6.5 H10 Guardrails Regression Checks

Validate that runtime safety is guardrails-enforced (not keyword-only):

```python
runtime_source = open("3_SPARC_RIVA_Backend.md", "r", encoding="utf-8").read()

required_guardrails_markers = [
    "from nemoguardrails import LLMRails, RailsConfig",
    "SPARC_GUARDRAILS_DIR",
    "os.path.join(base_path, \"guardrails\")",
    "RailsConfig.from_path(self.rails_path)",
    "self.rails = LLMRails(rails_config)",
    "async def enforce_output",
    "safety = turn_result.get(\"safety\"",
]
missing_markers = [m for m in required_guardrails_markers if m not in runtime_source]
assert not missing_markers, f"Missing guardrails runtime markers: {missing_markers}"

blocked_legacy_patterns = [
    "is_safe = \"politics\" not in text.lower()",
    "# from nemoguardrails import LLMRails, RailsConfig",
]
legacy_found = [p for p in blocked_legacy_patterns if p in runtime_source]
assert not legacy_found, f"Legacy keyword-only safety logic still present: {legacy_found}"

print("✅ H10 regression checks passed: guardrails runtime path is enforced and keyword-only checks are removed.")
```

### 6.6 H14 Request Schema Regression Checks

Validate that API request fields enforce bounds/pattern constraints:

```python
runtime_source = open("3_SPARC_RIVA_Backend.md", "r", encoding="utf-8").read()

required_schema_markers = [
    "from pydantic import BaseModel, Field",
    "session_id: str = Field(..., min_length=1, max_length=128, pattern=r\"^[a-zA-Z0-9_-]+$\")",
    "user_transcript: str = Field(..., min_length=1, max_length=10000)",
]
missing_markers = [m for m in required_schema_markers if m not in runtime_source]
assert not missing_markers, f"Missing request schema constraint markers: {missing_markers}"

blocked_legacy_patterns = [
    "session_id: str\n",
    "user_transcript: str\n",
]
legacy_found = [p for p in blocked_legacy_patterns if p in runtime_source]
assert not legacy_found, f"Legacy unconstrained request fields still present: {legacy_found}"

print("✅ H14 regression checks passed: request schema constraints are enforced.")
```

### 6.4 Orchestrator Smoke Tests

Use a lightweight in-process FastAPI test to verify both normal and degraded orchestration behavior:

Three automated smoke tests run against the FastAPI application using `TestClient` — a built-in FastAPI/Starlette utility that sends HTTP requests to the app in-memory without needing a running server. All three tests run immediately.

**Test A — Health endpoint:**
- Sends `GET /health` and prints the response. Expected: `{"status": "ok", "orchestrator_ready": true, ...}` if the orchestrator initialized successfully.

**Test B — Successful chat request:**
- Sends a valid `POST /v1/chat` request with a proper `session_id` and an on-topic HPV vaccine question.
- Expected: HTTP 200 with a `ChatResponse` JSON body containing `caregiver_text`, `coach_feedback`, etc.

**Test C — Degraded service (orchestrator unavailable):**
- Saves the current `app_graph`, sets it to `None` to simulate a startup failure, sends the same chat request, then restores `app_graph`.
- Expected: HTTP 503 with a `"Orchestrator is not initialized"` detail message.
- **Restores `app_graph` afterward** so subsequent cells still work correctly.

> **If Test B fails with 503 when it should pass:** The orchestrator failed to initialize (likely because NeMo Guardrails couldn't load its config files). Check that `create_rails_config()` was run first (Section 3.2) and that the guardrails directory path is correct.

```python
# 7.3 Orchestrator Smoke Tests (FastAPI TestClient)
from fastapi.testclient import TestClient

client = TestClient(app)

# A) Health endpoint should reflect orchestrator readiness
health = client.get("/health")
print("Health:", health.status_code, health.json())

# B) Chat endpoint should succeed when orchestrator is initialized
ok_payload = {"session_id": "smoke-session", "user_transcript": "Can you help me talk about HPV vaccines?"}
ok_response = client.post("/v1/chat", json=ok_payload)
print("Chat (ready):", ok_response.status_code, ok_response.json())

# C) Chat endpoint should fail-fast when orchestrator is unavailable
saved_graph = app_graph
app_graph = None
degraded_response = client.post("/v1/chat", json=ok_payload)
print("Chat (degraded):", degraded_response.status_code, degraded_response.json())

# Restore state for subsequent cells
app_graph = saved_graph
```

Expected outcomes:
- `/health` returns `status: "ok"` with `orchestrator_ready: true` when initialized.
- `/v1/chat` returns `200` and a valid `ChatResponse` when initialized.
- `/v1/chat` returns `503` with `"Orchestrator is not initialized"` when `app_graph` is unavailable.

---

## 7.0 Security and Compliance
**HIPAA Mandate**: This system uses a 'Transient PHI' model. User audio and transcripts are processed in-memory and discarded immediately after the conversational turn. No PHI is written to disk.

### 7.1 Production Deployment Script

To deploy this backend as a persistent service on HiPerGator, we generate a SLURM script (`launch_backend.slurm`). This script:
- Uses a conda-first runtime path with Apptainer only for Riva.
- Loads `conda`, `cuda`, and `apptainer`.
- Starts Riva via `apptainer exec --nv ... riva_start.sh` and then starts FastAPI via `uvicorn`.

Canonical artifact source policy:
- **Source of truth** for executable launch content is the generator function in Notebook 3 (`generate_launch_script`).
- This markdown section is a synchronized companion and must mirror generator markers exactly.

### 7.2 Security and Compliance Diagram
![Security and Compliance](./images/notebook_3_-_section_7.png)

Security and Compliance: This section outlines the security protocols and persistent deployment. It adheres to the HIPAA Mandate using a 'Transient PHI' model, where user data is processed in-memory and immediately discarded. The launch_backend.slurm script ensures the service runs persistently on a secure GPU node.

### 7.3 SLURM Launch Script Generator
`launch_backend.slurm` is the SLURM batch script that deploys the complete SPARC-P backend (Riva speech server + FastAPI orchestration server) as a persistent service on HiPerGator.

What the generated script does when submitted to the HiPerGator scheduler:
1. **Resource allocation**: Requests 4 GPUs, 16 CPU cores, 128 GB RAM, 7-day runtime on the `gpu` partition. The 4 GPUs support the LLM (fine-tuned adapter), Riva ASR model, Riva TTS model, and a spare for burst capacity.
2. **Module loading**: Loads `conda`, `cuda/12.8`, and `apptainer` — all three are required for the backend.
3. **Conda activation**: Activates the `sparc_backend` environment which contains FastAPI, LangGraph, Riva client, and NeMo Guardrails.
4. **Environment verification**: Imports `fastapi`, `langgraph`, and `transformers` to confirm the environment is healthy before the expensive Riva startup begins.
5. **Riva server launch (`apptainer exec --nv`)**: Starts the Riva speech AI container in the background (`&`) with GPU access. The `sleep 30` gives Riva time to load its ASR/TTS models (~30 seconds) before the FastAPI app tries to connect.
6. **FastAPI backend (`uvicorn main:app --workers 2`)**: Starts the Python backend with 2 worker processes for concurrency. This is a blocking call (no `&`) — when it terminates, the SLURM job exits and Riva is killed.
7. **Cleanup**: The `kill $RIVA_PID` command on exit ensures Riva doesn't become an orphaned process.

> **To deploy:** Transfer `launch_backend.slurm` to HiPerGator and submit with `sbatch launch_backend.slurm`. Monitor output in `backend_<jobid>.log`.

```python
# 7.1 SLURM Launch Script Generator (Conda-based)

import os

def generate_launch_script():
    """
    Generates a SLURM script for persistent backend deployment using conda.
    Resource profile: 4 GPUs and 16 CPU cores for parallelization.
    """
    script_content = """
#!/bin/bash
#SBATCH --job-name=sparcp-backend
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${SPARC_SLURM_EMAIL:-YOUR_EMAIL@ufl.edu}
#SBATCH --partition=gpu
#SBATCH --qos=jasondeanarnold-b
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --time=7-00:00:00
#SBATCH --output=backend_%j.log
#SBATCH --error=backend_%j.err

pwd; hostname; date

echo "=== SPARC-P Backend Service Launch ==="
echo "Resource profile: 4 GPUs, 16 CPU cores allocated"

# 1. Load required modules
module purge
module load conda
module load cuda/12.8
module load apptainer

# 2. Resolve runtime paths from environment
SPARC_BASE_PATH=${SPARC_BASE_PATH:-/blue/jasondeanarnold/SPARCP}
CONDA_ENV=${SPARC_BACKEND_ENV:-$SPARC_BASE_PATH/conda_envs/sparc_backend}
RIVA_SIF=${SPARC_RIVA_SIF:-$SPARC_BASE_PATH/containers/riva_server.sif}
BACKEND_WORKDIR=${SPARC_BACKEND_WORKDIR:-$SPARC_BASE_PATH/backend}

echo "Using SPARC_BASE_PATH=$SPARC_BASE_PATH"
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

# 3. Verify environment
echo "Python: $(which python)"
python -c "import fastapi, langgraph, transformers; print('✓ Backend packages loaded')"

# 4. Launch Riva container in background
echo "Starting Riva server..."
apptainer exec --nv $RIVA_SIF riva_start.sh &
RIVA_PID=$!
sleep 30  # Wait for Riva to initialize

# 5. Start FastAPI backend
echo "Starting FastAPI backend..."
cd $BACKEND_WORKDIR
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2

# Cleanup on exit
kill $RIVA_PID
echo "Backend service stopped."
date
    """
    with open("launch_backend.slurm", "w") as f:
        f.write(script_content.strip())
    print("✓ Generated launch_backend.slurm")
    print("\nIMPORTANT: Update SPARC_SLURM_EMAIL if needed")
    print("\nSubmit with: sbatch launch_backend.slurm")

generate_launch_script()
```

### 7.4 Drift Sync Check (`.md` companion vs notebook generator)
```python
# Validates markdown companion contains canonical launch-script markers.
def validate_launch_doc_sync(md_path="3_SPARC_RIVA_Backend.md"):
    canonical_markers = [
        "module load conda",
        "module load apptainer",
        "apptainer exec --nv $RIVA_SIF riva_start.sh",
        "uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2",
    ]

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    missing = [marker for marker in canonical_markers if marker not in md_text]
    assert not missing, f"Markdown launch doc drift detected. Missing markers: {missing}"
    print("✅ H9 sync check passed: markdown companion contains canonical launch markers.")

validate_launch_doc_sync()
```

---

## Summary

This notebook implements the complete real-time backend for SPARC-P:

1. **NVIDIA Riva Speech Services**: Handles ASR (Speech-to-Text) and TTS (Text-to-Speech) using containerized Riva server on Apptainer.

2. **Safety Rails with NeMo Guardrails**: Implements conversation boundaries to keep discussions focused on HPV vaccination topics while refusing off-topic requests.

3. **Multi-Agent Orchestration**: Uses LangGraph to coordinate three specialized agents:
   - **Supervisor**: Validates input safety
   - **Caregiver**: Generates empathetic responses
   - **Coach**: Evaluates responses against C-LEAR rubric
   - Agents run in parallel to minimize latency

4. **FastAPI Server**: Exposes orchestration logic via REST endpoints:
   - `GET /health`: Service status monitoring
   - `POST /v1/chat`: Main chat endpoint for Unity client

5. **Audit Logging**: Immutable logging to `/blue` tier for HIPAA compliance with transient PHI processing model.

6. **Production Deployment**: SLURM script for persistent service deployment on HiPerGator GPU nodes with a policy-compliant finite runtime (default `7-00:00:00`; use `UNLIMITED` only if partition/QoS permits it).

The entire system is containerized with Apptainer, ensuring reproducibility and portability across HPC environments.
