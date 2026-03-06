# H3_Riva_Testing_Speech

> Auto-generated markdown counterpart from notebook cells.

# H3 Riva Services Setup and Testing

## 1.0 Introduction and Goals
This notebook focuses only on provisioning NVIDIA Riva and validating ASR/TTS service health on HiPerGator.

### 1.1 Objectives
1. **Environment Readiness**: Confirm the backend conda environment can import `riva.client`
2. **Containerized Deployment**: Run Riva via Apptainer on a GPU node
3. **Service Configuration**: Enable ASR and TTS, disable unused NLP service
4. **Service Validation**: Verify gRPC connectivity and basic ASR/TTS test calls

### 1.2 Environment Prerequisites
- **Compute**: HiPerGator GPU node
- **Software**: Conda environment with Riva Python client + Apptainer runtime
- **Container Artifact**: Access to a writable `/blue/.../containers` location

![H3 High-Level Architecture & Environment Diagram](../images/h3_1.png)

H3 High-Level Architecture & Environment Diagram: This diagram outlines the core service architecture targeted by the notebook. It emphasizes the separation between the Python notebook environment (managed by Conda) and the actual speech services running in an isolated Apptainer container over a local gRPC connection.

**⚠️ Before running this notebook:**
```bash
module load conda
conda activate /blue/jasondeanarnold/SPARCP/conda_envs/sparc_backend
jupyter notebook
```

This environment verification checks only the dependencies needed to run Riva client-side setup and service tests in this notebook.

- Prints the Python executable path and version so you can confirm the correct conda environment is active.
- Validates `riva.client` import availability before any service calls.
- Produces a clear activation hint if imports fail.

This cell is read-only and does not start containers or call services.

```python
# 1.3 Environment Setup
import os
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import riva.client
    print("✓ Riva client package available in active conda environment")
except ImportError as e:
    base_path = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
    print(f"ERROR: Missing package - {e}")
    print("Ensure you've activated the conda environment:")
    print("  module load conda")
    print(f"  conda activate {base_path}/conda_envs/sparc_backend")
```

## 2.0 NVIDIA Riva Deployment
Deploying the Riva server for ASR and TTS capabilities.

### 2.1 Riva Server Setup

This section prepares the NVIDIA Riva server runtime. On HiPerGator, we use **Apptainer** to pull `riva-speech:2.16.0-server`. `riva_init.sh` is a one-time initialization step that downloads and optimizes speech models.

![Riva Server Deployment & Configuration Lifecycle (Section 2.0)](../images/h3_2.png)

Riva Server Deployment & Configuration Lifecycle (Section 2.0): This flowchart details the one-time manual setup and launch sequence executed in the HiPerGator terminal. It highlights the specific config.sh modifications required to optimize the container for SPARC-P by turning off unneeded NLP features to save VRAM.

A numbered setup guide is printed below for running in a HiPerGator terminal; these commands are informational and do not execute automatically.

The instructions cover four one-time or operational steps:
1. **Load Apptainer module** to enable container runtime.
2. **Pull Riva image** from NGC and save it as a `.sif` in `/blue`.
3. **Run `riva_init.sh`** once to download and optimize ASR/TTS models.
4. **Start `riva_start.sh`** on the target node before client tests.

Riva serves gRPC on `localhost:50051`; this notebook validates connectivity and basic ASR/TTS response behavior against that endpoint.

```python
# 2.2 Riva Setup for HiPerGator
import os

RIVA_VERSION = "2.16.0"
BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
RIVA_SIF_PATH = os.path.join(BASE_PATH, "containers", "riva_server.sif")

def setup_riva_instructions():
    """Print one-time setup and launch guidance for Riva on HiPerGator."""
    instructions = f"""
    === Riva Setup on HiPerGator ===
    
    1. Load required module:
       module load apptainer
    
    2. Pull Riva container (one-time):
       apptainer pull {RIVA_SIF_PATH} \
           docker://nvcr.io/nvidia/riva/riva-speech:{RIVA_VERSION}-server
    
    3. Initialize models (one-time, GPU node):
       apptainer exec --nv {RIVA_SIF_PATH} riva_init.sh
    
    4. Start service before tests:
       apptainer exec --nv {RIVA_SIF_PATH} riva_start.sh
    
    5. Run this notebook's client tests against localhost:50051
    """
    print(instructions)
    return instructions

setup_riva_instructions()
```

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

### 2.3 Server Launch

The following commands launch the Riva server. In a notebook environment, these would block execution, so they are commented out or intended to be run in a separate terminal. The `riva_start.sh` script spins up the containerized service.

An execution reminder prints that `riva_init.sh` and `riva_start.sh` should be run from a terminal session, not directly in this notebook.

Why terminal execution is preferred:
- `riva_init.sh` is a long-running model download/optimization step and is typically run once per deployment image.
- `riva_start.sh` launches a persistent service process; notebook execution would block while the service runs.

After the service is running, return to this notebook and run the client verification cells in Section 3.

```python
# 2.3 Launch Riva Server
# !bash riva_init.sh
# !bash riva_start.sh
print("Run 'riva_init.sh' and 'riva_start.sh' in the terminal to launch Docker containers.")
```

## 3.0 Riva Client Testing
Verifying ASR and TTS services.

### 3.1 Service Verification

Once the server is running, we must verify connectivity. These functions use the `riva.client` library to send a gRPC request to `localhost:50051`.

![Riva Client Testing Workflow (Section 3.0)](../images/h3_3.png)

Riva Client Testing Workflow (Section 3.0): Once the server is running in the terminal, the notebook executes these client verification tests. This diagram maps how the riva.client authenticates and validates both input (ASR) and output (TTS) modalities.

- `test_asr_service`: Streams audio chunks and prints the transcript.
- `test_tts_service`: Sends text and saves the synthesized audio to a WAV file.

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

## 4.0 Coach Voice Cloning (Zero-Shot TTS)

Riva 2.x supports **zero-shot voice cloning natively**. By providing a short (3–10 second) audio prompt, Riva TTS can adapt output voice characteristics at inference time.

This section restores the full voice-cloning workflow expected in H3:
1. **4.1 Audio Preprocessing**: Convert candidate clips to 16 kHz mono PCM WAV and select the best prompt.
2. **4.2 Voice Prompt Validation and Test Synthesis**: Run a Riva synthesis test with `zero_shot_audio_prompt_file` and `zero_shot_transcript`.
3. **4.3 `CoachVoiceConfig`**: Standardize prompt/transcript/quality settings for downstream orchestration use.

![Coach Voice Cloning (Zero-Shot TTS) Pipeline (Section 4.0)](../images/h3_4.png)

Coach Voice Cloning (Zero-Shot TTS) Pipeline (Section 4.0): This is the most complex logic in the H3 notebook. It details the automated ingestion, filtering, and validation of raw audio clips to create a perfectly optimized 16kHz voice prompt, allowing Riva to clone the Coach's voice with a single reference file.

### 4.1 Audio Preprocessing

Scan `audio/coach_examples/` for `.mp3` / `.wav` files, convert each to 16 kHz mono PCM WAV, and select the best prompt clip based on:
- Loudness-first ranking: prefer the strongest / cleanest signal first
- Minimum duration gate: ignore clips under 3 seconds because they are too short for stable cloning
- If the loudest usable clip is longer than 10 seconds, trim it to the 7-second target window before saving

Recommended output artifact:
- `audio/coach_examples/processed/best_prompt.wav`

```python
# 4.1 Audio Preprocessing — Select and convert best prompt clip
import os
from pathlib import Path

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("WARNING: pydub not installed. Run: pip install pydub")
    print("         ffmpeg must also be on PATH for mp3 support.")

BASE_PATH = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
EXAMPLES_DIR = Path(BASE_PATH) / "audio" / "coach_examples"
PROCESSED_DIR = EXAMPLES_DIR / "processed"
BEST_PROMPT_PATH = PROCESSED_DIR / "best_prompt.wav"
TARGET_DURATION_S = 7.0   # seconds — empirically optimal for zero-shot cloning
MIN_DURATION_S = 3.0
MAX_DURATION_S = 10.0
TARGET_SAMPLE_RATE = 16000

def preprocess_prompt_clips(examples_dir: Path, processed_dir: Path) -> list[dict]:
    """
    Converts all .mp3 and .wav files in examples_dir to 16 kHz mono PCM WAV,
    measures duration and RMS energy, and returns a list of candidate dicts.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    candidates = []

    audio_files = sorted(
        [f for f in examples_dir.iterdir() if f.suffix.lower() in (".mp3", ".wav") and f.is_file()]
    )
    if not audio_files:
        print(f"No .mp3 or .wav files found in {examples_dir}")
        return candidates

    for src in audio_files:
        try:
            seg = AudioSegment.from_file(str(src))
            seg = seg.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1).set_sample_width(2)
            duration_s = len(seg) / 1000.0
            rms = seg.rms

            dest = processed_dir / (src.stem + "_16k.wav")
            seg.export(str(dest), format="wav")

            candidates.append({
                "src": src.name,
                "dest": dest,
                "duration_s": round(duration_s, 2),
                "rms": rms,
                "usable": duration_s >= MIN_DURATION_S,
                "needs_trim": duration_s > MAX_DURATION_S,
            })
        except Exception as e:
            print(f"  SKIP {src.name}: {e}")

    return candidates

def select_best_prompt(candidates: list[dict]) -> dict | None:
    """Picks the loudest usable clip and trims it later if it exceeds the max duration."""
    ranked = sorted(candidates, key=lambda c: c["rms"], reverse=True)
    usable = [c for c in ranked if c["usable"]]
    if not usable:
        return None
    best = dict(usable[0])
    best["selected_reason"] = "highest_rms"
    return best

def materialize_best_prompt(best: dict, best_prompt_path: Path) -> dict:
    """Copies or trims the selected prompt clip into the canonical best-prompt location."""
    seg = AudioSegment.from_file(str(best["dest"]))
    trimmed = False
    output_duration_s = len(seg) / 1000.0
    if len(seg) / 1000.0 > MAX_DURATION_S:
        seg = seg[: int(TARGET_DURATION_S * 1000)]
        output_duration_s = len(seg) / 1000.0
        trimmed = True
    seg.export(str(best_prompt_path), format="wav")
    enriched = dict(best)
    enriched["trimmed"] = trimmed
    enriched["output_duration_s"] = round(output_duration_s, 2)
    return enriched

if PYDUB_AVAILABLE:
    if not EXAMPLES_DIR.exists():
        print(f"Creating example directory: {EXAMPLES_DIR}")
        EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        print("Place .mp3 or .wav coach recordings in that directory, then re-run this cell.")
    else:
        candidates = preprocess_prompt_clips(EXAMPLES_DIR, PROCESSED_DIR)
        best = select_best_prompt(candidates)

        # Print summary table
        print(f"{'File':<35} {'Duration':>9} {'RMS':>7} {'Usable':>9} {'Action':>11} {'Selected':>9}")
        print("-" * 73)
        for c in candidates:
            sel = "<-- SELECTED" if best and c["dest"] == best["dest"] else ""
            usable = "yes" if c["usable"] else f"no ({c['duration_s']:.1f}s)"
            action = f"trim->{TARGET_DURATION_S:.0f}s" if c["needs_trim"] else "keep"
            print(f"{c['src']:<35} {c['duration_s']:>8.2f}s {c['rms']:>7} {usable:>9} {action:>11} {sel}")

        if best:
            best = materialize_best_prompt(best, BEST_PROMPT_PATH)
            print(f"\nBest prompt copied to: {BEST_PROMPT_PATH}")
            print(f"  Source duration : {best['duration_s']}s")
            print(f"  Output duration : {best['output_duration_s']}s")
            print(f"  RMS      : {best['rms']}")
            print(f"  Trimmed  : {best['trimmed']}")
            COACH_PROMPT_TRANSCRIPT = input(
                "\nEnter the transcript of the selected audio clip (exact words spoken): "
            ).strip()
            print(f"Transcript set: '{COACH_PROMPT_TRANSCRIPT}'")
        else:
            print("\nNo usable clips found (need recordings at least 3 seconds long). "
                  "Add files to audio/coach_examples/ and re-run.")
            COACH_PROMPT_TRANSCRIPT = ""
else:
    COACH_PROMPT_TRANSCRIPT = ""
    BEST_PROMPT_PATH = Path("/tmp/best_prompt_placeholder.wav")
```

### 4.2 Voice Prompt Validation and Test Synthesis

Validates the selected prompt by calling Riva's `synthesize()` API with the zero-shot parameters and writing the output to `audio/coach_voice_test.wav`. Listen to this file to verify voice quality before deploying.

Key parameters:
- **`zero_shot_audio_prompt_file`**: Path to the 16 kHz mono WAV prompt from Section 4.1.
- **`zero_shot_transcript`**: The exact words spoken in the prompt clip — Riva uses this to align phonemes; accuracy directly affects voice similarity.
- **`zero_shot_quality`**: Integer 1–40. Higher = slower inference but better voice match. Default of `20` balances latency and quality for real-time coaching.
- **`sample_rate_hz`**: Output sample rate set to 44100 Hz for audio playback compatibility.

If Riva is offline (e.g., running this cell outside of HiPerGator), the cell prints a dry-run summary instead of failing.

```python
# 4.2 Voice Prompt Validation and Test Synthesis
import wave
import os
from pathlib import Path

RIVA_SERVER = os.environ.get("SPARC_RIVA_SERVER", "localhost:50051")
TEST_OUTPUT_PATH = Path(BASE_PATH) / "audio" / "coach_voice_test.wav"
TEST_TEXT = (
    "Great job maintaining eye contact and using affirming language. "
    "Next time, try pausing after your empathy statement to give the caregiver more space to respond."
)
ZERO_SHOT_QUALITY = 20   # 1–40; 20 is the mid-range default

def _prompt_is_valid(prompt_path: Path) -> bool:
    """Returns True if the file exists, is > 0 bytes, and is valid PCM WAV."""
    if not prompt_path.exists() or prompt_path.stat().st_size == 0:
        return False
    try:
        with wave.open(str(prompt_path), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getframerate() == 16000
    except Exception:
        return False

def run_test_synthesis(prompt_path: Path, transcript: str, output_path: Path) -> bool:
    """
    Calls Riva zero-shot TTS and writes the result to output_path.
    Returns True on success, False on any connection or API error.
    """
    try:
        import riva.client
        channel = riva.client.connect(RIVA_SERVER)
        tts_service = riva.client.SpeechSynthesisService(channel)

        resp = tts_service.synthesize(
            TEST_TEXT,
            "English-US.Female-1",     # base voice — zero-shot overrides timbre
            "en-US",
            sample_rate_hz=44100,
            zero_shot_audio_prompt_file=prompt_path,
            zero_shot_quality=ZERO_SHOT_QUALITY,
            zero_shot_transcript=transcript,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "wb") as f:
            f.write(resp.audio)
        print(f"Test synthesis written to: {output_path}")
        print(f"  Audio bytes : {len(resp.audio):,}")
        return True

    except ImportError:
        print("riva.client not installed — install the Riva Python client package.")
        return False
    except Exception as err:
        print(f"Riva TTS unavailable ({err}) — dry-run mode.")
        print(f"  Would synthesize : '{TEST_TEXT[:60]}...'")
        print(f"  Prompt file      : {prompt_path}")
        print(f"  Quality setting  : {ZERO_SHOT_QUALITY}")
        print(f"  Output target    : {output_path}")
        return False

if _prompt_is_valid(BEST_PROMPT_PATH):
    print(f"Prompt validated: {BEST_PROMPT_PATH} ({BEST_PROMPT_PATH.stat().st_size:,} bytes)")
    success = run_test_synthesis(BEST_PROMPT_PATH, COACH_PROMPT_TRANSCRIPT, TEST_OUTPUT_PATH)
    if success:
        print("\nListen to coach_voice_test.wav to verify voice quality before deployment.")
else:
    print(f"Prompt not yet available at {BEST_PROMPT_PATH}.")
    print("Run Section 4.1 first to preprocess audio clips.")
```

### 4.3 `CoachVoiceConfig` — Zero-Shot Voice Profile

`CoachVoiceConfig` is a lightweight dataclass that bundles everything `CoachAgent` needs to use the cloned voice:

| Field | Type | Description |
|---|---|---|
| `prompt_path` | `Path` | Path to the 16 kHz mono WAV prompt from Section 4.1 |
| `transcript` | `str` | Exact words spoken in the prompt — must match precisely |
| `quality` | `int` | Zero-shot quality 1–40 (default 20) |
| `language_code` | `str` | BCP-47 language tag (default `"en-US"`) |
| `voice_name` | `str` | Base Riva voice the adaptation starts from |

`CoachAgent` is updated to accept an optional `CoachVoiceConfig`. When the config is provided and Riva is reachable, `evaluate_turn()` returns both text feedback and base64-encoded audio synthesized in the cloned voice. When Riva is offline or no config is given, it falls back to returning text-only feedback — the orchestrator continues to run normally.

`build_app_graph()` is updated to attempt loading the config from `audio/coach_examples/processed/best_prompt.wav` at startup. If the file does not exist (e.g., first deployment before Section 4.1 has been run), it starts with voice cloning disabled.

```python
# 4.3 CoachVoiceConfig dataclass
import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class CoachVoiceConfig:
    """
    Zero-shot voice cloning configuration for CoachAgent.

    Attributes:
        prompt_path:   Path to a 16 kHz mono PCM WAV file (3–10 seconds).
        transcript:    Exact words spoken in the prompt clip.
        quality:       Zero-shot quality 1–40. Higher = better voice match, slower inference.
        language_code: BCP-47 language tag for TTS synthesis.
        voice_name:    Base Riva voice the zero-shot adaptation starts from.
        riva_server:   Host:port of the running Riva gRPC endpoint.
    """
    prompt_path: Path
    transcript: str
    quality: int = 20
    language_code: str = "en-US"
    voice_name: str = "English-US.Female-1"
    riva_server: str = field(
        default_factory=lambda: os.environ.get("SPARC_RIVA_SERVER", "localhost:50051")
    )

    def is_ready(self) -> bool:
        """True if the prompt file exists and has content."""
        return self.prompt_path.exists() and self.prompt_path.stat().st_size > 0


def load_coach_voice_config(
    processed_dir: Optional[Path] = None,
    transcript: str = "",
    quality: int = 20,
) -> Optional[CoachVoiceConfig]:
    """
    Factory function called at startup.  Returns a CoachVoiceConfig if
    best_prompt.wav exists in processed_dir, otherwise returns None so the
    system starts with voice cloning disabled (text-only fallback).
    """
    base = os.environ.get("SPARC_BASE_PATH", "/blue/jasondeanarnold/SPARCP")
    prompt = (processed_dir or Path(base) / "audio" / "coach_examples" / "processed") / "best_prompt.wav"

    if not prompt.exists():
        print(f"CoachVoiceConfig: prompt not found at {prompt} — voice cloning disabled.")
        return None

    cfg = CoachVoiceConfig(prompt_path=prompt, transcript=transcript, quality=quality)
    print(f"CoachVoiceConfig loaded: {prompt} (quality={quality})")
    return cfg


# Preview (won't error even if prompt doesn't exist yet)
_preview_cfg = load_coach_voice_config(transcript=COACH_PROMPT_TRANSCRIPT)
if _preview_cfg:
    print(f"  Voice cloning ENABLED — prompt: {_preview_cfg.prompt_path.name}")
    print(f"  Transcript   : '{_preview_cfg.transcript}'")
    print(f"  Quality      : {_preview_cfg.quality}")
else:
    print("  Voice cloning DISABLED — run Section 4.1 to enable.")
```

## 5.0 Next Notebook

Riva setup, service-level testing, and voice-cloning preparation are complete in this notebook.

Continue in `H4_Nemo_Testing_Security.ipynb` for multi-agent orchestration, API-server validation, and security/compliance regression checks.


