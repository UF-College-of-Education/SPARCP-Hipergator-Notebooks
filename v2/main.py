import asyncio
import base64
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import firebase_admin
import riva.client
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from firebase_admin import credentials, firestore
from nemoguardrails import LLMRails, RailsConfig
from peft import PeftModel
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_BASE_PATH = os.getenv("SPARC_MODEL_BASE_PATH", "/pubapps/SPARCP/models")
RIVA_SERVER = os.getenv("SPARC_RIVA_SERVER", "localhost:50051")
FIREBASE_CREDS = os.getenv("SPARC_FIREBASE_CREDS", "/pubapps/SPARCP/config/firebase-credentials.json")
GUARDRAILS_DIR = os.getenv("SPARC_GUARDRAILS_DIR", os.path.join(os.path.dirname(__file__), "guardrails"))

API_AUTH_ENABLED = os.getenv("SPARC_API_AUTH_ENABLED", "false").strip().lower() == "true"
API_KEY = os.getenv("SPARC_API_KEY", "")
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "SPARC_CORS_ALLOWED_ORIGINS",
        "https://hpvcommunicationtraining.com,https://hpvcommunicationtraining.org",
    ).split(",")
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
SPARC_AUDIO_CACHE_DIR = os.getenv(
    "SPARC_AUDIO_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "sparc_tts_audio"),
)
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("SPARC_TIMEOUT_CIRCUIT_THRESHOLD", "3"))
CIRCUIT_BREAKER_RESET_SECONDS = float(os.getenv("SPARC_TIMEOUT_CIRCUIT_RESET_SECONDS", "30"))
DEFAULT_ANIMATION_EMOTION = os.getenv("SPARC_DEFAULT_ANIMATION_EMOTION", "neutral")
DEFAULT_ANIMATION_GESTURE = os.getenv("SPARC_DEFAULT_ANIMATION_GESTURE", "speaking")
DEFAULT_LANGUAGE_CODE = os.getenv("SPARC_ASR_LANGUAGE_CODE", "en-US")
DEFAULT_SAMPLE_RATE_HZ = int(os.getenv("SPARC_ASR_SAMPLE_RATE_HZ", "16000"))
DEFAULT_CHANNEL_COUNT = int(os.getenv("SPARC_ASR_CHANNEL_COUNT", "1"))
DEFAULT_MAX_ALTERNATIVES = int(os.getenv("SPARC_ASR_MAX_ALTERNATIVES", "1"))
DEFAULT_AUTOMATIC_PUNCTUATION = os.getenv("SPARC_ASR_AUTO_PUNCT", "true").strip().lower() == "true"
DEFAULT_PROFANITY_FILTER = os.getenv("SPARC_ASR_PROFANITY_FILTER", "false").strip().lower() == "true"
DEFAULT_INTERIM_RESULTS = os.getenv("SPARC_ASR_INTERIM_RESULTS", "true").strip().lower() == "true"

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

try:
    presidio_analyzer = AnalyzerEngine()
    presidio_anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except Exception as presidio_init_error:
    presidio_analyzer = None
    presidio_anonymizer = None
    PRESIDIO_AVAILABLE = False
    logger.warning(
        "Presidio initialization failed; using fail-closed redaction placeholders: %s",
        presidio_init_error,
    )


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
        logger.exception(
            "Guardrails initialization failed: %s",
            sanitize_for_storage(str(guardrails_error)),
        )


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
        logger.exception(
            "Input guardrails failed: %s",
            sanitize_for_storage(str(guardrails_error)),
        )
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
        logger.exception(
            "Output guardrails failed: %s",
            sanitize_for_storage(str(guardrails_error)),
        )
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
        logger.warning(
            "Riva client initialization failed: %s",
            sanitize_for_storage(str(riva_init_error)),
        )


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
            logger.exception("Streaming ASR session failed: %s", sanitize_for_storage(str(asr_error)))
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
        "websocket_path": "/ws/audio",
        "default_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
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
            fallback_text = "I'm temporarily unable to generate a response right now. Please try again shortly."
            return ChatResponse(
                response_text=fallback_text,
                caregiver_text=fallback_text,
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback="Primary model temporarily unavailable.",
                coach_feedback_meta={
                    "safe": True,
                    "reason": "inference_circuit_open",
                    "summary": "Primary model temporarily unavailable.",
                },
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
            logger.warning(
                "Primary inference timed out after %.1fs%s",
                LLM_TIMEOUT_SECONDS,
                "; circuit opened" if circuit_opened else "",
            )
            fallback_text = "I'm temporarily unable to generate a response right now. Please try again shortly."
            return ChatResponse(
                response_text=fallback_text,
                caregiver_text=fallback_text,
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback="Primary model timeout fallback.",
                coach_feedback_meta={
                    "safe": True,
                    "reason": "inference_timeout",
                    "summary": "Primary model timeout fallback.",
                },
                active_agent=primary_adapter,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response_text = decoded.split("Assistant:")[-1].strip() or "I'm here to help with HPV vaccine communication practice."

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
            logger.warning(
                "Riva TTS timed out after %.1fs%s",
                TTS_TIMEOUT_SECONDS,
                "; circuit opened" if circuit_opened else "",
            )
        except Exception as riva_error:
            logger.warning("Riva TTS unavailable: %s", sanitize_for_storage(str(riva_error)))

        legacy_audio_b64 = build_legacy_audio_b64(audio_bytes, request.include_legacy_audio_b64)
        animation_cues = default_animation_cues()
        sanitized_user_message = sanitize_for_storage(normalized_user_message)
        sanitized_response_text = sanitize_for_storage(response_text)
        session_state["last_user_message"] = sanitized_user_message
        session_state["last_response"] = sanitized_response_text
        session_state["mode"] = primary_adapter
        session_state["phi_redaction"] = "presidio"
        session_state["phi_redaction_applied"] = True
        session_ref.set(session_state, merge=True)

        return ChatResponse(
            response_text=response_text,
            caregiver_text=response_text,
            audio_url=audio_url,
            caregiver_audio_b64=legacy_audio_b64,
            caregiver_animation_cues=animation_cues,
            coach_feedback=coach_feedback_text[:500],
            coach_feedback_meta={
                "safe": output_guard["allowed"],
                "reason": coach_feedback_reason,
                "summary": coach_feedback_text[:500],
            },
            active_agent=primary_adapter,
        )
    except Exception as e:
        logger.exception(
            "/v1/chat failed after sanitization path: %s",
            sanitize_for_storage(str(e)),
        )
        raise HTTPException(status_code=500, detail="Internal server error")
