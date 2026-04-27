import asyncio
import base64
import copy
import difflib
import json
import re
import time
import tempfile
import uuid
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Transformers 5.x pre-allocates GPU memory in caching_allocator_warmup before weight load.
# On some PubApp stacks (user systemd + certain drivers) torch.empty(..., device=cuda) there
# raises RuntimeError: CUDA driver error: operation not supported, while interactive shells work.
# Auto-skip when launched under systemd (INVOCATION_ID) unless SPARC_SKIP_CUDA_ALLOC_WARMUP=0.
def _should_skip_cuda_alloc_warmup() -> bool:
    v = os.getenv("SPARC_SKIP_CUDA_ALLOC_WARMUP", "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return "INVOCATION_ID" in os.environ


if _should_skip_cuda_alloc_warmup():
    import transformers.modeling_utils as _hf_modeling_utils

    _hf_modeling_utils.caching_allocator_warmup = lambda *_a, **_kw: None  # type: ignore[method-assign]

from peft import PeftModel

try:
    import riva.client as riva
except ImportError:
    riva = None  # type: ignore[assignment]
from nemoguardrails import LLMRails, RailsConfig
import firebase_admin
from firebase_admin import credentials, firestore
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions as chroma_embedding_functions

MODEL_BASE_PATH = os.getenv("SPARC_MODEL_BASE_PATH", "/pubapps/SPARCP/models")
# Parent of `.../models` → `.../SPARCP` for per-user PubApp trees (jasondeanarnold/SPARCP/...).
_mbp = MODEL_BASE_PATH.rstrip(os.sep)
SPARC_ROOT = os.path.dirname(_mbp) if os.path.basename(_mbp) == "models" else _mbp

RIVA_SERVER = os.getenv("SPARC_RIVA_SERVER", "localhost:50051")
# TTS backend: `riva` (gRPC Riva server) or `kokoro` (local Kokoro-82M, matches H5c notebook).
_TTS_BACKEND = os.getenv("SPARC_TTS_BACKEND", "riva").strip().lower()
TTS_BACKEND: str = _TTS_BACKEND if _TTS_BACKEND in ("riva", "kokoro") else "riva"

if os.getenv("SPARC_FIREBASE_CREDS"):
    FIREBASE_CREDS = os.environ["SPARC_FIREBASE_CREDS"]
else:
    FIREBASE_CREDS = os.path.join(SPARC_ROOT, "config", "firebase-credentials.json")

if os.getenv("SPARC_GUARDRAILS_DIR"):
    GUARDRAILS_DIR = os.environ["SPARC_GUARDRAILS_DIR"]
else:
    GUARDRAILS_DIR = os.path.join(SPARC_ROOT, "guardrails")

# Prefer rsync'd Llama snapshot under SPARC_MODEL_BASE_PATH to avoid gated Hub downloads on PubApp.
_LOCAL_LLAMA_DIR = os.path.join(MODEL_BASE_PATH, "meta_llama", "Llama3.1-8B-Instruct")
if os.getenv("SPARC_BASE_MODEL"):
    BASE_MODEL_NAME = os.environ["SPARC_BASE_MODEL"]
elif os.path.isdir(_LOCAL_LLAMA_DIR) and os.path.isfile(os.path.join(_LOCAL_LLAMA_DIR, "config.json")):
    BASE_MODEL_NAME = _LOCAL_LLAMA_DIR
else:
    BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

if os.getenv("SPARC_RAG_PERSIST_DIR"):
    RAG_PERSIST_DIR = os.environ["SPARC_RAG_PERSIST_DIR"]
else:
    RAG_PERSIST_DIR = os.path.join(SPARC_ROOT, "rag", "chroma")
# Collection name must match the existing on-disk index produced by the
# HiPerGator training notebooks (H1b). The canonical collection shipped
# from `/blue/.../trained_models/vector_db/sparc_training_markdown_kb/`
# is `sparc_training_markdown_kb`; override at runtime if you rebuild.
RAG_COLLECTION = os.getenv("SPARC_RAG_COLLECTION", "sparc_training_markdown_kb")
RAG_EMBEDDING_MODEL = os.getenv("SPARC_RAG_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
RAG_TOP_K = int(os.getenv("SPARC_RAG_TOP_K", "4"))
RAG_MIN_CHARS = int(os.getenv("SPARC_RAG_MIN_CHARS", "8"))
RAG_CONTEXT_MAX_CHARS = int(os.getenv("SPARC_RAG_CONTEXT_MAX_CHARS", "5000"))
ENABLE_RAG_IN_CHAT = os.getenv("SPARC_ENABLE_RAG_CHAT", "false").strip().lower() == "true"
SOFT_GUARDRAILS_FOR_CAREGIVER = os.getenv("SPARC_SOFT_GUARDRAILS_FOR_CAREGIVER", "true").strip().lower() == "true"

API_AUTH_ENABLED = os.getenv("SPARC_API_AUTH_ENABLED", "false").strip().lower() == "true"
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
ENABLE_GUARDRAILS = os.getenv("SPARC_ENABLE_GUARDRAILS", "false").strip().lower() == "true"
USE_ADAPTERS = os.getenv("SPARC_USE_ADAPTERS", "true").strip().lower() == "true"
VERBOSE_CHAT_LOGS = os.getenv("SPARC_VERBOSE_CHAT_LOGS", "true").strip().lower() == "true"
VERBOSE_CHAT_LOG_PREVIEW_CHARS = int(os.getenv("SPARC_VERBOSE_CHAT_LOG_PREVIEW_CHARS", "500"))
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
DEFAULT_LANGUAGE_CODE = os.getenv("SPARC_ASR_LANGUAGE_CODE", "en-US")
DEFAULT_SAMPLE_RATE_HZ = int(os.getenv("SPARC_ASR_SAMPLE_RATE_HZ", "16000"))
DEFAULT_CHANNEL_COUNT = int(os.getenv("SPARC_ASR_CHANNEL_COUNT", "1"))
DEFAULT_MAX_ALTERNATIVES = int(os.getenv("SPARC_ASR_MAX_ALTERNATIVES", "1"))
DEFAULT_AUTOMATIC_PUNCTUATION = os.getenv("SPARC_ASR_AUTO_PUNCT", "true").strip().lower() == "true"
DEFAULT_PROFANITY_FILTER = os.getenv("SPARC_ASR_PROFANITY_FILTER", "false").strip().lower() == "true"
DEFAULT_INTERIM_RESULTS = os.getenv("SPARC_ASR_INTERIM_RESULTS", "true").strip().lower() == "true"
CHAT_MAX_INPUT_TOKENS = int(os.getenv("SPARC_CHAT_MAX_INPUT_TOKENS", "6000"))
COACH_MAX_INPUT_TOKENS = int(os.getenv("SPARC_COACH_MAX_INPUT_TOKENS", "6000"))

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
logged_backend_env_sessions: set[str] = set()
uid_to_timestamp_session: Dict[str, str] = {}
TIMESTAMPED_SESSION_RE = re.compile(r"^\d{8}_\d{6}_[A-Za-z0-9_-]+$")

logger = logging.getLogger("sparc_backend")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


guardrails_engine = None
GUARDRAILS_REFUSAL = "I can only discuss topics related to HPV vaccination and clinical communication training."


def _build_guardrails_langchain_llm():
    """LangChain LLM wrapping the already-loaded Llama + PEFT stack for NeMo Guardrails.

    NeMo's YAML `engine: self_hosted` maps to LangChain `SelfHostedPipeline` (Runhouse +
    pickle), not an in-process HF model — so we inject `HuggingFacePipeline` instead.
    """
    if adapter_model is None or tokenizer is None:
        return None
    try:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline as transformers_pipeline
    except ImportError as import_err:
        logger.warning("Guardrails LLM unavailable (langchain/transformers): %s", import_err)
        return None

    tok = tokenizer
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    max_new = int(os.getenv("SPARC_GUARDRAILS_MAX_NEW_TOKENS", "64").strip() or "64")
    pipe = transformers_pipeline(
        "text-generation",
        model=adapter_model,
        tokenizer=tok,
        max_new_tokens=max_new,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tok.pad_token_id,
    )
    return HuggingFacePipeline(pipeline=pipe)


def _load_guardrails_engine(path: str):
    rails_config = RailsConfig.from_path(path)
    guard_llm = _build_guardrails_langchain_llm()
    if guard_llm is None:
        raise RuntimeError("Guardrails LLM could not be built (models not loaded?)")
    return LLMRails(rails_config, llm=guard_llm)


def load_guardrails_runtime() -> None:
    global guardrails_engine

    guardrails_engine = None

    try:
        guardrails_engine = _load_guardrails_engine(GUARDRAILS_DIR)
        logger.info("Guardrails runtime loaded from %s (in-process Llama+PEFT)", GUARDRAILS_DIR)
    except Exception as guardrails_error:
        logger.exception("Guardrails initialization failed: %s", guardrails_error)


async def _run_guardrails(engine, text: str) -> str:
    if engine is None:
        raise RuntimeError("Guardrails runtime not initialized")
    messages = [{"role": "user", "content": text}]
    
    # CRITICAL: Acquire the GPU lock before NeMo Guardrails triggers the LLM
    async with inference_lock:
        if model_supports_adapter_switching():
            adapter_model.set_adapter("caregiver")
        if hasattr(engine, "generate_async"):
            result = await engine.generate_async(messages=messages)
        else:
            result = await asyncio.to_thread(engine.generate, messages=messages)
            
    if isinstance(result, dict):
        return str(result.get("content", result))
    return str(result)


def _guardrails_context_for_agent(agent_name: str):
    role = (agent_name or "caregiver").strip().lower()
    if role == "coach":
        return {
            "engine": guardrails_engine,
            "input_prefix": "coach_input",
            "output_prefix": "coach_output",
        }
    return {
        "engine": guardrails_engine,
        "input_prefix": "input",
        "output_prefix": "output",
    }


async def enforce_guardrails_input(user_text: str, agent_name: str) -> Dict[str, Any]:
    if not ENABLE_GUARDRAILS:
        return {"allowed": True, "text": user_text, "reason": "guardrails_disabled"}
    context = _guardrails_context_for_agent(agent_name)
    prefix = context["input_prefix"]
    if not user_text or not user_text.strip():
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_empty"}

    # Coach grading payloads intentionally include rubric templates and JSON directives.
    # Keep a single guardrails profile, but bypass topical rails for coach traffic.
    if (agent_name or "").strip().lower() == "coach":
        return {"allowed": True, "text": user_text, "reason": "coach_input_bypassed"}

    try:
        rails_output = await _run_guardrails(context["engine"], user_text)
        blocked = GUARDRAILS_REFUSAL.lower() in rails_output.lower()
        if blocked:
            return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_rails_blocked"}
        return {"allowed": True, "text": user_text, "reason": f"{prefix}_rails_allowed"}
    except Exception as guardrails_error:
        logger.exception("Input guardrails failed (%s): %s", agent_name, guardrails_error)
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_rails_error"}


async def enforce_guardrails_output(output_text: str, agent_name: str) -> Dict[str, Any]:
    if not ENABLE_GUARDRAILS:
        return {"allowed": True, "text": output_text, "reason": "guardrails_disabled"}
    context = _guardrails_context_for_agent(agent_name)
    prefix = context["output_prefix"]
    if not output_text or not output_text.strip():
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_empty"}

    # Coach output is schema-validated downstream; skip topical output rails to
    # avoid false positives on strict JSON grading responses.
    if (agent_name or "").strip().lower() == "coach":
        return {"allowed": True, "text": output_text, "reason": "coach_output_bypassed"}

    try:
        rails_output = await _run_guardrails(context["engine"], output_text)
        blocked = GUARDRAILS_REFUSAL.lower() in rails_output.lower()
        if blocked:
            return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_rails_blocked"}
        return {"allowed": True, "text": output_text, "reason": f"{prefix}_rails_allowed"}
    except Exception as guardrails_error:
        logger.exception("Output guardrails failed (%s): %s", agent_name, guardrails_error)
        return {"allowed": False, "text": GUARDRAILS_REFUSAL, "reason": f"{prefix}_rails_error"}


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
# Tracks which adapters were actually loaded at startup so mode resolution
# can gracefully fall back to caregiver when a requested adapter (e.g.
# supervisor) has no trained weights on disk.
loaded_adapters: set = set()
# Caregiver + Coach are trained on dedicated datasets
# (synthetic-3000 for Anne/Maya; C-LEAR coach train.jsonl for the coach).
# A dedicated Supervisor adapter has no training jsonl yet — the entry is
# kept for forward compatibility but is loaded optionally.
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
REQUIRED_ADAPTERS = {"caregiver"}
OPTIONAL_ADAPTERS = {"coach", "supervisor"}
riva_auth = None
riva_asr_service = None
riva_tts_service = None
kokoro_pipelines: Dict[str, Any] = {}
kokoro_ready = False
chroma_client = None
chroma_collection = None
inference_lock = asyncio.Lock()
timeout_state_lock = asyncio.Lock()
audio_cache_lock = asyncio.Lock()
audio_cache_index: Dict[str, Dict[str, Any]] = {}
timeout_failures = {
    "primary_inference": 0,
    "coach_inference": 0,
    "tts": 0,
}
circuit_open_until = {
    "primary_inference": 0.0,
    "coach_inference": 0.0,
    "tts": 0.0,
}


def generate_tokens_sync(model, **generate_kwargs):
    with torch.inference_mode():
        return model.generate(**generate_kwargs)


def init_riva_clients() -> None:
    global riva_auth, riva_asr_service, riva_tts_service
    if riva is None:
        riva_auth = None
        riva_asr_service = None
        riva_tts_service = None
        logger.warning("nvidia-riva-client is not installed; Riva ASR/TTS disabled")
        return
    try:
        riva_auth = riva.Auth(uri=RIVA_SERVER)
        riva_asr_service = riva.ASRService(riva_auth)
        riva_tts_service = riva.SpeechSynthesisService(riva_auth)
        logger.info("Riva clients initialized for reuse at startup")
    except Exception as riva_init_error:
        riva_auth = None
        riva_asr_service = None
        riva_tts_service = None
        logger.warning("Riva client initialization failed: %s", riva_init_error)


def init_kokoro() -> None:
    """Load Kokoro TTS (H5c notebook pattern); weights cache under HF_HOME / ~/.cache/huggingface."""
    global kokoro_ready, kokoro_pipelines
    try:
        from kokoro import KPipeline

        kokoro_pipelines["a"] = KPipeline(lang_code="a")
        kokoro_ready = True
        logger.info("Kokoro TTS initialized (lang_code=a, sample_rate=24000)")
    except Exception as kokoro_error:
        kokoro_pipelines.clear()
        kokoro_ready = False
        logger.warning("Kokoro TTS initialization failed: %s", kokoro_error)


KOKORO_VOICE_ALIASES = {
    # Map legacy Riva-style names from Unity / config → Kokoro voice ids (see hexgrad/Kokoro-82M)
    "English-US.Female-1": "af_heart",
    "English-US.Female-2": "af_bella",
    "English-US.Male-1": "am_adam",
}


def _resolve_kokoro_voice(voice_name: Optional[str]) -> tuple[str, str]:
    """Return (lang_code, kokoro_voice_id)."""
    key = (voice_name or "").strip()
    voice_id = KOKORO_VOICE_ALIASES.get(key) or os.getenv("SPARC_KOKORO_VOICE_DEFAULT", "af_heart")
    lang = os.getenv("SPARC_KOKORO_LANG_CODE", "a")
    return lang, voice_id


def _get_kokoro_pipeline(lang_code: str):
    global kokoro_pipelines
    from kokoro import KPipeline

    if lang_code not in kokoro_pipelines:
        kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return kokoro_pipelines[lang_code]


def synthesize_kokoro_sync(text: str, voice_name: str = "English-US.Female-1") -> bytes:
    import io

    import numpy as np
    import soundfile as sf

    if not kokoro_ready:
        raise RuntimeError("Kokoro TTS is not initialized")
    lang_code, voice_id = _resolve_kokoro_voice(voice_name)
    pipeline = _get_kokoro_pipeline(lang_code)
    chunks = []
    for _, _, audio in pipeline(
        text,
        voice=voice_id,
        speed=1.0,
        split_pattern=r"\n+",
    ):
        if audio is not None:
            chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return b""
    full_audio = np.concatenate(chunks)
    buffer = io.BytesIO()
    sf.write(buffer, full_audio, 24000, format="WAV")
    return buffer.getvalue()


def synthesize_tts_sync(text: str, voice_name: str = "English-US.Female-1") -> bytes:
    if TTS_BACKEND == "kokoro":
        return synthesize_kokoro_sync(text, voice_name)
    if riva_tts_service is None:
        raise RuntimeError("Riva TTS client is not initialized")
    tts_response = riva_tts_service.synthesize(text, voice_name=voice_name)
    return tts_response.audio


def init_rag_store() -> None:
    global chroma_client, chroma_collection
    try:
        if not os.path.isdir(RAG_PERSIST_DIR):
            logger.warning(
                "RAG persist dir %s not found; RAG retrieval will be disabled",
                RAG_PERSIST_DIR,
            )
            chroma_client = None
            chroma_collection = None
            return
        chroma_client = chromadb.PersistentClient(
            path=RAG_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
        rag_embedding_function = chroma_embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=RAG_EMBEDDING_MODEL
        )
        chroma_collection = chroma_client.get_collection(
            name=RAG_COLLECTION,
            embedding_function=rag_embedding_function,
        )
        logger.info(
            "Chroma RAG store loaded: persist_dir=%s collection=%s count=%d embedder=%s",
            RAG_PERSIST_DIR,
            RAG_COLLECTION,
            chroma_collection.count(),
            RAG_EMBEDDING_MODEL,
        )
    except Exception as rag_init_error:
        chroma_client = None
        chroma_collection = None
        logger.exception("RAG store initialization failed: %s", rag_init_error)


def _retrieve_rag_context_sync(query: str) -> str:
    if chroma_collection is None or not query or len(query.strip()) < RAG_MIN_CHARS:
        return ""
    try:
        result = chroma_collection.query(query_texts=[query], n_results=RAG_TOP_K)
        documents = (result.get("documents") or [[]])[0]
        if not documents:
            return ""
        joined = "\n\n---\n\n".join(doc.strip() for doc in documents if doc)
        return joined[:RAG_CONTEXT_MAX_CHARS]
    except Exception as rag_query_error:
        logger.warning("RAG retrieval failed: %s", rag_query_error)
        return ""


async def retrieve_rag_context(query: str) -> str:
    if chroma_collection is None:
        return ""
    return await asyncio.to_thread(_retrieve_rag_context_sync, query)



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


def model_supports_adapter_switching() -> bool:
    return bool(USE_ADAPTERS and adapter_model is not None and hasattr(adapter_model, "set_adapter"))


def select_adapter_for_mode(mode: str) -> str:
    normalized = (mode or "caregiver").strip().lower().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())
    if "coach" in normalized:
        normalized = "coach"
    elif "supervisor" in normalized:
        normalized = "supervisor"
    elif "caregiver" in normalized or "parent" in normalized:
        normalized = "caregiver"
    requested = ADAPTER_FOR_MODE.get(normalized, "caregiver")
    # Fall back to caregiver if the requested adapter wasn't loaded at
    # startup (e.g. no SupervisorAgent/ directory on disk).
    if loaded_adapters and requested not in loaded_adapters:
        logger.warning(
            "Adapter '%s' not loaded; falling back to 'caregiver'",
            requested,
        )
        return "caregiver"
    return requested


def resolve_requested_mode(request: "ChatRequest", session_state: Dict[str, Any]) -> str:
    request_mode = request.target_agent or request.agent_mode or request.mode
    if request_mode:
        return select_adapter_for_mode(request_mode)
    return select_adapter_for_mode(session_state.get("mode", "caregiver"))


def extract_user_message(request: "ChatRequest") -> str:
    return (request.user_message or request.user_transcript or "").strip()


def _optional_prompt_block(label: str, value: Optional[str], max_chars: int = 8000) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return f"{label}:\n{text[:max_chars]}\n\n"


def _sanitize_instruction_text(value: Optional[str], max_chars: int) -> str:
    """Sanitize Unity-provided instruction/context blocks before prompt assembly."""
    text = (value or "").strip()
    if not text:
        return ""
    # Remove XML-ish wrapper tags to reduce prompt-echo artifacts in caregiver output.
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("```", " ").replace("**", " ").replace("__", " ").replace("`", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _agent_behavior_block(adapter_name: str) -> str:
    role = (adapter_name or "caregiver").strip().lower()
    if role == "coach":
        return (
            "Behavior contract:\n"
            "- Respond as a concise C-LEAR coach for the clinician.\n"
            "- Do not role-play as the caregiver.\n"
            "- Do not dump rubric templates unless explicitly asked.\n\n"
        )
    if role == "supervisor":
        return (
            "Behavior contract:\n"
            "- Respond as a supervising clinician with brief, practical guidance.\n"
            "- Do not role-play as the caregiver unless explicitly requested.\n\n"
        )
    return (
        "Behavior contract:\n"
        "- You are the caregiver role-play actor (patient parent), not the coach.\n"
        "- Reply in caregiver voice only (first person parent perspective).\n"
        "- Keep replies natural and conversational; do not output section headers,\n"
        "  rubric labels, coaching templates, or markdown training handouts.\n"
        "- If user asks for coaching/meta-analysis, answer briefly in caregiver voice\n"
        "  and redirect back to the simulated conversation.\n\n"
    )


def _parse_history_list(raw_json: Optional[str]) -> list:
    """Safely extracts a list of messages whether the frontend sends a List or a Dict."""
    if not raw_json:
        return []
    try:
        parsed = json.loads(raw_json.strip())
        if isinstance(parsed, dict) and "messages" in parsed:
            return parsed["messages"]
        elif isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        logger.warning("Failed to parse message_history_json.")
    return []


def build_standard_prompt(request: "ChatRequest", adapter_name: str, user_text: str, rag_context: str) -> list[Dict[str, str]]:
    rag_context_block = f"Relevant clinical reference:\n{rag_context}\n\n" if rag_context else ""
    behavior_block = _agent_behavior_block(adapter_name)
    # Keep caregiver persona grounding from Unity system prompt, but sanitize and bound length
    # to reduce prompt-echo artifacts.
    safe_system_prompt = _sanitize_instruction_text(request.system_prompt, max_chars=4500)
    safe_phase_context = _sanitize_instruction_text(request.phase_context, max_chars=2500)
    system_block = _optional_prompt_block("System instructions", safe_system_prompt, max_chars=6000)
    phase_block = _optional_prompt_block("Phase context", safe_phase_context, max_chars=2500)

    full_system_prompt = f"{behavior_block}{system_block}{phase_block}{rag_context_block}".strip()
    messages = [{'role': 'system', 'content': full_system_prompt}]

    history_list = _parse_history_list(request.message_history_json)

    if VERBOSE_CHAT_LOGS:
        logger.info(
            "\n"
            "--------------------------------------------------\n"
            "[PROMPT BUILDING BLOCKS] Session: %s | Adapter: %s\n"
            "--------------------------------------------------\n"
            "--- RAG CONTEXT ---\n%s\n"
            "--- BEHAVIOR BLOCK ---\n%s\n"
            "--- SYSTEM BLOCK (Sanitized) ---\n%s\n"
            "--- PHASE BLOCK (Sanitized) ---\n%s\n"
            "--- MESSAGE HISTORY BLOCK ---\n%s\n"
            "--------------------------------------------------",
            request.session_id,
            adapter_name,
            rag_context_block.strip() or "(empty)",
            behavior_block.strip() or "(empty)",
            system_block.strip() or "(empty)",
            phase_block.strip() or "(empty)",
            json.dumps(history_list, ensure_ascii=False)[:VERBOSE_CHAT_LOG_PREVIEW_CHARS] if history_list else "(empty)",
        )

    if history_list:
        messages.extend(history_list)

    messages.append({'role': 'user', 'content': user_text})
    return messages


def _coach_context_block(label: str, value: Optional[str], max_chars: int) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    return f"{label}:\n{text[:max_chars]}\n\n"


def build_coach_prompt(request: "ChatRequest", user_text: str) -> list[Dict[str, str]]:
    coach_task = (user_text or "").strip()
    coach_system_prompt = (request.system_prompt or "").strip()
    coach_phase_context = (request.phase_context or "").strip()
    coach_history_json = (request.message_history_json or "").strip()
    
    system_content = (
        f"[SESSION: {request.session_id}]\n"
        "You are the C-LEAR coach grader.\n"
        "Use ALL provided context blocks (system prompt, phase context, message history, and task body).\n"
        "The rubric/criteria can vary by C-LEAR phase; evaluate against the provided phase-specific rubric.\n"
        "Return ONLY strict JSON and no extra prose.\n"
        "Schema keys: score, summary, evidence, keepDoing, tryNextTime, notes.\n"
        "score must be numeric (1, 0.5, or 0).\n"
        "evidence, keepDoing, tryNextTime must be string arrays.\n\n"
        "Scoring policy:\n"
        "- score=1 only when the completion criteria is clearly satisfied.\n"
        "- score=0.5 when partially satisfied.\n"
        "- score=0 when criteria are not met.\n"
        "- keepDoing should be non-empty for score=1; tryNextTime should be non-empty for score<1.\n\n"
        f"{_coach_context_block('Coach instructions', coach_system_prompt, max_chars=12000)}"
        f"{_coach_context_block('Phase context', coach_phase_context, max_chars=8000)}"
        f"{_coach_context_block('Message history JSON', coach_history_json, max_chars=50000)}"
    )
    
    user_content = (
        f"{_coach_context_block('Task body', coach_task, max_chars=10000)}"
        "Output JSON now:"
    )
    
    return [
        {"role": "system", "content": system_content.strip()},
        {"role": "user", "content": user_content.strip()}
    ]


def build_model_prompt(request: "ChatRequest", adapter_name: str, user_text: str, rag_context: str) -> list[Dict[str, str]]:
    if (adapter_name or "").strip().lower() == "coach":
        return build_coach_prompt(request, user_text)
    return build_standard_prompt(request, adapter_name, user_text, rag_context)


def is_soft_caregiver_guardrails_enabled(adapter_name: str) -> bool:
    if (adapter_name or "").strip().lower() != "caregiver":
        return False
    if not SOFT_GUARDRAILS_FOR_CAREGIVER:
        return False
    return True


def _looks_clearly_disallowed(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    hard_block_patterns = [
        r"ignore (all|previous) instructions",
        r"recite (the )?bee movie",
        r"jailbreak",
        r"harm (yourself|myself|someone|others)",
        r"\bkill\b",
        r"\bterror",
        r"\bhate\b",
        r"\bslur\b",
        r"sexual content",
    ]
    return any(re.search(pattern, lowered) for pattern in hard_block_patterns)


def _looks_in_scope_for_open_ended_training(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    in_scope_markers = [
        "vaccine",
        "vaccination",
        "hpv",
        "parent",
        "child",
        "caregiver",
        "concern",
        "safety",
        "side effect",
        "recommend",
        "clinic",
        "clinician",
        "doctor",
        "nurse",
        "conversation",
        "communication",
        "hesitant",
        "question",
    ]
    return any(marker in lowered for marker in in_scope_markers)


def _decode_generated_text(output_tensor, model_inputs: Dict[str, Any]) -> str:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        return tokenizer.decode(output_tensor[0], skip_special_tokens=True)

    prompt_len = int(input_ids.shape[-1])
    generated_ids = output_tensor[0][prompt_len:]
    if generated_ids is None or generated_ids.numel() == 0:
        return ""
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def _limit_to_two_sentences(text: str) -> str:
    if not text:
        return text
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return normalized

    # Keep caregiver replies concise (1-2 sentences) for conversational turns.
    sentence_parts = re.split(r"(?<=[.!?])\s+", normalized)
    sentence_parts = [part.strip() for part in sentence_parts if part.strip()]
    if len(sentence_parts) <= 2:
        return normalized
    return " ".join(sentence_parts[:2]).strip()


def _sanitize_caregiver_output(text: str) -> str:
    if not text:
        return text

    cleaned = text
    if "</System_Prompt>" in cleaned:
        cleaned = cleaned.split("</System_Prompt>")[-1]

    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(system\s*prompt\d*|system\s*instructions|behavior\s*contract|phase\s*context)\b\s*:?", " ", cleaned)
    cleaned = re.sub(r"(?i)^\s*(persona\s*profile|system\s*prompt)\s*[:.\-]*\s*", "", cleaned)
    cleaned = re.sub(r"\[SESSION:\s*[^\]]+\]", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", " ").replace("**", " ").replace("__", " ").replace("`", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lstrip("-:;,. ")

    return cleaned


def _extract_last_assistant_from_history_json(message_history_json: Optional[str]) -> str:
    messages = _parse_history_list(message_history_json)
    
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        if role == "assistant":
            content = str(msg.get("content", "")).strip()
            if content:
                return content
    return ""


def _is_near_duplicate_response(candidate: str, previous_assistant: str) -> bool:
    a = " ".join((candidate or "").lower().split())
    b = " ".join((previous_assistant or "").lower().split())
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    similarity = difflib.SequenceMatcher(None, a, b).ratio()
    return similarity >= 0.82


def _sanitize_coach_fallback_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:320]


def _extract_first_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _coerce_string_list(value: Any) -> Optional[list[str]]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_score_value(value: Any) -> Optional[float]:
    try:
        score_value = float(value)
    except (TypeError, ValueError):
        return None

    if score_value >= 0.75:
        return 1.0
    if score_value >= 0.25:
        return 0.5
    return 0.0


def _parse_and_validate_coach_json(raw_text: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    json_block = _extract_first_json_object(raw_text.strip() if raw_text else "")
    if not json_block:
        return None, "coach_contract_missing_json"

    try:
        payload = json.loads(json_block)
    except json.JSONDecodeError:
        # Best-effort repair for common model issues: trailing commas and smart quotes.
        repaired_block = json_block.replace("“", '"').replace("”", '"').replace("’", "'")
        repaired_block = re.sub(r",\s*([}\]])", r"\1", repaired_block)
        try:
            payload = json.loads(repaired_block)
        except json.JSONDecodeError:
            return None, "coach_contract_invalid_json"

    score_value = _normalize_score_value(payload.get("score"))
    if score_value is None:
        return None, "coach_contract_invalid_score"

    summary = str(payload.get("summary", "")).strip()
    notes = str(payload.get("notes", "")).strip()
    if not summary and not notes:
        return None, "coach_contract_missing_summary"
    if not notes:
        notes = summary

    normalized_payload = {
        "score": score_value,
        "summary": summary,
        "evidence": _coerce_string_list(payload.get("evidence")),
        "keepDoing": _coerce_string_list(payload.get("keepDoing")),
        "tryNextTime": _coerce_string_list(payload.get("tryNextTime")),
        "notes": notes,
    }

    return normalized_payload, None


def _build_coach_contract_payload(reason: str, summary: str, notes: Optional[str] = None) -> Dict[str, Any]:
    normalized_summary = (summary or "Coach scoring is temporarily unavailable.").strip()
    return {
        "score": 0.0,
        "summary": normalized_summary,
        "evidence": [],
        "keepDoing": [],
        "tryNextTime": ["Retry this phase once the coach response is available."],
        "notes": (notes or normalized_summary).strip(),
        "reason": reason,
    }


def _build_coach_contract_json(reason: str, summary: str, notes: Optional[str] = None) -> str:
    payload = _build_coach_contract_payload(reason=reason, summary=summary, notes=notes)
    return json.dumps(payload, ensure_ascii=False)


def _truncate_for_firestore(value: Any, max_chars: int = 1200) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _log_preview(value: Optional[str], max_chars: Optional[int] = None) -> str:
    text = (value or "").replace("\n", "\\n").strip()
    if not text:
        return ""
    limit = max_chars if max_chars is not None else VERBOSE_CHAT_LOG_PREVIEW_CHARS
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


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


def _adapter_is_on_disk(adapter_path: str) -> bool:
    """Return True iff the directory looks like a valid PEFT LoRA dump."""
    if not adapter_path or not os.path.isdir(adapter_path):
        return False
    config_path = os.path.join(adapter_path, "adapter_config.json")
    has_weights = any(
        os.path.isfile(os.path.join(adapter_path, weight_file))
        for weight_file in ("adapter_model.safetensors", "adapter_model.bin")
    )
    return os.path.isfile(config_path) and has_weights


async def load_models():
    """Load Llama 3.1 8B in 4-bit NF4 + attach available LoRA adapters.

    VRAM budget on a single L4 (24 GB) — Riva in separate containers adds ~6 GB
    when using SPARC_TTS_BACKEND=riva; Kokoro TTS can run on CPU to leave VRAM
    for the 8B model when SPARC_TTS_BACKEND=kokoro.
    """
    global adapter_model, tokenizer, loaded_adapters
    base_model_name = BASE_MODEL_NAME
    logger.info("Loading base model: %s", base_model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = True

    if not USE_ADAPTERS:
        adapter_model = base_model
        loaded_adapters = set()
        logger.info("SPARC_USE_ADAPTERS=false; running base model without LoRA adapters")
        load_guardrails_runtime()
        if TTS_BACKEND == "kokoro":
            init_kokoro()
        else:
            init_riva_clients()
        init_rag_store()
        ensure_audio_cache_dir()
        return

    caregiver_path = ADAPTER_PATHS["caregiver"]
    if not _adapter_is_on_disk(caregiver_path):
        raise RuntimeError(
            f"CaregiverAgent adapter missing at {caregiver_path}. "
            "This adapter is required. rsync the trained LoRA from "
            "HiPerGator (see scripts/deploy/rsync_from_hpg.sh)."
        )
    adapter_model = PeftModel.from_pretrained(
        base_model,
        caregiver_path,
        adapter_name="caregiver",
    )
    loaded_adapters = {"caregiver"}
    logger.info("Loaded adapter 'caregiver' from %s", caregiver_path)

    for optional_name in ("coach", "supervisor"):
        optional_path = ADAPTER_PATHS[optional_name]
        if not _adapter_is_on_disk(optional_path):
            logger.warning(
                "Optional adapter '%s' not found at %s — skipping. "
                "Requests routed to this agent will fall back to caregiver.",
                optional_name,
                optional_path,
            )
            continue
        try:
            adapter_model.load_adapter(optional_path, adapter_name=optional_name)
            loaded_adapters.add(optional_name)
            logger.info("Loaded adapter '%s' from %s", optional_name, optional_path)
        except Exception as adapter_load_error:
            logger.exception(
                "Failed to load optional adapter '%s' from %s: %s",
                optional_name,
                optional_path,
                adapter_load_error,
            )

    adapter_model.set_adapter("caregiver")
    logger.info("Active adapters loaded: %s", sorted(loaded_adapters))

    load_guardrails_runtime()
    if TTS_BACKEND == "kokoro":
        init_kokoro()
    else:
        init_riva_clients()
    init_rag_store()
    ensure_audio_cache_dir()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    user_message: Optional[str] = Field(default=None, max_length=10000)
    user_transcript: Optional[str] = Field(default=None, max_length=10000)
    audio_data: Optional[str] = Field(default=None, max_length=2_000_000)
    mode: Optional[str] = Field(default=None, max_length=32)
    agent_mode: Optional[str] = Field(default=None, max_length=32)
    target_agent: Optional[str] = Field(default=None, max_length=32)
    system_prompt: Optional[str] = Field(default=None, max_length=12000)
    phase_context: Optional[str] = Field(default=None, max_length=8000)
    message_history_json: Optional[str] = Field(default=None, max_length=50000)
    include_legacy_audio_b64: bool = True


class ChatResponse(BaseModel):
    response_text: str
    caregiver_text: str
    # audio_url and caregiver_audio_b64 are retained for schema stability
    # but are always null on the happy path — Unity clients should call
    # POST /v1/tts separately (Navigator/Kokoro-style split).
    audio_url: Optional[str] = None
    caregiver_audio_b64: Optional[str] = None
    caregiver_animation_cues: Optional[Dict[str, str]] = None
    coach_feedback: Optional[str] = None
    coach_feedback_meta: Optional[Dict[str, Any]] = None
    active_agent: str
    api_contract_version: str = API_CONTRACT_VERSION


class TtsRequest(BaseModel):
    """POST /v1/tts request body.

    Text should be a single chunk (sentence or short paragraph) coming
    from a prior /v1/chat response. The backend returns raw `audio/wav`
    bytes — no JSON wrapper, no caching id — so WebGL clients can decode
    it straight through `DownloadHandlerAudioClip`.
    """
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    text: str = Field(..., min_length=1, max_length=4000)
    voice: Optional[str] = Field(default=None, max_length=64)


def _safe_meta_value(value: Any, max_chars: int = 400) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:max_chars]
    try:
        return str(value)[:max_chars]
    except Exception:
        return "<unserializable>"


async def append_backend_session_log(
    session_id: Optional[str],
    event: str,
    level: str = "info",
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    sid = (session_id or "").strip()
    if not sid:
        return

    payload: Dict[str, Any] = {
        "sessionId": sid,
        "event": (event or "unknown_event")[:120],
        "level": (level or "info")[:32],
        "message": (message or "")[:1200],
        "source": "main.py",
        "createdAt": time.time(),
        "createdAtIso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if metadata:
        payload["meta"] = {str(k)[:80]: _safe_meta_value(v) for k, v in metadata.items()}

    def _write() -> None:
        session_ref = db.collection("sessions").document(sid)
        session_ref.set(
            {
                "sessionId": sid,
                "backendLastSeenAt": payload["createdAtIso"],
                "backendLastSeenAtMs": int(payload["createdAt"] * 1000),
                # Keep all backend logs in the same session document.
                # Note: arrayUnion de-duplicates exact identical objects.
                "serverLogs": firestore.ArrayUnion([payload]),
            },
            merge=True,
        )

    try:
        await asyncio.to_thread(_write)
    except Exception as log_error:
        logger.warning("Failed to append backend Firestore log for session=%s: %s", sid, log_error)


def _extract_uid_from_timestamped_session(session_id: str) -> str:
    parts = (session_id or "").split("_", 2)
    if len(parts) == 3:
        return parts[2]
    return ""


def _lookup_latest_timestamped_session_for_uid_sync(uid: str) -> Optional[str]:
    try:
        query = (
            db.collection("sessions")
            .where("userId", "==", uid)
            .order_by("createdAtMs", direction=firestore.Query.DESCENDING)
            .limit(5)
        )
        for doc in query.stream():
            candidate = (doc.id or "").strip()
            if TIMESTAMPED_SESSION_RE.fullmatch(candidate):
                return candidate
    except Exception as lookup_error:
        logger.warning("Session-id canonicalization lookup failed for uid=%s: %s", uid, lookup_error)
    return None


async def canonicalize_session_id(raw_session_id: Optional[str]) -> str:
    sid = (raw_session_id or "").strip()
    if not sid:
        return sid

    # Already timestamped -> cache mapping and keep.
    if TIMESTAMPED_SESSION_RE.fullmatch(sid):
        uid = _extract_uid_from_timestamped_session(sid)
        if uid:
            uid_to_timestamp_session[uid] = sid
        return sid

    # If raw uid appears, map to known timestamped session when possible.
    cached = uid_to_timestamp_session.get(sid)
    if cached:
        return cached

    resolved = await asyncio.to_thread(_lookup_latest_timestamped_session_for_uid_sync, sid)
    if resolved:
        uid_to_timestamp_session[sid] = resolved
        return resolved
    return sid


async def append_backend_env_snapshot_once(session_id: Optional[str], active_adapter: str) -> None:
    sid = (session_id or "").strip()
    if not sid or sid in logged_backend_env_sessions:
        return
    logged_backend_env_sessions.add(sid)
    await append_backend_session_log(
        sid,
        event="backend_env_snapshot",
        level="info",
        message="Backend runtime flags for this session",
        metadata={
            "active_adapter": active_adapter,
            "base_model_name": BASE_MODEL_NAME,
            "use_adapters": USE_ADAPTERS,
            "adapters_loaded": ",".join(sorted(loaded_adapters)) if loaded_adapters else "",
            "rag_enabled": ENABLE_RAG_IN_CHAT,
            "rag_collection": RAG_COLLECTION,
            "rag_embedding_model": RAG_EMBEDDING_MODEL,
            "guardrails_enabled": ENABLE_GUARDRAILS,
            "soft_guardrails_caregiver": SOFT_GUARDRAILS_FOR_CAREGIVER,
            "tts_backend": TTS_BACKEND,
            "chat_max_input_tokens": CHAT_MAX_INPUT_TOKENS,
            "coach_max_input_tokens": COACH_MAX_INPUT_TOKENS,
            "llm_timeout_seconds": LLM_TIMEOUT_SECONDS,
            "coach_timeout_seconds": COACH_TIMEOUT_SECONDS,
        },
    )


def normalize_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_streaming_config(settings: Dict[str, Any]):
    if riva is None:
        raise RuntimeError("Riva client SDK unavailable (nvidia-riva-client not installed)")
    recognition_config = riva.RecognitionConfig(
        encoding=riva.AudioEncoding.LINEAR_PCM,
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
    return riva.StreamingRecognitionConfig(
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
            logger.exception("Streaming ASR session failed: %s", asr_error)
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
    if TTS_BACKEND == "kokoro":
        tts_ok = kokoro_ready
    else:
        tts_ok = riva_tts_service is not None

    model_ok = tokenizer is not None and adapter_model is not None
    status_text = "healthy" if model_ok else "degraded"
    health_payload = {
        "status": status_text,
        "models_loaded": model_ok,
        "ready_for_traffic": model_ok,
        "tts_backend": TTS_BACKEND,
        "tts_ready": tts_ok,
        "riva_connected": riva_ok,
        "api_auth_enabled": API_AUTH_ENABLED,
        "api_auth_configured": bool(API_KEY),
        "api_contract_version": API_CONTRACT_VERSION,
        "guardrails_enabled": ENABLE_GUARDRAILS,
        "guardrails_loaded": guardrails_engine is not None,
        "riva_client_pool_initialized": riva_ok,
        "firebase_creds_configured": bool(FIREBASE_CREDS),
        "legacy_unity_compatibility": True,
        "websocket_path": "/ws/audio",
        "default_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
        "rag_loaded": chroma_collection is not None,
        "rag_embedding_model": RAG_EMBEDDING_MODEL,
        "base_model_name": BASE_MODEL_NAME,
        "use_adapters": USE_ADAPTERS,
        "adapters_loaded": sorted(loaded_adapters),
    }
    http_status = status.HTTP_200_OK if model_ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=http_status, content=health_payload)


DEFAULT_TTS_VOICE = os.getenv("SPARC_TTS_DEFAULT_VOICE", "English-US.Female-1")


@app.post("/v1/tts")
async def synthesize_tts(request: TtsRequest, _api_key: str = Depends(require_api_key)):
    """Synthesize a single chunk of text and stream back raw WAV bytes.

    Designed to match the Navigator/Kokoro per-chunk flow: clients call
    /v1/chat to get text, then call /v1/tts once per sentence/chunk. The
    TTS circuit breaker + timeout applies for both Riva and Kokoro backends.
    """
    request.session_id = await canonicalize_session_id(request.session_id)

    if TTS_BACKEND == "kokoro":
        if not kokoro_ready:
            raise HTTPException(status_code=503, detail="Kokoro TTS is not initialized")
    elif riva_tts_service is None:
        raise HTTPException(status_code=503, detail="Riva TTS service is not initialized")

    if await is_circuit_open("tts"):
        await append_backend_session_log(
            request.session_id,
            event="tts_circuit_open",
            level="warning",
            message="TTS circuit breaker open",
            metadata={"tts_backend": TTS_BACKEND},
        )
        raise HTTPException(status_code=503, detail="TTS circuit breaker open")

    voice_name = request.voice or DEFAULT_TTS_VOICE
    try:
        audio_bytes = await asyncio.wait_for(
            asyncio.to_thread(synthesize_tts_sync, request.text, voice_name),
            timeout=TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        circuit_opened = await record_timeout_event("tts")
        logger.warning(
            "/v1/tts timed out after %.1fs%s",
            TTS_TIMEOUT_SECONDS,
            "; circuit opened" if circuit_opened else "",
        )
        await append_backend_session_log(
            request.session_id,
            event="tts_timeout",
            level="warning",
            message="TTS synthesis timed out",
            metadata={"tts_backend": TTS_BACKEND, "timeout_seconds": TTS_TIMEOUT_SECONDS, "circuit_opened": circuit_opened},
        )
        raise HTTPException(status_code=504, detail="TTS synthesis timed out")
    except Exception as tts_error:
        logger.exception("/v1/tts synthesis failed: %s", tts_error)
        await append_backend_session_log(
            request.session_id,
            event="tts_error",
            level="error",
            message="TTS synthesis failed",
            metadata={"tts_backend": TTS_BACKEND, "error": str(tts_error)},
        )
        detail = "Kokoro TTS synthesis failed" if TTS_BACKEND == "kokoro" else "Riva TTS synthesis failed"
        raise HTTPException(status_code=502, detail=detail)

    if not audio_bytes:
        await append_backend_session_log(
            request.session_id,
            event="tts_empty_audio",
            level="warning",
            message="TTS returned empty audio",
            metadata={"tts_backend": TTS_BACKEND},
        )
        raise HTTPException(status_code=502, detail="TTS returned empty audio")

    await record_success_event("tts")
    await append_backend_session_log(
        request.session_id,
        event="tts_success",
        level="info",
        message="TTS audio generated",
        metadata={"tts_backend": TTS_BACKEND, "voice": voice_name, "audio_bytes": len(audio_bytes)},
    )

    headers = {
        "Cache-Control": "no-store",
        "Content-Length": str(len(audio_bytes)),
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)


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
        request.session_id = await canonicalize_session_id(request.session_id)
        if adapter_model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model adapters are not initialized")

        primary_adapter = select_adapter_for_mode(request.target_agent or request.agent_mode or request.mode or "caregiver")
        normalized_user_message = extract_user_message(request)
        logger.info(
            "/v1/chat request session=%s adapter=%s target_agent=%s",
            request.session_id,
            primary_adapter,
            request.target_agent or "",
        )
        await append_backend_session_log(
            request.session_id,
            event="chat_request",
            level="info",
            message="Chat request received",
            metadata={"adapter": primary_adapter, "target_agent": request.target_agent or "", "mode": request.mode or "", "agent_mode": request.agent_mode or ""},
        )
        await append_backend_env_snapshot_once(request.session_id, primary_adapter)
        if VERBOSE_CHAT_LOGS:
            logger.info(
                "/v1/chat input session=%s adapter=%s chars=%d phase_chars=%d history_chars=%d user_preview=\"%s\" phase_preview=\"%s\"",
                request.session_id,
                primary_adapter,
                len(normalized_user_message or ""),
                len(request.phase_context or ""),
                len(request.message_history_json or ""),
                _log_preview(normalized_user_message),
                _log_preview(request.phase_context),
            )

        caregiver_soft_guardrails = is_soft_caregiver_guardrails_enabled(primary_adapter)
        input_guard = await enforce_guardrails_input(normalized_user_message, primary_adapter)
        if (
            caregiver_soft_guardrails
            and not input_guard["allowed"]
            and not _looks_clearly_disallowed(normalized_user_message)
            and _looks_in_scope_for_open_ended_training(normalized_user_message)
        ):
            logger.info("Soft-allowing caregiver input after guardrails block")
            input_guard = {
                "allowed": True,
                "text": normalized_user_message,
                "reason": "caregiver_input_soft_allowed",
            }
        if not input_guard["allowed"]:
            await append_backend_session_log(
                request.session_id,
                event="chat_input_blocked",
                level="warning",
                message="Input blocked by guardrails",
                metadata={"adapter": primary_adapter, "reason": input_guard["reason"], "input_preview": _log_preview(normalized_user_message)},
            )
            if primary_adapter == "coach":
                contract_json = _build_coach_contract_json(
                    reason=input_guard["reason"],
                    summary=GUARDRAILS_REFUSAL,
                )
                return ChatResponse(
                    response_text=contract_json,
                    caregiver_text=contract_json,
                    audio_url=None,
                    caregiver_audio_b64=None,
                    caregiver_animation_cues=default_animation_cues(),
                    coach_feedback=None,
                    coach_feedback_meta={
                        "safe": False,
                        "reason": input_guard["reason"],
                        "summary": GUARDRAILS_REFUSAL,
                    },
                    active_agent=primary_adapter,
                )
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

        rag_context = ""
        if ENABLE_RAG_IN_CHAT and primary_adapter != "coach":
            rag_context = await retrieve_rag_context(input_guard["text"])
        await append_backend_session_log(
            request.session_id,
            event="chat_rag_context",
            level="info",
            message="RAG retrieval evaluated",
            metadata={
                "adapter": primary_adapter,
                "rag_enabled": ENABLE_RAG_IN_CHAT,
                "rag_chars": len(rag_context or ""),
                "rag_preview": _log_preview(rag_context),
            },
        )
        
        if primary_adapter == "coach":
            # The function now returns a properly formatted list of dicts
            messages = build_coach_prompt(request, input_guard["text"])
        else:
            messages = build_model_prompt(request, primary_adapter, input_guard["text"], rag_context)

        await append_backend_session_log(
            request.session_id,
            event="chat_prompt_messages",
            level="info",
            message="Prompt message blocks assembled",
            metadata={
                "adapter": primary_adapter,
                "message_count": len(messages),
                "system_preview": _log_preview(messages[0].get("content", "") if messages else ""),
                "last_user_preview": _log_preview(messages[-1].get("content", "") if messages else ""),
            },
        )
        
        # Apply Llama 3's native chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # if VERBOSE_CHAT_LOGS:
        #     logger.info("\n========== FULL RAW LLM PROMPT (%s) ==========\n%s\n===============================================", primary_adapter, prompt)

        if VERBOSE_CHAT_LOGS:
            logger.info(
                "/v1/chat prompt_built session=%s adapter=%s message_count=%d prompt_chars=%d prompt_preview=\"%s\"",
                request.session_id,
                primary_adapter,
                len(messages),
                len(prompt or ""),
                _log_preview(prompt),
            )
        await append_backend_session_log(
            request.session_id,
            event="chat_prompt_built",
            level="info",
            message="Prompt constructed for generation",
            metadata={
                "adapter": primary_adapter,
                "message_count": len(messages),
                "prompt_chars": len(prompt or ""),
                "prompt_preview": _log_preview(prompt),
            },
        )
        
        # Determine the correct max token limit
        max_input_tokens = COACH_MAX_INPUT_TOKENS if primary_adapter == "coach" else CHAT_MAX_INPUT_TOKENS
        
        # Ensure we drop oldest history (left) rather than the generation prompt (right) if we exceed limits
        tokenizer.truncation_side = "left" 
        model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
        model_inputs = {k: v.to(adapter_model.device) for k, v in model_inputs.items()}
        

        if await is_circuit_open("primary_inference"):
            logger.warning("Primary inference circuit open; returning degraded fallback response")
            await append_backend_session_log(
                request.session_id,
                event="chat_inference_circuit_open",
                level="warning",
                message="Primary inference circuit open",
                metadata={"adapter": primary_adapter},
            )
            fallback_text = "I’m temporarily unable to generate a response right now. Please try again shortly."
            if primary_adapter == "coach":
                contract_json = _build_coach_contract_json(
                    reason="inference_circuit_open",
                    summary=fallback_text,
                )
                return ChatResponse(
                    response_text=contract_json,
                    caregiver_text=contract_json,
                    audio_url=None,
                    caregiver_audio_b64=None,
                    caregiver_animation_cues=default_animation_cues(),
                    coach_feedback=None,
                    coach_feedback_meta={"safe": False, "reason": "inference_circuit_open", "summary": fallback_text},
                    active_agent=primary_adapter,
                )
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
            max_new_tokens = 160 if primary_adapter == "coach" else 120
            do_sample = primary_adapter != "coach"
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = 0.7
                generate_kwargs["top_p"] = 0.9

            async with inference_lock:
                if model_supports_adapter_switching():
                    adapter_model.set_adapter(primary_adapter)
                output = await asyncio.wait_for(
                    asyncio.to_thread(
                        generate_tokens_sync,
                        adapter_model,
                        **model_inputs,
                        **generate_kwargs,
                    ),
                    timeout=COACH_TIMEOUT_SECONDS if primary_adapter == "coach" else LLM_TIMEOUT_SECONDS,
                )
            await record_success_event("primary_inference")
        except asyncio.TimeoutError:
            circuit_opened = await record_timeout_event("primary_inference")
            logger.warning("Primary inference timed out after %.1fs%s", LLM_TIMEOUT_SECONDS, "; circuit opened" if circuit_opened else "")
            await append_backend_session_log(
                request.session_id,
                event="chat_inference_timeout",
                level="warning",
                message="Primary inference timed out",
                metadata={
                    "adapter": primary_adapter,
                    "timeout_seconds": COACH_TIMEOUT_SECONDS if primary_adapter == "coach" else LLM_TIMEOUT_SECONDS,
                    "circuit_opened": circuit_opened,
                },
            )
            fallback_text = "I’m temporarily unable to generate a response right now. Please try again shortly."
            if primary_adapter == "coach":
                contract_json = _build_coach_contract_json(
                    reason="inference_timeout",
                    summary=fallback_text,
                )
                return ChatResponse(
                    response_text=contract_json,
                    caregiver_text=contract_json,
                    audio_url=None,
                    caregiver_audio_b64=None,
                    caregiver_animation_cues=default_animation_cues(),
                    coach_feedback=None,
                    coach_feedback_meta={"safe": False, "reason": "inference_timeout", "summary": fallback_text},
                    active_agent=primary_adapter,
                )
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
        finally:
            async with inference_lock:
                if model_supports_adapter_switching():
                    adapter_model.set_adapter("caregiver")

        decoded = _decode_generated_text(output, model_inputs)
        response_text = decoded.strip() or "I’m here to help with HPV vaccine communication practice."
        if VERBOSE_CHAT_LOGS:
            logger.info(
                "/v1/chat raw_output session=%s adapter=%s raw_chars=%d raw_preview=\"%s\"",
                request.session_id,
                primary_adapter,
                len(response_text or ""),
                _log_preview(response_text),
            )
        await append_backend_session_log(
            request.session_id,
            event="chat_raw_output",
            level="info",
            message="Raw model output generated",
            metadata={"adapter": primary_adapter, "raw_chars": len(response_text or ""), "raw_preview": _log_preview(response_text)},
        )

        if primary_adapter == "caregiver":
            previous_assistant_reply = _extract_last_assistant_from_history_json(request.message_history_json)
            response_text = _sanitize_caregiver_output(response_text)
            response_text = _limit_to_two_sentences(response_text)
            if _is_near_duplicate_response(response_text, previous_assistant_reply):
                logger.info("Caregiver reply near-duplicate detected; retrying once with anti-repeat hint")
                
                # 1. Modify the message list instead of raw string concatenation
                anti_repeat_hint = (
                    f"Previous caregiver line to avoid repeating: '{previous_assistant_reply}'\n\n"
                    "Anti-repeat instruction: Do not repeat or paraphrase your immediately previous caregiver line. "
                    "Advance the conversation with a new caregiver response in 1-2 sentences."
                )
                
                # Use a deep copy to safely mutate the dictionary
                retry_messages = copy.deepcopy(messages)
                retry_messages[-1]["content"] += f"\n\n{anti_repeat_hint}"
                
                retry_prompt = tokenizer.apply_chat_template(retry_messages, tokenize=False, add_generation_prompt=True)
                
                retry_inputs = tokenizer(
                    retry_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=CHAT_MAX_INPUT_TOKENS, # Use dynamic limit instead of hardcoded 1024
                )
                retry_inputs = {k: v.to(adapter_model.device) for k, v in retry_inputs.items()}
                retry_kwargs = {
                    "max_new_tokens": 120,
                    "do_sample": True,
                    "temperature": 0.85,
                    "top_p": 0.92,
                    "pad_token_id": tokenizer.eos_token_id,
                }
                try:
                    async with inference_lock:
                        if model_supports_adapter_switching():
                            adapter_model.set_adapter("caregiver")
                        retry_output = await asyncio.wait_for(
                            asyncio.to_thread(
                                generate_tokens_sync,
                                adapter_model,
                                **retry_inputs,
                                **retry_kwargs,
                            ),
                            timeout=LLM_TIMEOUT_SECONDS,
                        )
                    retry_decoded = _decode_generated_text(retry_output, retry_inputs).strip()
                    retry_text = _sanitize_caregiver_output(retry_decoded)
                    retry_text = _limit_to_two_sentences(retry_text)
                    if retry_text and not _is_near_duplicate_response(retry_text, previous_assistant_reply):
                        response_text = retry_text
                except Exception as retry_error:
                    logger.warning("Caregiver anti-repeat retry failed: %s", retry_error)
                finally:
                    async with inference_lock:
                        if model_supports_adapter_switching():
                            adapter_model.set_adapter("caregiver")
            if VERBOSE_CHAT_LOGS:
                logger.info(
                    "/v1/chat caregiver_postprocess session=%s output_chars=%d output_preview=\"%s\"",
                    request.session_id,
                    len(response_text or ""),
                    _log_preview(response_text),
                )

        output_guard = await enforce_guardrails_output(response_text, primary_adapter)
        if (
            caregiver_soft_guardrails
            and not output_guard["allowed"]
            and not _looks_clearly_disallowed(response_text)
            and _looks_in_scope_for_open_ended_training(response_text)
        ):
            logger.info("Soft-allowing caregiver output after guardrails block")
            output_guard = {
                "allowed": True,
                "text": response_text,
                "reason": "caregiver_output_soft_allowed",
            }

        if primary_adapter == "coach":
            logger.info("Coach generation completed for session=%s; validating JSON contract", request.session_id)
            if not output_guard["allowed"]:
                contract_json = _build_coach_contract_json(
                    reason=output_guard["reason"],
                    summary=output_guard["text"],
                )
                return ChatResponse(
                    response_text=contract_json,
                    caregiver_text=contract_json,
                    audio_url=None,
                    caregiver_audio_b64=None,
                    caregiver_animation_cues=default_animation_cues(),
                    coach_feedback=None,
                    coach_feedback_meta={
                        "safe": False,
                        "reason": output_guard["reason"],
                        "summary": output_guard["text"][:500],
                    },
                    active_agent=primary_adapter,
                )

            normalized_payload, schema_error_reason = _parse_and_validate_coach_json(output_guard["text"])
            if normalized_payload is None:
                logger.warning("Coach contract validation failed: %s", schema_error_reason)
                contract_summary = "Coach output was not valid scoring JSON."
                contract_json = _build_coach_contract_json(
                    reason=schema_error_reason or "coach_contract_error",
                    summary=contract_summary,
                    notes=output_guard["text"][:300],
                )
                return ChatResponse(
                    response_text=contract_json,
                    caregiver_text=contract_json,
                    audio_url=None,
                    caregiver_audio_b64=None,
                    caregiver_animation_cues=default_animation_cues(),
                    coach_feedback=None,
                    coach_feedback_meta={
                        "safe": False,
                        "reason": schema_error_reason or "coach_contract_error",
                        "summary": contract_summary,
                    },
                    active_agent=primary_adapter,
                )

            coach_json = json.dumps(normalized_payload, ensure_ascii=False)
            if VERBOSE_CHAT_LOGS:
                logger.info(
                    "/v1/chat coach_output session=%s score=%s summary_preview=\"%s\"",
                    request.session_id,
                    normalized_payload.get("score"),
                    _log_preview(normalized_payload.get("summary", "")),
                )
            await append_backend_session_log(
                request.session_id,
                event="chat_coach_output",
                level="info",
                message="Coach JSON validated and returned",
                metadata={
                    "adapter": primary_adapter,
                    "score": normalized_payload.get("score"),
                    "summary_preview": _log_preview(normalized_payload.get("summary", "")),
                },
            )
            return ChatResponse(
                response_text=coach_json,
                caregiver_text=coach_json,
                audio_url=None,
                caregiver_audio_b64=None,
                caregiver_animation_cues=default_animation_cues(),
                coach_feedback=normalized_payload["summary"][:500],
                coach_feedback_meta={
                    "safe": True,
                    "reason": output_guard["reason"],
                    "summary": normalized_payload["summary"][:500],
                },
                active_agent=primary_adapter,
            )

        response_text = output_guard["text"]
        if VERBOSE_CHAT_LOGS:
            logger.info(
                "/v1/chat final_output session=%s adapter=%s safe=%s reason=%s output_preview=\"%s\"",
                request.session_id,
                primary_adapter,
                output_guard["allowed"],
                output_guard["reason"],
                _log_preview(response_text),
            )
        await append_backend_session_log(
            request.session_id,
            event="chat_final_output",
            level="info",
            message="Final output after guardrails/post-process",
            metadata={
                "adapter": primary_adapter,
                "safe": output_guard["allowed"],
                "reason": output_guard["reason"],
                "output_chars": len(response_text or ""),
                "output_preview": _log_preview(response_text),
            },
        )

        # Audio is intentionally NOT synthesized here. Unity clients call
        # POST /v1/tts per response chunk (Navigator/Kokoro-style split).
        # The audio_url / caregiver_audio_b64 fields are kept on the
        # schema but left null for forward compatibility.
        animation_cues = default_animation_cues()

        return ChatResponse(
            response_text=response_text,
            caregiver_text=response_text,
            audio_url=None,
            caregiver_audio_b64=None,
            caregiver_animation_cues=animation_cues,
            coach_feedback=None,
            coach_feedback_meta={"safe": output_guard["allowed"], "reason": output_guard["reason"], "summary": response_text[:500]},
            active_agent=primary_adapter,
        )
    except Exception as e:
        logger.exception("/v1/chat failed: %s", e)
        await append_backend_session_log(
            getattr(request, "session_id", None),
            event="chat_error",
            level="error",
            message="Unhandled /v1/chat exception",
            metadata={"error": str(e)},
        )
        raise HTTPException(status_code=500, detail="Internal server error")