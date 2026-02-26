# SPARC-P Notebook API Documentation

## Documentation Baseline

- Baseline date: **February 26, 2026**
- Execution model: **conda-first** on HiPerGator and PubApps
- This document reflects the callable/runtime-facing surfaces described in active notebook guides.

---

## Scope

This API documentation covers the notebook tracks in this repository:

- `1_SPARC_Agent_Training.ipynb`
- `2_SPARC_Containerization_and_Deployment.ipynb`
- `2b_SPARC_Containerization_and_Deployment.ipynb`
- `3_SPARC_RIVA_Backend.ipynb`
- `4_SPARC_PubApp_Deployment.ipynb`
- `4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb`

Companion markdown sources:

- `1_SPARC_Agent_Training.md`
- `2_SPARC_Containerization_and_Deployment.md`
- `2b_SPARC_Containerization_and_Deployment.md`
- `3_SPARC_RIVA_Backend.md`
- `4_SPARC_PubApp_Deployment.md`
- `4b_SPARC_PubApp_Deployment_PixelStreaming.md`

> Note: Some functions are notebook scaffolds/prototypes for workflow assembly and validation, not standalone production package APIs.

---

## Table of Contents

1. Shared Runtime Context
2. Notebook 1 API (Training)
3. Notebook 2 / 2b API (Containerization & Deployment)
4. Notebook 3 API (Real-Time Backend)
5. Notebook 4 API (PubApps Backend)
6. Notebook 4b API (Pixel Streaming Variant)
7. Endpoint Contracts by Track
8. Generated Artifacts
9. Operational Notes

---

## Shared Runtime Context

### Representative constants across notebook tracks

- Base model identifiers are defined per track (training and serving profiles may differ by hardware target).
- HiPerGator paths are generally rooted under `/blue/...`.
- PubApps paths are generally rooted under `/pubapps/...`.
- Riva runtime guidance is aligned to `2.16.0` in backend setup paths.

### Shared dependency domains

- Model/training: `torch`, `transformers`, `bitsandbytes`, `peft`, `trl`
- Data/validation: `datasets`, `pydantic`, `json`
- Retrieval: `langchain`, `langchain-chroma`, `sentence-transformers`
- Sanitization: `presidio-analyzer`, `presidio-anonymizer`
- Runtime API: `fastapi`, `uvicorn`, `langgraph`, `nemoguardrails`
- Speech: `riva-python-clients`

---

## Notebook 1 API (Training)

Source: `1_SPARC_Agent_Training.md`

### Data preparation and sanitization

- `sanitize_text_with_presidio(text: str) -> str`
- `extract_text_from_document(doc_path)`

### RAG ingestion and vectorization

- `build_vector_store(doc_paths: List[str], collection_name: str)`
- `ingest_documents(source_path: str, collection_name: str)`

### Synthetic data and schema conversion

- `generate_synthetic_qa(document_chunk: str, num_pairs: int = 5)`
- `format_to_chat_schema(raw_data: List[Dict]) -> Dataset`
- `load_and_process_data(agent_type: str) -> Dataset`

### Fine-tuning and validation

- `run_qlora_training(train_file_path: str, output_dir: str)`
- `validate_agent(agent_name: str, test_prompts: List[str], model_schema: BaseModel = None)`

### Interaction helpers

- `load_agent_adapter(agent_name)`
- `chat_individual(message, history, agent_selection)`
- `multi_agent_orchestrator(user_message, history)`

### Job generation

- `generate_slurm_script()`

---

## Notebook 2 / 2b API (Containerization & Deployment)

Sources:

- `2_SPARC_Containerization_and_Deployment.md`
- `2b_SPARC_Containerization_and_Deployment.md`

### Baseline container/deployment helpers

- `create_dockerfile()`
- `generate_production_script()`

### Command/reference cells

- Podman/Apptainer command blocks for local and cluster-aligned service startup
- Artifact staging/build guidance for baseline and Pixel Streaming variants

### Notes

- Conda remains canonical for host/runtime execution.
- Container paths are used where deployment architecture requires them (notably Riva and selected service packaging).

---

## Notebook 3 API (Real-Time Backend)

Source: `3_SPARC_RIVA_Backend.md`

### Riva setup and validation helpers

- `configure_riva()`
- `test_asr_service(audio_file_path)`
- `test_tts_service(text_input)`

### Guardrails setup

- `create_rails_config()`

### Orchestration classes/functions

- `SupervisorAgent`
- `CaregiverAgent`
- `CoachAgent`
- `handle_user_turn(user_transcript: str, supervisor, caregiver, coach)`

### FastAPI models and endpoints (Notebook 3 track)

#### `ChatRequest`

- `session_id: str` (pattern/length constrained)
- `user_transcript: str` (bounded)

#### `ChatResponse`

- `response_text: str`
- `audio_data_base64: Optional[str]`
- `emotion: str`
- `animation_cues: dict`
- `coach_feedback: Optional[dict]`

#### Endpoints

- `GET /health`
- `POST /v1/chat`

---

## Notebook 4 API (PubApps Backend)

Source: `4_SPARC_PubApp_Deployment.md`

### Backend runtime helpers (representative)

- Guardrails input/output enforcement helpers
- Sanitization helpers for persisted session metadata
- Audio persistence helpers (bounded size + TTL + cache pruning)
- Inference helpers with async-safe offloading pattern

### FastAPI models and endpoints (Notebook 4 track)

#### `ChatRequest`

- `session_id: str` (pattern/length constrained)
- `user_message: str` (bounded)
- `audio_data: Optional[str]` (bounded)

#### `ChatResponse`

- `response_text: str`
- `audio_url: Optional[str]`
- `emotion: str`
- `animation_cues: dict`
- `coach_feedback: Optional[dict]`

#### Endpoints

- `GET /health`
- `GET /v1/audio/{audio_id}`
- `POST /v1/chat` (includes in-app API key dependency in the notebook template)

---

## Notebook 4b API (Pixel Streaming Variant)

Source: `4b_SPARC_PubApp_Deployment_PixelStreaming.md`

Notebook 4b is a deployment/runtime topology variant (server-side rendering + signaling). It generally reuses the same backend API contract family as Notebook 4 unless explicitly overridden in that variant guide.

Primary additions are operational/deployment concerns:

- Signaling service integration
- Unity Linux server runtime deployment model
- Single-L4 VRAM budget-oriented runtime guidance

---

## Endpoint Contracts by Track

## 1) Notebook 3 backend contract

### `GET /health`

Returns service status and readiness metadata for backend runtime.

### `POST /v1/chat`

Request shape:

```json
{
  "session_id": "string",
  "user_transcript": "string"
}
```

Response shape:

```json
{
  "response_text": "string",
  "audio_data_base64": "string|null",
  "emotion": "string",
  "animation_cues": {},
  "coach_feedback": {}
}
```

## 2) Notebook 4/4b PubApps-facing contract (v1)

### `GET /health`

Returns status plus readiness/auth/guardrails/runtime metadata used by deployment operations.

### `GET /v1/audio/{audio_id}`

Serves cached TTS audio artifact by generated identifier.

### `POST /v1/chat`

Request shape:

```json
{
  "session_id": "string",
  "user_message": "string",
  "audio_data": null
}
```

Response shape:

```json
{
  "response_text": "string",
  "audio_url": "string|null",
  "emotion": "supportive",
  "animation_cues": {},
  "coach_feedback": {
    "safe": true,
    "summary": "string"
  }
}
```

---

## Generated Artifacts

Notebook APIs generate configuration/deployment artifacts such as:

- `Dockerfile.mas`
- `train_<agent>.slurm`
- `sparc_production.slurm`
- `launch_backend.slurm`
- `config.yml`
- `topical_rails.co`

Deployment tracks may also generate optional validation/diagnostic scripts and service unit definitions depending on notebook section.

---

## Operational Notes

- Storage and audit examples assume UF RC storage layout (`/blue` and `/pubapps`).
- Notebook examples include scaffolding and mock paths in places; adapt these before production use.
- Apply environment-specific secrets, network policy, and risk-assessment controls for deployment.
- For remediation history and evidence, refer to `QUALITY_REVIEW_BACKLOG.md`.

---

**End of API Documentation**
