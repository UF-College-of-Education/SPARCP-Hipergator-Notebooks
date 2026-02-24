# SPARC AI System Quality Review — Distilled Backlog

This backlog synthesizes reviewer notes and repository analysis into a deduplicated issue list.

Current register size: **38 issues** (expanded by integrating non-duplicate items from external condensed review lists).

## Method

- Evidence sources: quality review packet, developer revision notes, notebooks (`.ipynb`), companion docs (`.md`), env specs (`.yml`), and API reference.
- Deduplication rule: repeated findings were merged into one issue by root cause.
- Priority rule: Impact × Likelihood, with ties escalated to the higher risk tier.
- Scope rule: merged list includes corroborated code/doc issues plus cross-file conflicts that would affect delivery, compliance, or runtime reliability.

## Critical Priority

### C1 — Backend endpoint uses undefined graph object (`app_graph`)

**Why this is a real issue**
- `POST /v1/chat` invokes `app_graph.ainvoke(...)`, but no graph construction/assignment exists in the project. This is a direct runtime blocker.

**Exact locations**
- Invocation in backend doc: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L335)
- Invocation in backend notebook: [3_SPARC_RIVA_Backend.ipynb](3_SPARC_RIVA_Backend.ipynb#L434)
- No definition found across repo for `app_graph = ...` (search corroboration)

**Backlog action**
- Define canonical graph initialization lifecycle (build, inject, startup validation) and add a fail-fast health check when graph is uninitialized.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (Option B from PDF: keep existing async orchestration path and inject a canonical `app_graph` adapter).
- Implemented in notebook: [3_SPARC_RIVA_Backend.ipynb](3_SPARC_RIVA_Backend.ipynb)
	- Added `AsyncOrchestrationGraph` with `ainvoke(...)` wrapping `handle_user_turn(...)`.
	- Added `build_app_graph()` and `initialize_orchestrator()` lifecycle.
	- Added fail-fast `503` in `/v1/chat` when orchestrator is unavailable.
	- Updated `/health` to report `orchestrator_ready` and degraded status.
- Synced in companion markdown: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md)

---

### C2 — Training flow calls undefined `train_agent(...)`

**Why this is a real issue**
- Training execution cell calls `train_agent(...)` three times, but no `def train_agent(...)` exists. Notebook execution fails at this step.

**Exact locations**
- Calls in training doc: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L385-L393)
- Calls in training notebook: [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb#L474-L482)
- Existing function with different name: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L309)

**Backlog action**
- Standardize one executable training entrypoint and align all invocation cells/scripts to that entrypoint.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (aligned to PDF recommendation by standardizing on `run_qlora_training(...)`).
- Implemented in notebook: [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb)
	- Consolidated imports at top to cover `List`, `Dataset`, `BaseModel`, `ValidationError`, and `json`.
	- Replaced undefined `train_agent(...)` invocations with standardized `run_qlora_training(train_file_path, output_dir)` flow.
	- Added C2 smoke-test cell to validate imports/entrypoint and confirm no legacy dependency on `train_agent`.
- Synced in companion markdown: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md)

---

### C3 — Referenced training artifact `run_qlora_training.py` does not exist

**Why this is a real issue**
- Generated SLURM command references a script that is absent in the repository, creating a hard handoff/deployment blocker.

**Exact locations**
- Script reference in doc: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L607)
- Script not present in repo (`**/run_qlora_training.py` returns no files)

**Backlog action**
- Publish canonical script artifact policy (script present vs notebook-only execution) and enforce reference validation in deployment docs.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (Option B from PDF: notebook-only execution via `nbconvert`).
- Updated notebook SLURM generator in [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb)
	- Replaced legacy script invocation with `jupyter nbconvert --to notebook --execute 1_SPARC_Agent_Training.ipynb`.
	- Added `RUN_TRAINING=true` environment handoff so training executes from Notebook 1 without standalone `.py` files.
	- Added C3 smoke-test cell validating notebook-execution command and stale reference removal.
- Synced companion docs in [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md) and updated README training command in [README.md](README.md).

---

### C4 — Multi-adapter loading risk on shared base model in PubApps startup

**Why this is a real issue**
- Three adapters are loaded from the same `base_model` object without an explicit named-adapter strategy, creating risk of adapter state collision/override and incorrect agent behavior.

**Exact locations**
- Shared base model load: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L318)
- Adapter loads on same base: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L325-L333)

**Backlog action**
- Adopt explicit adapter-management design (`load_adapter`/`set_adapter` or isolated base instances) and add tests proving each agent uses intended weights.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (named adapter strategy on one PEFT model using `adapter_name` + `load_adapter` + `set_adapter`).
- Updated notebook startup/inference path in [4_SPARC_PubApp_Deployment.ipynb](4_SPARC_PubApp_Deployment.ipynb)
	- Replaced three shared-object adapter loads with named adapters: caregiver, coach, supervisor.
	- Added explicit adapter selection by session mode (`select_adapter_for_mode(...)` + `set_adapter(...)`).
	- Added explicit adapter restore logic after coach-feedback generation to avoid adapter bleed.
	- Updated health readiness to reflect model/tokenizer initialization state.
- Synced documentation in [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md).
- Added C4 smoke test that validates named-adapter markers and blocks legacy shared-mutation load pattern.

---

### C5 — `/v1/chat` has no in-app auth guard

**Why this is a real issue**
- The route is directly exposed with no auth dependency in handler signature, while security narrative assumes external controls; this is a high-impact control gap if gateway protections are misconfigured.

**Exact locations**
- Endpoint definition: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L371)
- Handler signature without auth guard dependency: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L372)
- Separate narrative mentions external auth layer: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L523-L664)

**Backlog action**
- Add explicit API authentication/authorization requirement and enforce defense-in-depth at application layer.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (added in-app API key guard to `/v1/chat` for defense-in-depth).
- Updated notebook backend generation in [4_SPARC_PubApp_Deployment.ipynb](4_SPARC_PubApp_Deployment.ipynb)
	- Added `require_api_key(...)` dependency using `X-API-Key` header.
	- Added env-configurable auth controls: `SPARC_API_AUTH_ENABLED`, `SPARC_API_KEY`.
	- Enforced dependency in endpoint signature (`Depends(require_api_key)`).
	- Extended `/health` payload with `api_auth_enabled` and `api_auth_configured`.
- Synced documentation in [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md).
- Extended smoke test to verify auth-guard markers and reject legacy unauthenticated handler signature.

---

### C6 — Training format risk: conversational `messages` passed directly with `packing=True`

**Why this is a real issue**
- Training setup expects chat data but uses `dataset_text_field="messages"` directly with packing, which can mis-format supervision text if chat templating is not explicitly applied.

**Exact locations**
- Chat-shaped dataset creation: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L248-L266)
- Trainer field and packing config: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L368-L370)

**Backlog action**
- Standardize supervised text rendering via explicit chat-template formatting and validate packed sample outputs before training runs.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (explicit chat rendering with `formatting_func` and pre-train render validation).
- Updated training logic in [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb)
	- Added explicit chat rendering using `tokenizer.apply_chat_template(..., tokenize=False)` with fallback string renderer.
	- Replaced `dataset_text_field="messages"` with `formatting_func=format_chat` in `SFTTrainer`.
	- Disabled risky chat packing path (`packing=False`) until explicit packing QA is introduced.
	- Added preview/render checks and packed-preview diagnostics before trainer construction.
- Synced companion documentation in [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md).
- Added C6 smoke test to validate rendered-string behavior and block legacy `dataset_text_field="messages"` usage.

## High Priority

### H1 — PHI policy contradiction: claims “transient/no disk PHI” but raw transcript is logged

**Why this is a real issue**
- Security/compliance text states transcripts are in-memory only and not written to disk, yet logging writes raw user transcript to audit file.

**Exact locations**
- PHI claim: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L353-L365)
- Raw transcript logging: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L331)

**Backlog action**
- Define redacted audit schema and retention policy; remove/replace raw content logging with compliant metadata.

---

### H2 — PubApps retention contradiction: “transient PHI” but session content is persisted

**Why this is a real issue**
- Deployment guide states transient in-memory PHI model but sample implementation stores last user message/response in Firestore.

**Exact locations**
- Retention claim: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L666)
- Persisted fields: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L444-L445)

**Backlog action**
- Resolve policy vs implementation by defining allowed stored fields, PHI classification, and TTL/deletion controls.

---

### H3 — API contract mismatch across backend tracks (`user_transcript` vs `user_message`; response schema drift)

**Why this is a real issue**
- Client integration and API docs are inconsistent, increasing integration failures and ambiguous contract ownership.

**Exact locations**
- API docs request/response contract: [API_DOCUMENTATION.md](API_DOCUMENTATION.md#L211-L275)
- PubApps implementation contract: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L341-L371)

**Backlog action**
- Establish one versioned API contract and align all notebooks/docs/examples to the same request/response shape.

---

### H4 — Model identity drift across notebooks/docs (incompatible base assumptions)

**Why this is a real issue**
- Multiple base-model identifiers appear across training/deployment variants, risking adapter incompatibility and failed loads.

**Exact locations**
- Training notebook uses Llama-2 constant: [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb#L104)
- Same notebook also uses `openai/gpt-oss-120b`: [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb#L407)
- SLURM cell references Llama-2 again: [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb#L726)
- Pixel Streaming variant assumes `gpt-oss-20b`: [4b_SPARC_PubApp_Deployment_PixelStreaming.md](4b_SPARC_PubApp_Deployment_PixelStreaming.md#L37-L228)

**Backlog action**
- Publish a model compatibility matrix (base model, adapter target, serving runtime) and version-pin each track.

---

### H5 — CORS policy is insecure/incompatible for credentialed browser calls

**Why this is a real issue**
- `allow_origins=["*"]` with `allow_credentials=True` is not a safe production posture and is incompatible with credentialed CORS expectations.

**Exact locations**
- CORS configuration: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L299-L300)

**Backlog action**
- Define trusted origin allowlist and credential strategy for production and test environments.

---

### H6 — PII sanitization currently fails open

**Why this is a real issue**
- On anonymization failure, original unsanitized text is returned; this can propagate PHI into downstream storage/retrieval/training.

**Exact locations**
- Fail-open branch: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L124-L125)

**Backlog action**
- Adopt fail-closed/quarantine handling with explicit error channel and recovery workflow.

---

### H7 — Container build path references missing `requirements.txt`

**Why this is a real issue**
- Dockerfile generation depends on `requirements.txt`, but no such file exists, so the build path is not reproducible as documented.

**Exact locations**
- Dockerfile lines expecting requirements: [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md#L60-L61)
- File absence in repo (`**/requirements.txt` returns no files)

**Backlog action**
- Define canonical dependency artifact for container build (or remove stale path from deployment guidance).

---

### H8 — PubApps resource profile conflicts with `gpt-oss-120b` serving assumption

**Why this is a real issue**
- The deployment track states a single L4 (24GB) target, while backend startup loads `gpt-oss-120b` as the base model. This is a practical deployability risk for the documented default profile.

**Exact locations**
- PubApps resource profile: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L16-L37)
- Base model load target: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L312-L320)

**Backlog action**
- Define an approved serving-model profile per hardware tier (L4 vs larger GPU) and enforce compatibility in deployment docs.

---

### H9 — Source-of-truth drift in backend launch script (`.md` vs `.ipynb`)

**Why this is a real issue**
- The markdown companion and notebook define materially different deployment commands and runtime strategy for `launch_backend.slurm`, which can cause failed or inconsistent production rollout.

**Exact locations**
- Markdown launch flow (`srun apptainer run`): [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L370-L387)
- Notebook launch flow (conda + Riva exec + uvicorn): [3_SPARC_RIVA_Backend.ipynb](3_SPARC_RIVA_Backend.ipynb#L484-L548)

**Backlog action**
- Declare a canonical artifact source and add sync checks so `.md` and `.ipynb` cannot drift for executable sections.

---

### H10 — Guardrails are configured as files but not enforced in runtime path

**Why this is a real issue**
- Guardrails config files are generated, but runtime logic uses a simple keyword check and commented-out NeMo integration, leaving safety controls weak and inconsistent.

**Exact locations**
- Guardrails files generated: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L179-L210)
- NeMo runtime import commented out: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L235)
- Keyword-only safety check: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L243)

**Backlog action**
- Implement one enforced safety pipeline (input + output) and add regression tests for bypass attempts.

---

### H11 — Async endpoint performs synchronous GPU generation (event-loop stall risk)

**Why this is a real issue**
- `async def process_chat` executes blocking model generation inline, which can degrade concurrency and service responsiveness under load.

**Exact locations**
- Async route handler: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L372)
- Blocking generation calls: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L407-L424)

**Backlog action**
- Offload blocking inference work from the event loop and add load tests proving health endpoint responsiveness.

---

### H12 — Health check can report healthy while model readiness is unknown

**Why this is a real issue**
- Health response hard-codes `"models_loaded": True` without checking model/tokenizer objects, risking false-positive readiness during startup failures.

**Exact locations**
- Health endpoint logic: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L354-L367)

**Backlog action**
- Add explicit model/tokenizer readiness checks and degraded status behavior.

---

### H13 — Raw internal exception details are returned to API clients

**Why this is a real issue**
- Error handler returns `detail=str(e)`, which can leak implementation details and infrastructure internals.

**Exact locations**
- Exception response: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L458)

**Backlog action**
- Return sanitized error payloads to clients and keep detailed diagnostics in internal logs only.

---

### H14 — Request schema lacks bounds and pattern constraints

**Why this is a real issue**
- `ChatRequest` uses unconstrained strings, allowing oversized payloads and malformed identifiers.

**Exact locations**
- Request model: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L341-L344)

**Backlog action**
- Add strict Pydantic field constraints (length/pattern) and reject invalid payloads early.

---

### H15 — Quantization setup is under-specified for deterministic low-memory startup

**Why this is a real issue**
- Model loading uses `load_in_4bit=True` directly but does not define explicit `BitsAndBytesConfig`/`quantization_config`, increasing portability risk across runtime/library variants.

**Exact locations**
- Current load pattern: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L318-L321)
- No explicit quantization config in this deployment file

**Backlog action**
- Standardize explicit quantization configuration and validate memory profile on target hardware.

## Medium Priority

### M1 — RAG ingestion path inconsistency (embedding model + directory naming drift)

**Why this is a real issue**
- Two ingestion flows use different embedding models and different persistence directory names (`vectordb` vs `vector_db`), fragmenting retrieval behavior and reproducibility.

**Exact locations**
- Flow A model/path: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L168-L171)
- Flow B model/path: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L229-L236)

**Backlog action**
- Consolidate one canonical retrieval config profile and define migration/compatibility handling for existing stores.

---

### M2 — Environment spec fragility from mixed CUDA packaging strategy

**Why this is a real issue**
- Both env files pin `cuda`, `cudatoolkit`, and `pytorch-cuda` together, increasing solver/compatibility fragility across HPC/PubApps variants.

**Exact locations**
- Training env: [environment_training.yml](environment_training.yml#L24-L31)
- Backend env: [environment_backend.yml](environment_backend.yml#L25-L30)

**Backlog action**
- Create tested, platform-specific lockfiles and documented package provenance for each runtime target.

---

### M3 — Potentially unresolvable backend dependency (`riva-asrlib-decoder`)

**Why this is a real issue**
- Dependency is listed without corroborated installation path in current setup guidance; this can break reproducible env creation.

**Exact locations**
- Package reference: [environment_backend.yml](environment_backend.yml#L67)

**Backlog action**
- Validate package availability/source, or replace with supported dependency chain and update install docs.

---

### M4 — Riva version guidance conflict (`2.16.0` vs `riva_quickstart_v2.14.0` path)

**Why this is a real issue**
- Mixed version references in setup instructions increase the chance of operator error and configuration drift.

**Exact locations**
- Version constant: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L73)
- Conflicting quickstart path text: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L116)

**Backlog action**
- Standardize one supported Riva version and align all command/path examples.

---

### M5 — Container dependency manager mismatch (Poetry narrative vs `pip -r requirements.txt` implementation)

**Why this is a real issue**
- Build strategy description says Poetry-based installation, but generated Dockerfile uses `requirements.txt` and `pip`; this creates maintenance and reproducibility confusion.

**Exact locations**
- Poetry narrative: [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md#L24-L29)
- Pip implementation lines: [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md#L60-L61)

**Backlog action**
- Pick one dependency workflow (Poetry or requirements export) and update docs/generator consistently.

---

### M6 — API documentation is stale relative to current execution tracks

**Why this is a real issue**
- API documentation asserts artifact names and Apptainer-first assumptions that diverge from the conda-oriented current notebooks and generated scripts.

**Exact locations**
- API doc claim (`train_agent.slurm` / Apptainer): [API_DOCUMENTATION.md](API_DOCUMENTATION.md#L129)
- API doc claim (`launch_backend.slurm` via `srun apptainer run`): [API_DOCUMENTATION.md](API_DOCUMENTATION.md#L241)
- API operational assumption: [API_DOCUMENTATION.md](API_DOCUMENTATION.md#L299)
- Repo migration direction toward conda: [README.md](README.md#L16)

**Backlog action**
- Re-baseline API docs against the canonical 2026 execution path and version them with the notebook release.

---

### M7 — No explicit timeout/circuit-breaker controls in core runtime calls

**Why this is a real issue**
- Inference and downstream calls are executed without explicit timeout wrappers, so hung dependencies can stall request handling.

**Exact locations**
- Core generation/TTS path without timeout guard: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L407-L440)

**Backlog action**
- Add bounded timeout/circuit-breaker policy and graceful degraded responses.

---

### M8 — Riva client objects created per request (pooling/reuse not defined)

**Why this is a real issue**
- Request path creates fresh Riva auth/service objects for each call, increasing per-request overhead.

**Exact locations**
- Per-request TTS client creation: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L435-L437)

**Backlog action**
- Define startup-initialized reusable client/session strategy and monitor latency impact.

---

### M9 — Hardcoded infrastructure/config values reduce portability and safety

**Why this is a real issue**
- Fixed absolute paths and credential locations are embedded in deployment code/docs, increasing environment drift and secret-handling risk.

**Exact locations**
- Hardcoded credentials path: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L282)
- Firebase credential load from fixed path: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L285)

**Backlog action**
- Move sensitive/runtime-specific values to environment/config layer with validation at startup.

**Resolution update (2026-02-24)**
- Status: ✅ Implemented (environment-driven infrastructure config with startup validation).
- Updated deployment notebook configuration in [4_SPARC_PubApp_Deployment.ipynb](4_SPARC_PubApp_Deployment.ipynb)
	- Added environment-driven base path and endpoints: `SPARC_BASE_PATH`, `SPARC_HIPERGATOR_SOURCE_MODELS`, `SPARC_PUBAPPS_SSH_USER`, `SPARC_PUBAPPS_HOST`.
	- Replaced fixed rsync target construction with config-derived values.
- Updated generated backend config in [4_SPARC_PubApp_Deployment.ipynb](4_SPARC_PubApp_Deployment.ipynb)
	- Replaced fixed runtime values with env-backed keys: `SPARC_MODEL_BASE_PATH`, `SPARC_RIVA_SERVER`, `SPARC_FIREBASE_CREDS`.
	- Added startup validation for Firebase credentials path (`FIREBASE_CREDS` non-empty and file existence check) before SDK initialization.
- Synced companion documentation in [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md) and expanded smoke-test coverage to include M9 config markers.

---

### M10 — Podman GPU guidance uses fixed device mapping without CDI profile abstraction

**Why this is a real issue**
- Quadlet example hardcodes individual NVIDIA device nodes, which is brittle across host configurations.

**Exact locations**
- Quadlet GPU device mappings: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L213-L215)

**Backlog action**
- Standardize supported GPU passthrough profile for PubApps Podman runtime and document validation steps.

---

### M11 — TTS payload delivery uses inline base64 WAV in JSON response

**Why this is a real issue**
- Large inline audio payloads increase response size and memory pressure on API clients/servers.

**Exact locations**
- Inline base64 response composition: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L439)

**Backlog action**
- Define bounded audio-delivery strategy (streaming/chunking/object URL) with payload size limits.

## Low Priority

### L1 — Operational policy ambiguity for unlimited SLURM runtime in examples

**Why this is a real issue**
- Example uses `#SBATCH --time=UNLIMITED`; if not permitted by partition/QoS policy, users will hit avoidable job submission failures.

**Exact locations**
- Runtime directive: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L379)

**Backlog action**
- Replace with policy-compliant defaults and document when/if unlimited runtime is permitted.

---

### L2 — Container runtime version text drift (`python:3.10-slim` in prose vs `3.11-slim` in Dockerfile)

**Why this is a real issue**
- Version mismatch in narrative vs generated artifact increases onboarding errors and troubleshooting overhead.

**Exact locations**
- Prose says `python:3.10-slim`: [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md#L30)
- Dockerfile template uses `python:3.11-slim`: [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md#L50-L66)

**Backlog action**
- Align narrative and generated artifact versions to a single pinned runtime.

---

### L3 — Audit log directory existence is not guaranteed before logger initialization

**Why this is a real issue**
- Logging is configured to a fixed file path, but directory creation is not shown in the setup path; this can cause first-run failures depending on environment state.

**Exact locations**
- Log file path + logger setup: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L310-L311)

**Backlog action**
- Add explicit log-directory creation/permission validation to startup preflight guidance.

---

### L4 — `build_vector_store()` has side effects only and no return contract

**Why this is a real issue**
- Function persists vectors and prints status but does not return the created object/path, reducing composability and testability.

**Exact locations**
- Function declaration: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L147)
- Terminal side-effect line: [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md#L177)

**Backlog action**
- Define a return contract for downstream use and validation tests.

---

### L5 — FastAPI startup hook uses deprecated event pattern

**Why this is a real issue**
- Startup initialization uses `@app.on_event("startup")`, which is a deprecating pattern in modern FastAPI lifecycle guidance.

**Exact locations**
- Startup hook usage: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md#L306)

**Backlog action**
- Migrate startup/shutdown orchestration to lifespan-based app lifecycle.

---

### L6 — Guardrails config output path is implicit to current working directory

**Why this is a real issue**
- Config generation writes files to relative CWD, which can vary by execution context and break predictable deployment.

**Exact locations**
- Relative config writes: [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md#L187-L205)

**Backlog action**
- Use explicit, configured output directories for guardrails artifacts.

## Deduplication Notes

- Repeated PHI contradiction findings across backend and deployment docs were merged into H1/H2.
- Repeated model inconsistency findings across notebook/doc variants were merged into H4.
- Repeated runtime breakage findings around missing orchestration/training entrypoints were merged into C1/C2/C3.

## PDF Verbatim Quote Mapping (All 38 Issues)

Source: [SPARC_AI_System_Quality_Review__PDF_EXTRACT.txt](SPARC_AI_System_Quality_Review__PDF_EXTRACT.txt) extracted from [SPARC AI System - Quality Review (1).pdf](SPARC%20AI%20System%20-%20Quality%20Review%20(1).pdf).

Note: Each issue below includes the full available PDF detail fields: **Finding**, **Issue Identified**, and **Possible Solution** (or **Not set in PDF row** when absent).

### Critical

- **C1**
	- **Finding (PDF):** "app_graph undefined — /v1/chat crashes with NameError." (Page 9)
	- **Issue Identified (PDF):** "NB3 Sec 6.1 chat_endpoint(): calls await app_graph.ainvoke(initial_state) but no StateGraph is ever built or compiled anywhere in the notebook. handle_user_turn() in Sec 5.1 wires agents via asyncio but is never called from the endpoint — it is dead code. Endpoint raises NameError at runtime."
	- **Possible Solution (PDF):** "Option A: Build a LangGraph StateGraph (supervisor_check, caregiver_respond, coach_evaluate nodes) and compile to app_graph. Option B (simpler): replace app_graph.ainvoke() with the existing handle_user_turn() call and update docs to reflect asyncio orchestration."

- **C2**
	- **Finding (PDF):** "Missing imports cause NameError on cell execution." (Page 17)
	- **Issue Identified (PDF):** "NB1: List[str] without typing import (Sec 4.3), Dataset without import (Sec 4.2), BaseModel/ValidationError without pydantic import (Sec 6.2), import json missing (Secs 6.2 and 8.0), train_agent() called but only run_qlora_training() defined (Sec 5.2)."
	- **Possible Solution (PDF):** "Add consolidated import cell at top of NB1 with all required imports. Rename train_agent() calls to run_qlora_training()."

- **C3**
	- **Finding (PDF):** "SLURM script references wrong filename train_agent.py." (Page 16)
	- **Issue Identified (PDF):** "NB1 Sec 7.1 generate_slurm_script(): SLURM calls python train_agent.py but actual training function is run_qlora_training() in Sec 5.0. No train_agent.py exists. Also incorrectly referenced in README.md."
	- **Possible Solution (PDF):** "Option A: Create train_agent.py as a CLI wrapper with argparse. Option B: Update SLURM to use jupyter nbconvert --to script --execute."

- **C4**
	- **Finding (PDF):** "LoRA adapters corrupt each other on shared base model." (Page 9)
	- **Issue Identified (PDF):** "NB4 Sec 6.1 load_models(): PeftModel.from_pretrained(base_model, ...) mutates base_model in-place on each call. Each call overwrites prior adapter weights. All 3 variables point to the same mutated object — only the last (Supervisor) adapter is active. Caregiver and Coach silently use wrong weights at inference."
	- **Possible Solution (PDF):** "Use PeftModel named adapters: model = PeftModel.from_pretrained(base, caregiver_path, adapter_name='caregiver'), then model.load_adapter(coach_path, adapter_name='coach'). Switch at inference with model.set_adapter('caregiver')."

- **C5**
	- **Finding (PDF):** "No API authentication on v1/chat." (Page 22)
	- **Issue Identified (PDF):** "FastAPI endpoint has no authentication. Anyone who can reach port 8000 on the PubApps server can call the LLM. That means free GPU access for anyone, and the ability to inject fake sessions into Firebase."
	- **Possible Solution (PDF):** "Not set in PDF row."

- **C6**
	- **Finding (PDF):** "SFTTrainer receives list-of-dicts instead of string — training data corrupted." (Pages 15–16)
	- **Issue Identified (PDF):** "NB1 Sec 5.0 run_qlora_training(): dataset_text_field='messages' expects plain text strings. The messages field is a list-of-dicts in ChatML format. SFTTrainer either errors or stringifies the list, producing corrupted training data. packing=True further corrupts conversational data."
	- **Possible Solution (PDF):** "Use formatting_func: def format_chat(ex): return tokenizer.apply_chat_template(ex['messages'], tokenize=False). Pass formatting_func=format_chat to SFTTrainer."

### High

- **H1**
	- **Finding (PDF):** "Audit log writes PHI to disk — HIPAA violation." (Page 22)
	- **Issue Identified (PDF):** "NB3 Sec 6.1: logs full user_transcript to audit.log on disk. Sec 7.0 of same notebook states 'No PHI is written to disk.' Direct contradiction."
	- **Possible Solution (PDF):** "Remove {request.user_transcript} from all log lines. Log only: session_id, timestamp, agent_type, is_safe flag, latency_ms."

- **H2**
	- **Finding (PDF):** "Volatile PHI model vs. training/fine-tuning goal." (Page 7)
	- **Issue Identified (PDF):** "State persistence is not fully defined beyond 'volatile PHI model.' We process PHI in memory only and do not store it. session_state['last_user_message'] = request.user_message. session_state['last_response'] = response_text. session_ref.set(session_state, merge=True)."
	- **Possible Solution (PDF):** "Separate temporary session state from anonymized structured performance DB." (Page 7)

- **H3**
	- **Finding (PDF):** "Schema enforcement at API layer and may cause runtime breakage risk (e.g: keyerror, datatype mismatch, parsing issues)." (Page 18)
	- **Issue Identified (PDF):** "Missing API Schema Enforcement: The API layer lacks strict schema validation, creating a high risk of runtime breakage due to malformed JSON, datatypes mismatches, or KeyErrors." (PDF findings summary)
	- **Possible Solution (PDF):** "Define datatypes and constraints in classes." (Page 18)

- **H4**
	- **Finding (PDF):** "Incorrect base model set." (Page 7)
	- **Issue Identified (PDF):** "Currently set as gpt-oss-120b in 1_SPARC_Agent_Training.ipynb. NVIDIA L4 GPU is unable to handle 120b version."
	- **Possible Solution (PDF):** "Will need to configure to use gpt-oss-20b or other open source models."

- **H5**
	- **Finding (PDF):** "CORS allows all origins with credentials — startup error." (Page 23)
	- **Issue Identified (PDF):** "NB4 Sec 6.1: allow_origins=['*'] + allow_credentials=True is invalid per CORS spec. Starlette 0.27+ raises ValueError at startup."
	- **Possible Solution (PDF):** "Set allow_origins=['https://sparc-p.rc.ufl.edu']. Remove allow_credentials=True. Restrict allow_methods=['GET','POST'] and allow_headers."

- **H6**
	- **Finding (PDF):** "PII sanitization fails open — unsanitized text enters model weights." (Page 22)
	- **Issue Identified (PDF):** "NB1 Sec 4.2 sanitize_text_with_presidio(): except Exception: return text — returns ORIGINAL text on any Presidio error. PII flows into ChromaDB and LoRA training data. Once embedded in weights, cannot be removed post-deployment."
	- **Possible Solution (PDF):** "Fail closed: return '' or raise on error. Log the error. Add tenacity retry before failing. Never pass unsanitized text downstream."

- **H7**
	- **Finding (PDF):** "Dockerfile COPY commands reference wrong paths — container builds fail." (Page 16)
	- **Issue Identified (PDF):** "NB2b Secs 2.2-2.4: Dockerfile.mas uses COPY requirements.txt — no requirements.txt exists. Dockerfile.unity-server uses COPY Build/LinuxServer/ — wrong path. Dockerfile.signaling uses COPY signaling/ — also wrong. All three podman build commands fail with COPY failed: file not found. NB4b Sec 4.2 references these same broken images."
	- **Possible Solution (PDF):** "Create missing artifacts: (1) generate requirements.txt from pip deps, (2) add Unity Linux server build output, (3) add signaling server source. Document the build pipeline."

- **H8**
	- **Finding (PDF):** "Hardware & Model Mismatch." (PDF findings summary)
	- **Issue Identified (PDF):** "The backend training notebook sets the base model to gpt-oss-120b, which cannot run on the 24GB NVIDIA L4 GPU."
	- **Possible Solution (PDF):** "Will need to configure to use gpt-oss-20b or other open source models."

- **H9**
	- **Finding (PDF):** "Compile ipynb into plain py files." (Page 14)
	- **Issue Identified (PDF):** "Jupyter Notebook files will work fine especially in development, but they can execute in an order that is not intuitive."
	- **Possible Solution (PDF):** "Several possibilities. Jupytext may be a good option because it allows py and ipynb versions of code to sync back and forth rather than having to manually upkeep the sync."

- **H10**
	- **Finding (PDF):** "NeMo Guardrails configured but never loaded." (Page 9)
	- **Issue Identified (PDF):** "Config files exist (NB3 Sec 3.2). In NB3 Sec 5.1, NeMo import is commented out. SupervisorAgent uses is_safe = 'politics' not in text.lower(). NB4 Sec 6.1 uses a 5-word keyword blocklist. NeMo in environment_backend.yml but never instantiated."
	- **Possible Solution (PDF):** "Uncomment NeMo import in NB3 Sec 5.1. Add to SupervisorAgent.__init__: self.rails = LLMRails(RailsConfig.from_path('./guardrails/')). Replace keyword check with rails generate path. Repeat in NB4 Sec 6.1."

- **H11**
	- **Finding (PDF):** "async process_chat() blocks event loop during inference." (Page 23)
	- **Issue Identified (PDF):** "NB4 Sec 6.1: async def but model.generate() is synchronous and GPU-bound. Blocks entire asyncio event loop during inference. Server completely unresponsive to all requests including /health during every call."
	- **Possible Solution (PDF):** "Change to regular def (uvicorn runs sync handlers in threadpool), or wrap with await asyncio.to_thread(model.generate, **kwargs)."

- **H12**
	- **Finding (PDF):** "Health check reports healthy when models fail to load." (Page 23)
	- **Issue Identified (PDF):** "NB4 Sec 6.1 /health: checks Riva connectivity but not LLM model readiness. If load_models() fails, tokenizer/model remain None but endpoint returns status:healthy. Load balancers route traffic to broken instance."
	- **Possible Solution (PDF):** "Add: model_ok = tokenizer is not None and caregiver_model is not None. Return HTTP 503 or status:'degraded' if models not loaded."

- **H13**
	- **Finding (PDF):** "Raw exception details leak to client on inference failure." (Page 23)
	- **Issue Identified (PDF):** "NB3, NB4: model.generate() failure raises HTTPException(500, detail=str(e)). Internal file paths, CUDA errors, Python stack traces sent directly to client — information disclosure vulnerability."
	- **Possible Solution (PDF):** "Catch inference exceptions and return generic: HTTPException(500, detail='Internal server error'). Log actual exception server-side only."

- **H14**
	- **Finding (PDF):** "ChatRequest missing field constraints — DoS and log injection." (Page 23)
	- **Issue Identified (PDF):** "NB3, NB4: user_message/user_transcript defined as plain str with no max_length. Multi-MB strings exhaust RAM during tokenization. session_id accepts newlines (log injection) and ../ path traversal. audio_data (NB4) accepts unbounded base64 strings."
	- **Possible Solution (PDF):** "Add Pydantic Field validators: user_message: str = Field(..., max_length=10000); session_id: str = Field(..., max_length=128, pattern=r'^[a-zA-Z0-9_-]+$')."

- **H15**
	- **Finding (PDF):** "load_in_4bit not a valid from_pretrained parameter — model loads at full precision." (Page 16)
	- **Issue Identified (PDF):** "NB4 Sec 6.1 load_models(): from_pretrained(name, load_in_4bit=True) — not a valid kwarg. Model loads at full precision → OOM on L4 24 GB. NB1 Sec 5.0 does this correctly with quantization_config=bnb_config."
	- **Possible Solution (PDF):** "Use BitsAndBytesConfig: bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16). Pass quantization_config=bnb to from_pretrained."

### Medium

- **M1**
	- **Finding (PDF):** "Duplicate RAG ingestion with incompatible embeddings." (Pages 9–10)
	- **Issue Identified (PDF):** "NB1 build_vector_store() (Sec 4.3) uses all-MiniLM-L6-v2 (384-dim) persisting to vectordb/. ingest_documents() (Sec 4.1) uses all-mpnet-base-v2 (768-dim) persisting to vector_db/. Different dimensions = incompatible vectors. Different directory names = data in two separate locations. Wrong function at query time produces silently incorrect retrieval."
	- **Possible Solution (PDF):** "Remove build_vector_store(). Keep ingest_documents() with all-mpnet-base-v2. Standardize persist dir to OUTPUT_DIR/vector_db/. Use same embedding model at index-build and query time."

- **M2**
	- **Finding (PDF):** "cudatoolkit=12.8 does not exist — conda env creation fails." (Page 15)
	- **Issue Identified (PDF):** "environment_training.yml, environment_backend.yml: cudatoolkit=12.8 does not exist as a conda package (max ~11.8 on conda-forge/nvidia). Also redundant with cuda=12.8 already listed. conda env create fails with unsatisfiable solver error before any packages install. Blocks all project setup."
	- **Possible Solution (PDF):** "Remove cudatoolkit=12.8 from both YAMLs. Keep only cuda=12.8 from nvidia channel. pytorch-cuda=12.8 provides the necessary CUDA runtime."

- **M3**
	- **Finding (PDF):** "riva-asrlib-decoder not on public PyPI — backend env fails." (Page 15)
	- **Issue Identified (PDF):** "environment_backend.yml: pip dependency riva-asrlib-decoder is NVIDIA-internal, not on public PyPI. pip install fails with 'No matching distribution found.' Backend conda env cannot be created."
	- **Possible Solution (PDF):** "Remove from pip deps. It is bundled inside the Riva container image and not needed in the host conda env."

- **M4**
	- **Finding (PDF):** "Riva version mismatch between setup cells." (Page 17)
	- **Issue Identified (PDF):** "NB3 Sec 2.3 defines RIVA_VERSION='2.16.0' pulling riva-speech:2.16.0-server. Sec 2.2 references riva_quickstart_v2.14.0/config.sh. Incompatible quickstart scripts for the pulled server version."
	- **Possible Solution (PDF):** "Standardize on one Riva version throughout — update all references to match RIVA_VERSION."

- **M5**
	- **Finding (PDF):** "Direct pip not recommended for package management on HiPerGator." (Pages 14–15)
	- **Issue Identified (PDF):** "Because pip installs packages into shared or user-level directories, it can conflict with packages installed elsewhere in the HiPerGator environment, resulting in incorrect package versions and other unexpected behaviors."
	- **Possible Solution (PDF):** "Conda environments should be used to isolate project dependencies."

- **M6**
	- **Finding (PDF):** "Review Scope ... 8. Documentation completeness." (Page 1)
	- **Issue Identified (PDF):** "Evaluation Criteria ... Documentation clarity." (Page 26)
	- **Possible Solution (PDF):** "Recommended Output Format: Each reviewer will submit ... categorized findings ... specific code references ... concrete recommendations. After individual reviews are submitted: findings will be consolidated and action items will be prioritized." (Page 26)

- **M7**
	- **Finding (PDF):** "No defined timeout or circuit-breaker for downstream agent calls." (Page 9)
	- **Issue Identified (PDF):** "If the Parent Agent or Riva TTS hangs, the entire session will stall with no fallback."
	- **Possible Solution (PDF):** "Add asyncio.wait_for() timeouts on all agent calls (e.g., 10s for LLM, 5s for Riva). On timeout, the Supervisor should route a graceful fallback response and flag the incident in the audit log."

- **M8**
	- **Finding (PDF):** "Riva gRPC connection created per-request — no pooling." (Page 24)
	- **Issue Identified (PDF):** "NB4 Sec 6.1: New riva.client.Auth and SpeechSynthesisService created on every /v1/chat request. Adds TCP setup latency per request and leaks file descriptors since channels never explicitly closed."
	- **Possible Solution (PDF):** "Create Auth and service objects once in load_models() at startup. Reuse across all requests."

- **M9**
	- **Finding (PDF):** "Hardcoded paths in 30+ locations." (Page 16)
	- **Issue Identified (PDF):** "All notebooks: /blue/jasondeanarnold/SPARCP/ hardcoded across all notebooks. jayrosen@ufl.edu in SLURM scripts. pubapps-vm.rc.ufl.edu in NB4. No other team member can run without manual find-and-replace."
	- **Possible Solution (PDF):** "Config cell at top of each notebook: BASE_PATH = os.environ.get('SPARC_BASE_PATH', '/blue/jasondeanarnold/SPARCP'). SLURM_EMAIL = os.environ.get('SPARC_SLURM_EMAIL', 'jayrosen@ufl.edu')."

- **M10**
	- **Finding (PDF):** "Riva container uses wrong GPU passthrough — CUDA init fails." (Page 23)
	- **Issue Identified (PDF):** "NB4 Sec 5.1 Quadlet: raw AddDevice=/dev/nvidia0 instead of CDI Device=nvidia.com/gpu=all. NVIDIA runtime libraries invisible inside Podman container. Riva fails with CUDA init error at startup."
	- **Possible Solution (PDF):** "Replace AddDevice lines with Device=nvidia.com/gpu=all. Requires nvidia-container-toolkit CDI on PubApp VM."

- **M11**
	- **Finding (PDF):** "TTS audio as inline base64 WAV — unbounded response size." (Page 24)
	- **Issue Identified (PDF):** "NB4 Sec 6.1: 30-second WAV at 22kHz/16-bit ~1.7 MB base64 embedded in JSON response body. Concurrent requests on 16 GB RAM server create significant memory pressure."
	- **Possible Solution (PDF):** "Use compressed format (opus/mp3), store in temp file and return URL, or stream over WebSocket."

### Low

- **L1**
	- **Finding (PDF):** "SLURM --time=UNLIMITED invalid format." (Page 17)
	- **Issue Identified (PDF):** "NB2 Sec 4.2: #SBATCH --time=UNLIMITED is not a valid SLURM time format. Scheduler rejects unless QOS explicitly permits it."
	- **Possible Solution (PDF):** "Replace with #SBATCH --time=7-00:00:00 or the maximum allowed by the QOS."

- **L2**
	- **Finding (PDF):** "Phase 2: Code Quality & Maintainability Review." (Page 14)
	- **Issue Identified (PDF):** "Evaluate: Code organization and folder structure; Naming conventions and readability; Presence of modular design patterns; Presence of technical debt; Duplication of logic; Hard-coded logic vs. configurable logic; Test coverage; Error handling; Logging consistency."
	- **Possible Solution (PDF):** "Immediate stabilization recommendations: if .py files used: backend/main.py, supervisor.py, coach.py, parent.py, schemas.py, scoring.py, safety.py."

- **L3**
	- **Finding (PDF):** "No error logging: Currently logging only user input, not errors or exceptions." (Page 19)
	- **Issue Identified (PDF):** "Sensitive data exposure: logging full user transcripts can expose sensitive information + unauthorized data leak risk."
	- **Possible Solution (PDF):** "Add structured logging (JSON logs), Request ID per session, and try/except error logging with sanitized transcript handling."

- **L4**
	- **Finding (PDF):** "build_vector_store() missing return statement." (Page 17)
	- **Issue Identified (PDF):** "NB1 Sec 4.3: Creates Chroma object but has no return statement. Object lost on exit. Also missing collection_name param unlike ingest_documents()."
	- **Possible Solution (PDF):** "Add return vector_store. Add collection_name=collection_name to Chroma.from_documents() call."

- **L5**
	- **Finding (PDF):** "Deprecated FastAPI startup event handler." (Page 24)
	- **Issue Identified (PDF):** "NB4 Sec 6.1: @app.on_event('startup') deprecated since FastAPI 0.93+. Deprecation warnings generated; will be removed in a future version."
	- **Possible Solution (PDF):** "Migrate to lifespan handler using @asynccontextmanager and FastAPI(lifespan=lifespan)."

- **L6**
	- **Finding (PDF):** "Guardrails config written to unpredictable directory." (Page 18)
	- **Issue Identified (PDF):** "NB3 Sec 3.2 create_rails_config(): config.yml and topical_rails.co written to notebook CWD. May overwrite unrelated files."
	- **Possible Solution (PDF):** "Write to os.path.join(OUTPUT_DIR, 'guardrails', 'config.yml') with makedirs exist_ok=True."

