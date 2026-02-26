# SPARC-P: Standardized Patient Assessment & Remediation System

SPARC-P is a multi-agent digital human training platform for clinician communication practice. This repository is the operational notebook and documentation workspace for training, backend orchestration, and deployment across UF HiPerGator and UF PubApps.

It is designed to be the implementation companion to the Unity client stack and focuses on model training, speech pipeline integration, runtime safety, API behavior, and deployment operations.

---

## Current Project Baseline (February 2026)

This repository follows a conda-first UF RC workflow and includes hardening updates across runtime, API, and deployment paths.

### Core outcomes now reflected in the notebooks and guides

- Conda is the canonical environment path for training and backend execution on UF infrastructure.
- Training execution handoff is notebook-driven (`nbconvert` path) instead of relying on missing standalone script artifacts.
- API contract alignment is standardized around a canonical v1 shape and shared request/response expectations.
- Runtime security posture now includes in-app auth guard support, trusted-origin CORS strategy, and enforced guardrails flow.
- Safety and privacy controls now emphasize sanitized persistence/logging paths and fail-closed handling in key pipelines.
- Operational resilience includes readiness-aware health semantics, async offloading for blocking generation paths, and bounded audio delivery patterns.

For detailed implementation evidence and issue-level traceability, see [QUALITY_REVIEW_BACKLOG.md](QUALITY_REVIEW_BACKLOG.md) and [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md).

---

## Quick Links

- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) — conda migration and environment transition guide
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) — canonical API reference and contract details
- [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md) — model training and data pipeline
- [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md) — baseline container/deployment path
- [2b_SPARC_Containerization_and_Deployment.md](2b_SPARC_Containerization_and_Deployment.md) — Pixel Streaming container/deployment variant
- [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md) — real-time backend runtime and orchestration
- [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md) — baseline PubApps deployment (WebGL path)
- [4b_SPARC_PubApp_Deployment_PixelStreaming.md](4b_SPARC_PubApp_Deployment_PixelStreaming.md) — PubApps Pixel Streaming deployment path

---

## What This Repository Contains

The repository combines executable notebooks (`.ipynb`) and companion markdown (`.md`) guides for each major pipeline stage:

1. Training and RAG preparation
2. Containerization and deployment packaging
3. Real-time backend runtime behavior
4. Public serving deployment on PubApps

The two deployment variants are intentionally maintained:

- **WebGL baseline path** for conventional browser delivery.
- **Pixel Streaming path** for server-side rendering thin-client deployments.

---

## Architecture Overview

### Functional roles

- **Caregiver agent**: Simulated caregiver interaction and persona behavior.
- **Coach agent**: Communication performance feedback and rubric-oriented coaching.
- **Supervisor agent**: Orchestration, safety policy routing, and final response assembly.

### Platform split

- **HiPerGator**: Training, dataset processing, model adaptation, backend validation workflows.
- **PubApps**: Persistent serving, reverse proxy/public access, and production runtime services.

### Primary technology stack

- **Model adaptation**: QLoRA + PEFT workflows
- **RAG**: Chroma-based vector retrieval pipeline
- **Speech**: NVIDIA Riva ASR/TTS integration
- **Safety**: NeMo Guardrails runtime checks
- **API**: FastAPI backend with typed request/response contracts
- **Containers**:
  - HiPerGator: Apptainer for selected workflows
  - PubApps: Podman and systemd user services

---

## Notebook and Document Map

### Training and preparation

- [1_SPARC_Agent_Training.ipynb](1_SPARC_Agent_Training.ipynb) / [1_SPARC_Agent_Training.md](1_SPARC_Agent_Training.md)
  - Data preparation and sanitization
  - RAG/vector-store preparation
  - QLoRA training and validation flows

### Packaging and deployment prep

- [2_SPARC_Containerization_and_Deployment.ipynb](2_SPARC_Containerization_and_Deployment.ipynb) / [2_SPARC_Containerization_and_Deployment.md](2_SPARC_Containerization_and_Deployment.md)
  - Baseline packaging and deployment workflow
- [2b_SPARC_Containerization_and_Deployment.ipynb](2b_SPARC_Containerization_and_Deployment.ipynb) / [2b_SPARC_Containerization_and_Deployment.md](2b_SPARC_Containerization_and_Deployment.md)
  - Pixel Streaming-oriented packaging variant

### Backend runtime

- [3_SPARC_RIVA_Backend.ipynb](3_SPARC_RIVA_Backend.ipynb) / [3_SPARC_RIVA_Backend.md](3_SPARC_RIVA_Backend.md)
  - Backend orchestration, guardrails integration, runtime controls, and launch flow

### PubApps deployment

- [4_SPARC_PubApp_Deployment.ipynb](4_SPARC_PubApp_Deployment.ipynb) / [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md)
  - Baseline PubApps deployment path
- [4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb](4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb) / [4b_SPARC_PubApp_Deployment_PixelStreaming.md](4b_SPARC_PubApp_Deployment_PixelStreaming.md)
  - Pixel Streaming variant for server-side rendering

---

## Architecture Diagrams by Notebook

(Work in progress section)
Some diagrams may reflect earlier wording or flow details, but they remain useful for conceptual understanding.

### Notebook 1 — Agent Training

#### 1.1 Introduction and System Purpose
![Introduction](images/notebook%201%20-%20section%201.png)

#### 1.2 System Configuration
![System Config](images/notebook%201%20-%20section%203.3.png)

#### 1.3 Data Pipeline
![Data Pipeline](images/notebook%201%20-%20section%204.png)

#### 1.4 Fine-Tuning (QLoRA)
![QLoRA](images/notebook%201%20-%20section%205.png)

#### 1.5 Validation
![Validation](images/notebook%201%20-%20section%206.png)

#### 1.6 Interfaces
![Interfaces](images/notebook%201%20-%20section%207-8.png)

### Notebook 2 — Containerization & Deployment

#### 2.1 Objectives
![Deployment Objectives](images/notebook%202%20-%20section%201.png)

#### 2.2 Container Build Strategy
![Build Strategy](images/notebook%202%20-%20section%202.png)

#### 2.3 Local Development (Podman)
![Podman](images/notebook%202%20-%20section%203.png)

#### 2.4 Production Deployment
![Production Deployment](images/notebook%202%20-%20section%204.png)

### Notebook 3 — Real-Time Backend

#### 3.1 Runtime Objectives
![Runtime Goals](images/notebook%203%20-%20section%201.png)

#### 3.2 Riva & Guardrails
![Riva Setup](images/notebook%203%20-%20section%202-3.png)

#### 3.3 Multi-Agent Orchestration
![Orchestration](images/notebook%203%20-%20section%205.png)

#### 3.4 API Server Integration
![API Server](images/notebook%203%20-%20section%206.png)

#### 3.5 Security and Compliance
![Security](images/notebook%203%20-%20section%207.png)

---

## Environment and Dependency Model

### Canonical environments

- [environment_training.yml](environment_training.yml): Training and adaptation environment
- [environment_backend.yml](environment_backend.yml): Backend and serving-oriented environment
- [setup_conda_env.sh](setup_conda_env.sh): Automated conda environment setup helper

### Dependency artifacts

- [requirements.txt](requirements.txt) is used where container build workflows require pip-based dependency resolution.
- Conda remains the canonical host/runtime environment path for UF RC workflows.

For migration guidance and compatibility notes, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

---

## End-to-End Workflow

### Phase 1 — Train and validate on HiPerGator

1. Prepare conda environments using [setup_conda_env.sh](setup_conda_env.sh) or manual conda creation from environment files.
2. Execute Notebook 1 to run data sanitization, RAG prep, and QLoRA adaptation.
3. Optionally use Notebook 2/2b for packaging paths tied to your deployment variant.
4. Execute Notebook 3 to validate backend orchestration and runtime behavior before public deployment.

### Phase 2 — Deploy to PubApps

1. Provision PubApps resources and complete required UF RC risk/compliance steps.
2. Transfer trained model artifacts from HiPerGator storage to PubApps storage.
3. Stand up runtime services (Riva/backend and related support services) using the selected deployment path:
   - [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md) for WebGL baseline
   - [4b_SPARC_PubApp_Deployment_PixelStreaming.md](4b_SPARC_PubApp_Deployment_PixelStreaming.md) for Pixel Streaming
4. Validate health/readiness behavior, service logs, and end-to-end client interactions.

---

## Choosing Between WebGL and Pixel Streaming

### Use Notebook 4 (WebGL baseline) when

- Browser clients can run rendering locally.
- You want the simpler baseline deployment topology.
- You do not need server-side Unity rendering.

### Use Notebook 4b (Pixel Streaming) when

- You need thin-client delivery with server-side rendering.
- Browser/device GPU capability is limited or inconsistent.
- You are prepared to operate signaling + streamed rendering infrastructure.

---

## API and Runtime Contract Summary

The backend tracks use a canonical versioned API contract documented in [API_DOCUMENTATION.md](API_DOCUMENTATION.md).

At a high level:

- Request payloads are schema-constrained and validated before orchestration.
- Response payloads are structured and include assistant output plus speech-related fields.
- Health endpoints reflect readiness state rather than assuming successful model load.
- Deployment templates include guardrails/auth/cors controls suitable for production hardening.

For implementation details, endpoint definitions, and schema fields, refer to [API_DOCUMENTATION.md](API_DOCUMENTATION.md) and the active deployment guide for your chosen path.

---

## Security, Privacy, and Compliance Posture

SPARC-P is operated with a transient-processing approach and explicit safeguards in the application/runtime path.

### Current posture highlights

- **Guardrails enforcement**: Input/output moderation and policy checks are integrated into runtime orchestration paths.
- **Auth strategy**: External controls (for example, gateway/SSO) can be complemented with in-app authentication guard support.
- **CORS strategy**: Trusted-origin allowlist behavior is preferred over wildcard production settings.
- **Sanitization controls**: Pipelines emphasize fail-closed behavior and sanitized storage/logging paths.
- **Audit model**: Operational metadata is retained for observability/compliance needs without relying on raw sensitive payload logging as the default.

Always align final deployment controls with UF RC policy, institutional requirements, and your approved risk assessment.

---

## Operations and Reliability Notes

Recent repository updates improve runtime reliability and operator visibility:

- Readiness-aware health behavior for startup and degraded states.
- Async-safe offloading for blocking model generation paths.
- Timeout/circuit-breaker posture in critical runtime call paths.
- Bounded speech artifact delivery patterns for safer client payload handling.

Use the deployment notebooks and generated scripts to validate these controls in your target environment.

---

## Getting Started (Minimal Commands)

The exact commands depend on your UF group path and allocation setup, but the sequence is:

```bash
# 1) Clone repository
git clone https://github.com/UF-College-of-Education/SPARCP-Hipergator-Notebooks.git
cd SPARCP-Hipergator-Notebooks

# 2) Create conda environments
bash setup_conda_env.sh both

# 3) Run Notebook 1 execution path (example)
jupyter nbconvert --to notebook --execute 1_SPARC_Agent_Training.ipynb --output executed_1_SPARC_Agent_Training.ipynb
```

For complete environment and migration steps, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md).

---

## Repository Structure

```text
Sparc Hipergator Notebooks/
├── README.md
├── API_DOCUMENTATION.md
├── MIGRATION_GUIDE.md
├── QUALITY_REVIEW_BACKLOG.md
├── IMPLEMENTATION_SUMMARY.md
├── environment_training.yml
├── environment_backend.yml
├── requirements.txt
├── setup_conda_env.sh
├── 1_SPARC_Agent_Training.md
├── 1_SPARC_Agent_Training.ipynb
├── 2_SPARC_Containerization_and_Deployment.md
├── 2_SPARC_Containerization_and_Deployment.ipynb
├── 2b_SPARC_Containerization_and_Deployment.md
├── 2b_SPARC_Containerization_and_Deployment.ipynb
├── 3_SPARC_RIVA_Backend.md
├── 3_SPARC_RIVA_Backend.ipynb
├── 4_SPARC_PubApp_Deployment.md
├── 4_SPARC_PubApp_Deployment.ipynb
├── 4b_SPARC_PubApp_Deployment_PixelStreaming.md
├── 4b_SPARC_PubApp_Deployment_PixelStreaming.ipynb
├── images/
├── instructions/
├── training_data/
└── trained_models/
```

---

## Troubleshooting Entry Points

- Conda/environment issues: [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- API and schema behavior: [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- Baseline deployment troubleshooting: [4_SPARC_PubApp_Deployment.md](4_SPARC_PubApp_Deployment.md)
- Pixel Streaming deployment troubleshooting: [4b_SPARC_PubApp_Deployment_PixelStreaming.md](4b_SPARC_PubApp_Deployment_PixelStreaming.md)
- Change history and remediation details: [QUALITY_REVIEW_BACKLOG.md](QUALITY_REVIEW_BACKLOG.md)

---

## Version Snapshot

### v2.0 (February 2026)

- Conda-first migration across operational workflows
- PubApps deployment guidance for baseline and Pixel Streaming variants
- API/runtime hardening and documentation re-baseline

### v1.0 (2025)

- Initial training + backend + deployment notebook release

---

## Support

- UF RC support portal: https://support.rc.ufl.edu/
- PubApps documentation: https://docs.rc.ufl.edu/services/web_hosting/
- Conda on HiPerGator: https://docs.rc.ufl.edu/software/conda_installing_packages/

For project-specific questions, use your internal SPARC-P project communication channels.