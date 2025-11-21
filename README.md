# SPARC-P: Standardized Patient Assessment & Remediation System - Pediatric

## Overview
SPARC-P is a digital human training platform designed to help clinicians practice vaccine communication skills. This project implements the backend infrastructure, agent training pipeline, and deployment strategy for the SPARC-P system on the University of Florida's HiPerGator AI SuperPOD.

The system employs a **Hybrid RAG (Retrieval-Augmented Generation) and Fine-Tuning** architecture to create specialized agents:
- **Caregiver Agent**: Simulates a hesitant parent/caregiver with specific personas and emotional responses.
- **Coach Agent**: Evaluates the clinician's performance using the C-LEAR rubric and provides feedback.
- **Supervisor Agent**: Orchestrates the conversation, ensures safety (via NeMo Guardrails), and routes messages.

## System Architecture
The SPARC-P backend is designed for high-performance computing (HPC) environments and strict data compliance.

- **Compute**: HiPerGator AI SuperPOD (NVIDIA A100/B200 GPUs).
- **Containerization**: Apptainer/Singularity (Docker is used for development, Apptainer for HPC).
- **Storage**: `/blue` tier for large datasets and models (Home directory usage is minimized).
- **Orchestration**: LangGraph for managing the multi-agent state machine.
- **Speech Services**: NVIDIA Riva for Automatic Speech Recognition (ASR) and Text-to-Speech (TTS).
- **Safety**: NVIDIA NeMo Guardrails for content filtering and topic adherence.

## Project Notebooks

The project is documented and implemented across three primary Jupyter notebooks. Below is a detailed breakdown of each notebook with architectural diagrams.

### 1. Agent Training (`1_SPARC_Agent_Training.ipynb`)
This notebook implements the training pipeline for the SPARC-P agents.

#### 1.1 Introduction and System Purpose
![Introduction](images/notebook%201%20-%20section%201.png)
**Description:** This diagram illustrates the hybrid architecture used in this notebook. It shows how the system splits into two parallel tracks: RAG (Retrieval-Augmented Generation) for factual grounding using vector databases, and PEFT (Parameter-Efficient Fine-Tuning) using QLoRA to adapt the base model's style and behavior to specific personas (Caregiver, Coach, Supervisor).

#### 1.2 System Configuration
![System Config](images/notebook%201%20-%20section%203.3.png)
**Description:** This section initializes the environment settings on HiPerGator. It defines constants, verifies GPU availability, sets the base model ID (gpt-oss-120b), and crucially defines the persistent storage paths on the /blue storage tier, which is required for handling large-scale datasets that exceed standard home directory limits.

#### 1.3 Data Pipeline
![Data Pipeline](images/notebook%201%20-%20section%204.png)
**Description:** This section covers the data preparation lifecycle. Raw clinical text is first passed through Microsoft Presidio to strip Personally Identifiable Information (PII). The sanitized text is then split: one path builds the RAG Vector Store (ChromaDB) for factual queries, while the other uses a "Teacher Model" to generate synthetic question-answer pairs for fine-tuning.

#### 1.4 Fine-Tuning (QLoRA)
![QLoRA](images/notebook%201%20-%20section%205.png)
**Description:** This diagram visualizes the QLoRA training loop. It highlights how the massive base model is frozen and quantized to 4-bit precision to fit on the GPU. Small, trainable "Adapter" layers are attached to the attention modules. The SFTTrainer updates only these adapters based on the synthetic dataset, resulting in a lightweight, portable model file.

#### 1.5 Validation
![Validation](images/notebook%201%20-%20section%206.png)
**Description:** After training, the system must validate that the agents produce valid outputs. This workflow loads the base model combined with the new adapter, runs sample inference prompts, and uses Pydantic schemas to validate the structure of the JSON output (e.g., checking for specific fields like emotion or grade) before saving the final adapters.

#### 1.6 Interfaces
![Interfaces](images/notebook%201%20-%20section%207-8.png)
**Description:** This section covers the final testing and submission interfaces. It generates a SLURM script to run the training job on a GPU node via Apptainer. It also includes a Gradio interface that simulates the full multi-agent loop, showing how the Supervisor routes messages to the Caregiver or Coach and aggregates the response.

---

### 2. Containerization & Deployment (`2_SPARC_Containerization_and_Deployment.ipynb`)
This notebook handles the packaging and deployment of the backend system.

#### 2.1 Objectives
![Deployment Objectives](images/notebook%202%20-%20section%201.png)
**Description:** This section sets the objectives for packaging and deploying the backend. The goal is to containerize the Multi-Agent System (MAS), configure the WebSocket-to-gRPC bridge for Unity connectivity, and generate robust production SLURM scripts for deployment to HiPerGator.

#### 2.2 Container Build Strategy
![Build Strategy](images/notebook%202%20-%20section%202.png)
**Description:** This flow shows the Multi-Stage Build strategy used to create secure and small containers. A "Builder" stage uses Poetry to compile dependencies, and then only the necessary artifacts are copied over to a slim "Runtime" stage. This excludes compiler tools and cache files from the final production image.

#### 2.3 Local Development (Podman)
![Podman](images/notebook%202%20-%20section%203.png)
**Description:** This illustrates the local development environment using Podman Pods. Unlike standard Docker containers which are isolated, a "Pod" shares a network namespace (localhost). This allows the Riva Server, WebSocket Bridge, and MAS (Multi-Agent System) to communicate locally, perfectly simulating the production environment on a developer's machine.

#### 2.4 Production Deployment
![Production Deployment](images/notebook%202%20-%20section%204.png)
**Description:** This diagram shows the execution flow of the sparc_production.slurm script on HiPerGator. It details how the SLURM scheduler allocates resources (GPUs) and then launches three concurrent Apptainer containers in the background, keeping them alive with a wait command.

---

### 3. Real-Time Backend (`3_SPARC_RIVA_Backend.ipynb`)
This notebook implements the runtime execution of the backend.

#### 3.1 Runtime Objectives
![Runtime Goals](images/notebook%203%20-%20section%201.png)
**Description:** This section defines the objectives for the real-time backend. It implements the Real-Time, Multi-Agent Backend on HiPerGator, utilizing Apptainer for containerization, LangGraph for orchestration, and immutable audit logging to the /blue tier for compliance.

#### 3.2 Riva & Guardrails
![Riva Setup](images/notebook%203%20-%20section%202-3.png)
**Description:** This chart depicts the initialization of the speech services and safety rails. The Riva server is initialized with ASR (Speech-to-Text) and TTS (Text-to-Speech) enabled. Concurrently, NeMo Guardrails configuration files (config.yml, topical_rails.co) are generated to define the "boundary" of the conversation (e.g., refusing political topics).

#### 3.3 Multi-Agent Orchestration
![Orchestration](images/notebook%203%20-%20section%205.png)
**Description:** This is the core logic of the backend. It visualizes the Supervisor-Worker pattern. The User Input is first checked by the Supervisor (Guardrails). If safe, it triggers the Caregiver (generating the response) and the Coach (evaluating the response) in parallel to minimize latency. The results are aggregated into a single JSON response.

#### 3.4 API Server Integration
![API Server](images/notebook%203%20-%20section%206.png)
**Description:** This diagram maps the data flow through the FastAPI application. The Unity Client sends a request to /v1/chat. The server logs the request for auditing, invokes the LangGraph orchestration loop (defined in Section 5), and returns the structured ChatResponse containing text, audio (Base64), and animation cues.

#### 3.5 Security and Compliance
![Security](images/notebook%203%20-%20section%207.png)
**Description:** This section outlines the security protocols and persistent deployment. It adheres to the HIPAA Mandate using a 'Transient PHI' model, where user data is processed in-memory and immediately discarded. The launch_backend.slurm script ensures the service runs persistently on a secure GPU node.

## Prerequisites
- **Hardware**: Access to a GPU-enabled node (e.g., NVIDIA A100).
- **Software**: 
  - Python 3.10+
  - Apptainer (Singularity)
  - NVIDIA Riva
- **Storage**: Access to the `/blue` storage tier on HiPerGator.

## Usage

### 1. Training Agents
Open `1_SPARC_Agent_Training.ipynb` to:
1. Configure storage paths (ensure `/blue` directory access).
2. Run the data ingestion and sanitization pipeline.
3. Execute QLoRA fine-tuning for Caregiver, Coach, and Supervisor agents.
4. Validate models using the provided Gradio interfaces.

### 2. Deploying Infrastructure
Open `2_SPARC_Containerization_and_Deployment.ipynb` to:
1. Build the MAS Docker container.
2. Convert Docker images to Apptainer SIF images.
3. Generate the `sparc_production.slurm` script.
4. Submit the SLURM job to launch the backend services.

### 3. Running the Backend
Open `3_SPARC_RIVA_Backend.ipynb` to:
1. Initialize and launch the NVIDIA Riva server.
2. Verify ASR and TTS services.
3. Start the FastAPI server to handle incoming requests from the Unity client.

## Security & Compliance
- **HIPAA Compliance**: The system is designed to handle 'Transient PHI'. Audio and transcripts are processed in volatile memory and are strictly not logged to disk.
- **Audit Logging**: Non-sensitive operational logs are written to `/blue/my_group/sparc-p/logs/audit.log` for compliance tracking.
- **Guardrails**: NeMo Guardrails ensure the agents do not engage in off-topic or unsafe discussions (e.g., politics, medical advice outside scope).
