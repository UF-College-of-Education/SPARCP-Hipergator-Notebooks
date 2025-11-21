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
The project is documented and implemented across three primary Jupyter notebooks:

### 1. Agent Training (`1_SPARC_Agent_Training.ipynb`)
This notebook implements the training pipeline for the SPARC-P agents.
- **Hybrid Architecture**: Combines RAG for factual accuracy with QLoRA fine-tuning for stylistic alignment.
- **Data Pipeline**: 
  - Sanitizes clinical text using Microsoft Presidio to remove PII.
  - Builds a ChromaDB vector store for RAG.
  - Generates synthetic training data using a "Teacher Model".
- **Fine-Tuning**: Uses QLoRA (Quantized Low-Rank Adaptation) to adapt the `gpt-oss-120b` base model.
- **Validation**: Validates agent outputs against Pydantic schemas and provides Gradio interfaces for individual and multi-agent testing.

### 2. Containerization & Deployment (`2_SPARC_Containerization_and_Deployment.ipynb`)
This notebook handles the packaging and deployment of the backend system.
- **Container Build**: Defines a multi-stage Dockerfile (`Dockerfile.mas`) for the Multi-Agent System (MAS).
- **Local Development**: Uses Podman Pods to simulate the production network environment locally.
- **HPC Deployment**: 
  - Instructions for converting Docker images to Apptainer (`.sif`) format.
  - Generates the `sparc_production.slurm` script for persistent service deployment on HiPerGator.
  - Configures the WebSocket-to-gRPC bridge for Unity connectivity.

### 3. Real-Time Backend (`3_SPARC_RIVA_Backend.ipynb`)
This notebook implements the runtime execution of the backend.
- **Riva Setup**: Automates the deployment of NVIDIA Riva Server (ASR/TTS).
- **Safety Rails**: Configures NeMo Guardrails (`config.yml`, `topical_rails.co`) to enforce safety and topic boundaries.
- **Orchestration Logic**: Implements the Supervisor-Worker pattern using LangGraph to manage the Supervisor, Caregiver, and Coach agents.
- **API Server**: Exposes a `POST /v1/chat` endpoint via FastAPI for the Unity client.
- **Compliance**: Implements a 'Transient PHI' model where sensitive data is processed in-memory and not persisted.

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
