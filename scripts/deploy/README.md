# SPARC-P PubApp deployment bundle

This directory contains everything needed to stand up the FastAPI backend
(`v2/main.py`) on a single-L4 PubApp VM, using the LoRA adapters trained
on HiPerGator.

## Why this exists

The original deploy docs (`v2/md/P1_*.md`, `v2/md/P2_*.md`) still reference
`gpt-oss-20b`, ship Quadlet units that bind-mount `MODEL_DIR` but never
point the code at a concrete base model, and assume three LoRA adapters
(Caregiver, Coach, Supervisor) are all on disk. None of those assumptions
match the current repo:

- `v2/main.py` hard-codes `BASE_MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"`.
- Training happens in the `sparc_training_clean` conda env
  (`peft==0.18.1`, `transformers==5.5.0`, `torch==2.10.0+cu128`), so the
  adapters must be loaded by a matching runtime — `environment_backend.yml`
  in the repo root is now pinned accordingly.
- Only the Caregiver (Anne + Maya, fine-tuned on the synthetic-3000
  dataset — the adapter lives at `live_jupyter_runs/CaregiverAgent/full/`)
  and the C-LEAR Coach (`live_jupyter_runs/C-LEAR_CoachAgent/full/`) exist
  today. The sibling directories `full-d/`, `full-synthetic-d/`, and
  `full-llama-4-9-26/` under `CaregiverAgent/` are A/B snapshots kept
  around for comparing v3 / v4 dataset styles — the deploy scripts pull
  them so you can switch with `CAREGIVER_RUN=...`, but `full/` is the
  production default. There is no Supervisor JSONL yet, so the backend
  treats that adapter as optional.

## L4 VRAM budget

Single L4 GPU (24 GB) shared between Riva and the backend:

| Component                              | VRAM  |
| -------------------------------------- | ----- |
| Llama 3.1 8B, 4-bit NF4 + bf16 compute | ~5.6  |
| KV cache (max_length 1024)             | ~1.5  |
| 2× LoRA adapters hot-swap              | ~0.1  |
| Activations / workspace                | ~1.5  |
| **sparc-backend**                      | ~8.7  |
| Riva ASR (Parakeet) + TTS (FastPitch)  | ~6.0  |
| **Total used**                         | ~14.7 |
| Headroom                               | ~9.3  |

If you enable the Supervisor adapter later, budget another ~50 MB for
the weights plus ~0.5 GB of activation headroom during adapter swaps.

## Files

```
scripts/deploy/
├── README.md                      (this file)
├── rsync_from_hpg.sh              pull artefacts from HiPerGator -> staging
├── layout_pubapp_models.sh        materialize /pubapps/SPARCP/ tree
├── quadlet/
│   ├── sparc-backend.container    Podman Quadlet unit for the FastAPI backend
│   └── riva-server.container      Podman Quadlet unit for Riva ASR+TTS
└── guardrails/                    NeMo Rails config shipped with the backend
    ├── config.yml
    ├── prompts.yml
    └── rails/topical.co
```

## End-to-end runbook

```bash
# --- On a host with SSH access to HiPerGator ---------------------------------
export HPG_USER=<your-gatorlink>
export STAGING_DIR=/tmp/sparcp_staging
bash scripts/deploy/rsync_from_hpg.sh

# scp the staging tree to the PubApp VM
scp -r "$STAGING_DIR" pubapp-vm:/tmp/

# --- On the PubApp VM --------------------------------------------------------
# 1. Stage the trained artefacts.
sudo -u pubapps bash scripts/deploy/layout_pubapp_models.sh \
  --source /tmp/sparcp_staging

# 2. Drop the Firebase service-account JSON into /pubapps/SPARCP/config/
#    (this file is NOT in the rsync manifest).
sudo install -o pubapps -m 0600 \
  firebase-credentials.json /pubapps/SPARCP/config/firebase-credentials.json

# 3. Create the backend conda env (or build the container image).
conda env create -f environment_backend.yml \
  -p /pubapps/SPARCP/conda_envs/sparc_backend

# 4. Initialize Riva (one-time, downloads ASR + TTS model artefacts).
cd /pubapps/SPARCP/riva_quickstart
bash riva_init.sh

# 5. Install the Quadlet units and start services.
sudo cp scripts/deploy/quadlet/*.container /etc/containers/systemd/
sudo systemctl daemon-reload
sudo systemctl start riva-server sparc-backend

# 6. Smoke test.
curl -sS http://localhost:8000/health | jq
```

The `/health` payload now surfaces which adapters actually loaded
(`adapters_loaded: ["caregiver", "coach"]`), which base model is active
(`base_model_name`), and whether guardrails + RAG are up.
