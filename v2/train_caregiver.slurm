#!/bin/bash
#SBATCH --job-name=sparc-Caregiver
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayrosen@ufl.edu
#SBATCH --partition=hpg-b200
#SBATCH --account=jasondeanarnold
#SBATCH --qos=jasondeanarnold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=5:00:00
#SBATCH --output=train_Caregiver_%j.log
#SBATCH --error=train_Caregiver_%j.err

set -eo pipefail

pwd
hostname
date

echo "=== SPARC-P Training: Caregiver Agent ==="
echo "Resource profile: 1 GPU(s), 4 CPU core(s) allocated"

module purge
module load conda
module load cuda/12.8

SPARC_BASE_PATH="/blue/jasondeanarnold/SPARCP"
CONDA_ENV="$SPARC_BASE_PATH/conda_envs/sparc_training_clean"
NOTEBOOK_DIR="$SPARC_BASE_PATH/SPARCP-Hipergator-Notebooks/v2"
NOTEBOOK_FILE="H1b_Model_Fine_Tuning_PyTorch_local_meta_default.ipynb"
NOTEBOOK_PATH="$NOTEBOOK_DIR/$NOTEBOOK_FILE"

echo "Using SPARC_BASE_PATH=$SPARC_BASE_PATH"
echo "Using NOTEBOOK_PATH=$NOTEBOOK_PATH"
echo "Activating conda environment: $CONDA_ENV"

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "$CONDA_ENV"
set -u

export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python: $(which python)"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0))
PY

test -f "$NOTEBOOK_PATH" || { echo "Notebook not found: $NOTEBOOK_PATH"; exit 1; }

# Optional runtime flags used by earlier notebook sections
export RUN_TRAINING=true
export SPARC_AGENT_NAME=Caregiver
export SPARC_MODEL_NAME="${SPARC_MODEL_NAME:-$SPARC_BASE_PATH/trained_models/meta_llama/Llama3.1-8B-Instruct}"
export ENABLE_REMOTE_MODEL_TESTS="${ENABLE_REMOTE_MODEL_TESTS:-false}"
export COMPARE_BOTH_MODELS="${COMPARE_BOTH_MODELS:-false}"
export LLAMA3_QUANT_MODE="${LLAMA3_QUANT_MODE:-full}"
export LLAMA3_COMPARE_PRECISIONS="${LLAMA3_COMPARE_PRECISIONS:-false}"
export DOWNLOAD_META_MODEL="${DOWNLOAD_META_MODEL:-false}"

echo "Starting notebook execution for Caregiver..."
cd "$NOTEBOOK_DIR"

jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" \
    --output "executed_Caregiver_H1b_Model_Fine_Tuning_PyTorch_local_meta_default.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=sparc_training_clean

echo "Training notebook execution completed."

mkdir -p "$SPARC_BASE_PATH/trained_models/Caregiver"
conda env export > "$SPARC_BASE_PATH/trained_models/Caregiver/environment_snapshot.yml"

date
