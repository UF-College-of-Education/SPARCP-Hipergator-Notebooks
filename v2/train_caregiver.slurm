#!/bin/bash
#SBATCH --job-name=sparc-Caregiver
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayrosen@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --account=jasondeanarnold
#SBATCH --qos=jasondeanarnold
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=125gb
#SBATCH --time=48:00:00
#SBATCH --output=train_Caregiver_%j.log
#SBATCH --error=train_Caregiver_%j.err

set -eo pipefail

pwd
hostname
date

echo "=== SPARC-P Training: Caregiver Agent ==="

module purge
module load conda
module load cuda/12.8

SPARC_BASE_PATH="/blue/jasondeanarnold/SPARCP"
CONDA_ENV="$SPARC_BASE_PATH/conda_envs/sparc_training"
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

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

export RUN_TRAINING=true
export SPARC_AGENT_NAME=Caregiver
export SPARC_NUM_EPOCHS=3
export SPARC_MODEL_NAME="${SPARC_MODEL_NAME:-$SPARC_BASE_PATH/trained_models/meta_llama/Llama3.1-8B-Instruct}"
export ENABLE_REMOTE_MODEL_TESTS="${ENABLE_REMOTE_MODEL_TESTS:-false}"
export COMPARE_BOTH_MODELS="${COMPARE_BOTH_MODELS:-false}"
export LLAMA3_QUANT_MODE="${LLAMA3_QUANT_MODE:-full}"
export LLAMA3_COMPARE_PRECISIONS="${LLAMA3_COMPARE_PRECISIONS:-false}"

test -f "$NOTEBOOK_PATH" || { echo "Notebook not found: $NOTEBOOK_PATH"; exit 1; }

cd "$NOTEBOOK_DIR"

echo "Starting notebook execution for Caregiver..."
jupyter nbconvert --to notebook --execute "$NOTEBOOK_FILE" \
  --output "executed_Caregiver_H1b_Model_Fine_Tuning_PyTorch_local_meta_default.ipynb" \
  --ExecutePreprocessor.timeout=-1

echo "Training notebook execution completed."

mkdir -p "$SPARC_BASE_PATH/trained_models/Caregiver"
conda env export > "$SPARC_BASE_PATH/trained_models/Caregiver/environment_snapshot.yml"

date