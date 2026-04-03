#!/bin/bash
#SBATCH --job-name=sparc-Caregiver
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jayrosen@ufl.edu
#SBATCH --partition=gpu
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

pwd; hostname; date

echo "=== SPARC-P Training: Caregiver Agent ==="
echo "Resource profile: 3 GPUs, 16 CPU cores allocated"

module purge
module load conda
module load cuda/12.8

SPARC_BASE_PATH=${SPARC_BASE_PATH:-/blue/jasondeanarnold/SPARCP}
CONDA_ENV=${SPARC_TRAINING_ENV:-$SPARC_BASE_PATH/conda_envs/sparc_training}

echo "Using SPARC_BASE_PATH=$SPARC_BASE_PATH"
echo "Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Run-mode controls for Section 5.2
export RUN_TRAINING=true
export SPARC_AGENT_NAME=Caregiver
export SPARC_NUM_EPOCHS=3

# Single-model mode (default): use the local Meta-downloaded Llama checkpoint
export SPARC_MODEL_NAME=${SPARC_MODEL_NAME:-$SPARC_BASE_PATH/trained_models/meta_llama/Llama3.1-8B-Instruct}

# Keep remote / future models disabled for now
export ENABLE_REMOTE_MODEL_TESTS=${ENABLE_REMOTE_MODEL_TESTS:-false}

# Compare-mode toggle (set true later together with ENABLE_REMOTE_MODEL_TESTS=true):
export COMPARE_BOTH_MODELS=${COMPARE_BOTH_MODELS:-false}

# Llama 3 precision controls:
#   LLAMA3_QUANT_MODE: full (default first run) or 4bit (QLoRA later)
#   LLAMA3_COMPARE_PRECISIONS: when true, run both 4bit and full
export LLAMA3_QUANT_MODE=${LLAMA3_QUANT_MODE:-full}
export LLAMA3_COMPARE_PRECISIONS=${LLAMA3_COMPARE_PRECISIONS:-false}

echo "Starting notebook execution for Caregiver..."
cd $SPARC_BASE_PATH

jupyter nbconvert --to notebook --execute H1b_Model_Fine_Tuning_PyTorch_local_meta_default.ipynb \
    --output executed_Caregiver_H1b_Model_Fine_Tuning_PyTorch_local_meta_default.ipynb \
    --ExecutePreprocessor.timeout=-1

echo "Training notebook execution completed."

mkdir -p $SPARC_BASE_PATH/trained_models/Caregiver
conda env export > $SPARC_BASE_PATH/trained_models/Caregiver/environment_snapshot.yml

date