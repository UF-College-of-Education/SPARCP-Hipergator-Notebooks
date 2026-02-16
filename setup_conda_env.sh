#!/bin/bash
# SPARC-P Conda Environment Setup Script for HiPerGator
# 
# This script automates the creation of conda environments following UF RC best practices
# Usage: bash setup_conda_env.sh <training|backend|both>

set -e  # Exit on error

# Configuration - EDIT THESE VALUES
GROUP_NAME="jasondeanarnold"      # Your HiPerGator group name
USER_NAME="${USER}"               # Your username (defaults to current user)
CONDA_BASE="/blue/jasondeanarnold/SPARCP/conda_envs"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}SPARC-P Conda Environment Setup for HiPerGator${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""

# Check if we're on HiPerGator
if [[ ! -d "/blue" ]]; then
    echo -e "${RED}ERROR: /blue directory not found. This script must be run on HiPerGator.${NC}"
    exit 1
fi

# Parse arguments
SETUP_TYPE=${1:-both}

if [[ "$SETUP_TYPE" != "training" && "$SETUP_TYPE" != "backend" && "$SETUP_TYPE" != "both" ]]; then
    echo -e "${RED}ERROR: Invalid argument. Use: training, backend, or both${NC}"
    echo "Usage: bash setup_conda_env.sh <training|backend|both>"
    exit 1
fi

# Check if GROUP_NAME was updated
if [[ "$GROUP_NAME" == "YOUR_GROUP" ]]; then
    echo -e "${YELLOW}WARNING: Please edit this script and set GROUP_NAME to your HiPerGator group${NC}"
    echo "Your group name can be found with: id -gn"
    echo ""
    read -p "Enter your HiPerGator group name: " GROUP_NAME
    CONDA_BASE="/blue/${GROUP_NAME}/${USER_NAME}/conda_envs"
fi

echo "Configuration:"
echo "  Group: ${GROUP_NAME}"
echo "  User: ${USER_NAME}"
echo "  Conda base: ${CONDA_BASE}"
echo ""

# Create conda directory if it doesn't exist
mkdir -p "${CONDA_BASE}"

# Load conda module
echo -e "${GREEN}Loading conda module...${NC}"
module purge
module load conda

# Function to create environment
create_env() {
    local env_name=$1
    local env_file=$2
    local env_path="${CONDA_BASE}/${env_name}"
    
    echo ""
    echo -e "${GREEN}=== Creating ${env_name} environment ===${NC}"
    echo "Environment path: ${env_path}"
    
    if [[ -d "${env_path}" ]]; then
        echo -e "${YELLOW}Environment already exists. Updating...${NC}"
        conda env update -f "${env_file}" -p "${env_path}"
    else
        echo "Creating new environment..."
        conda env create -f "${env_file}" -p "${env_path}"
    fi
    
    echo -e "${GREEN}✓ ${env_name} environment ready${NC}"
    echo "To activate: conda activate ${env_path}"
}

# Create requested environments
if [[ "$SETUP_TYPE" == "training" || "$SETUP_TYPE" == "both" ]]; then
    create_env "sparc_training" "environment_training.yml"
fi

if [[ "$SETUP_TYPE" == "backend" || "$SETUP_TYPE" == "both" ]]; then
    create_env "sparc_backend" "environment_backend.yml"
fi

# Print summary
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Next steps:"
echo ""

if [[ "$SETUP_TYPE" == "training" || "$SETUP_TYPE" == "both" ]]; then
    echo "For Training Jobs:"
    echo "  1. Use this in your SLURM scripts:"
    echo "     module load conda"
    echo "     conda activate ${CONDA_BASE}/sparc_training"
    echo ""
fi

if [[ "$SETUP_TYPE" == "backend" || "$SETUP_TYPE" == "both" ]]; then
    echo "For Backend Services:"
    echo "  1. Use this in your SLURM scripts:"
    echo "     module load conda"
    echo "     conda activate ${CONDA_BASE}/sparc_backend"
    echo ""
fi

echo "To verify installation:"
echo "  module load conda"
echo "  conda activate ${CONDA_BASE}/sparc_training  # or sparc_backend"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\")"
echo ""
