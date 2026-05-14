#!/bin/bash
# Create a conda environment for LangGraph Data Augmentation on HiPerGator

ENV_NAME="sparc_augmentation"

echo "Creating conda environment: $ENV_NAME..."
# We use Python 3.10 as it's very stable for these AI frameworks
conda create -n $ENV_NAME python=3.10 -y

# Activating conda environment within bash script
# This finds the base conda path dynamically
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

echo "Activating environment..."
conda activate $ENV_NAME

echo "Installing LangGraph and LangChain dependencies..."
# We include python-dotenv in case you want to manage API keys via a .env file later
pip install langchain-core langchain-openai langgraph pydantic python-dotenv

echo ""
echo "=========================================================="
echo "✅ Environment '$ENV_NAME' created and dependencies installed!"
echo "=========================================================="
echo "To start using it, run:"
echo "conda activate $ENV_NAME"
echo "=========================================================="
