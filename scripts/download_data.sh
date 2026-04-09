#!/bin/bash

cd ..

# HuggingFace requires token auth for LFS even on public repos.
# Run `huggingface-cli login` first, or set the HF_TOKEN env variable.
HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
if [ -z "$HF_TOKEN" ]; then
    # Fallback: read token via huggingface_hub Python API
    HF_TOKEN=$(python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token() or '')" 2>/dev/null)
fi
if [ -z "$HF_TOKEN" ]; then
    echo "Error: No HuggingFace token found."
    echo "  Run 'huggingface-cli login' or export HF_TOKEN=<your_token>"
    exit 1
fi

echo "Downloading BOLD5000-QA dataset..."
git clone "https://hf:${HF_TOKEN}@huggingface.co/datasets/PPWangyc/BOLD5000-QA" data/BOLD5000-QA

# echo "Downloading WAVE-BOLD5000 dataset..."
# git clone "https://hf:${HF_TOKEN}@huggingface.co/datasets/PPWangyc/WAVE-BOLD5000" data/WAVE-BOLD5000

# Download non-HF dataset submodules via git
# git submodule update --init --recursive data/algonauts_2025.competitors

cd scripts
