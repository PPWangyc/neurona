#!/bin/bash

subject=$1
. ~/.bashrc
export PATH=$PWD/third_party/Jacinle/bin:$PATH
cd ../..

conda activate neurona
# Create dataset
script_args="--subject $subject"
echo "Running create_bold5000_fqa.py with args: $script_args"
jac-run src/create_dataset/create_bold5000_fqa.py $script_args
conda deactivate
cd scripts/dataset