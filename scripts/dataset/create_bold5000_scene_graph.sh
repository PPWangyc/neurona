#!/bin/bash

. ~/.bashrc

cd ../..
export PATH=$PWD/third_party/Jacinle/bin:$PATH
conda activate neurona
# Create dataset
jac-run src/create_dataset/create_bold5000_scene_graph.py
conda deactivate
cd scripts/dataset