#!/bin/bash

cd ..
# Create a new conda environment
conda env create -f env.yaml -y

# Activate the environment
conda activate neurona

# Initialize only third-party library submodules
git submodule update --init --recursive third_party/Concepts
git submodule update --init --recursive third_party/Jacinle

# Set the path to the Jacinle bin
export PATH=$PWD/third_party/Jacinle/bin:$PATH

# Install the Concepts package
pip install -e third_party/Concepts

# # Install the rest of the packages
pip install PyYAML
pip install peewee
pip install opencv-python

cd scripts