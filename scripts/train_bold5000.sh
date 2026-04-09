#!/bin/bash

config=${1:-neurona_bold5000}
subject=$2

. ~/.bashrc

cd ../

conda activate neurona
export PATH=$PWD/third_party/Jacinle/bin:$PATH

scripts_args="--config configs/${config}.yaml"

# Override subject if provided as second argument
if [ -n "$subject" ]; then
    scripts_args="$scripts_args --subject $subject"
fi

echo "Running with config: configs/${config}.yaml"
echo "Args: $scripts_args"
jac-run src/train_bold5000.py $scripts_args

conda deactivate
cd scripts
