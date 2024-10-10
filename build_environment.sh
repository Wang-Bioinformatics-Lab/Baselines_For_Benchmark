#!/bin/bash

cd src/shared

# Check if mamba is installed
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found, installing with conda"

    conda env create -f environment.yml -p ./conda_env
else
    echo "Using mamba to install the environment"

    mamba env create -f environment.yml -p ./conda_env  
fi