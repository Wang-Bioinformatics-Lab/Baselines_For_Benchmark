#!/usr/bin/env bash

source ./venv/bin/activate

python3 runner.py --spectra '../../../data/nf_output/spectra/Bruker_Fragmentation_Prediction_train.parquet' \
                 --metadata '../../../data/nf_output/summary/Bruker_Fragmentation_Prediction_train.csv' \
                 --num_cpus 4 \
                 --test