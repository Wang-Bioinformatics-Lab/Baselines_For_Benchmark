#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate ../conda_env

python3 ../src/preprocess_data.py   --train_data_spectra_path "/scratch/mstro016/Baselines for Benchmark/data/nf_output/spectra/Bruker_Fragmentation_Prediction_train.parquet" \
                                    --train_data_metadata_path "/scratch/mstro016/Baselines for Benchmark/data/nf_output/summary/Bruker_Fragmentation_Prediction_train.csv" \
                                    --test_data_spectra_path "/scratch/mstro016/Baselines for Benchmark/data/nf_output/spectra/Bruker_Fragmentation_Prediction_test.parquet" \
                                    --test_data_metadata_path "/scratch/mstro016/Baselines for Benchmark/data/nf_output/summary/Bruker_Fragmentation_Prediction_test.csv" 