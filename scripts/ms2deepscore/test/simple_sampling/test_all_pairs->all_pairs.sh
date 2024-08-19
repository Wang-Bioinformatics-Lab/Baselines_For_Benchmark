#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="../../data/structural_similarity_basic/processed"
MODEL_PATH="./simple_sampling/model_21_06_2024_09_13_17.hdf5"

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Inference for MS2DeepScore with a basic split, filtered by pairs"

cd $BIN_DIR

python3 test_presampled.py --test_path $DATA_DIR"/data/ALL_GNPS_positive_test_split.pickle" \
                 --presampled_pairs_path $DATA_DIR"/test_data/all_pairs_new.parquet" \
                 --save_dir "./simple_sampling" \
                 --n_jobs 30 \
                 --model_path $MODEL_PATH