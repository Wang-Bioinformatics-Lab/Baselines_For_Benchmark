#!/bin/bash

BIN_DIR="../src/shared"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate "$BIN_DIR/conda_env"

curr_dir=$(pwd)

# cd $BIN_DIR

DATA_DIR="./structural_similarity_basic/"

python3 ./presample_test_data.py --n_jobs 6 \
                                 --subsample 100_000 \
                                 --metadata_path $DATA_DIR"/raw/test_rows.csv" \
                                 --pairwise_similarities_path $DATA_DIR"/processed/test_tanimoto_df.csv" \
                                 --train_test_similarities_path $DATA_DIR"/processed/train_test_tanimoto_df.csv" \
                                 --output_path $DATA_DIR"/processed/test_data/all_pairs.parquet" \
                                 --temp_file_dir "/data/nas-gpu/tmp"