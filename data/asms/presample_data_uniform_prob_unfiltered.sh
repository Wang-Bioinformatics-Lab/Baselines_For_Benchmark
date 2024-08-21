#!/bin/bash

BIN_DIR="../src/shared"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

curr_dir=$(pwd)

DATA_DIR="./../data/asms/"

mkdir -p $DATA_DIR"/processed/data/presampled_basic_training_data_filtered/logs"

cd ../
conda activate "$BIN_DIR/conda_env"
mkdir -p  $DATA_DIR"/processed/data/presampled_basic_training_data/logs"

python3 ./presample_pairs_generic.py  --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_train_split.pickle" \
                                    --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                    --save_dir $DATA_DIR"/processed/data/presampled_basic_training_data" \
                                    --num_epochs 150


mkdir -p  $DATA_DIR"/processed/data/presampled_basic_validation_data/logs"

python3 ./presample_pairs_generic.py --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                                    --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                    --save_dir $DATA_DIR"/processed/data/presampled_basic_validation_data" \
                                    --num_epochs 1 \
                                    --num_turns 10
