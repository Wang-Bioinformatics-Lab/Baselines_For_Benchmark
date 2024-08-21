#!/bin/bash

BIN_DIR="../src/shared"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

curr_dir=$(pwd)

DATA_DIR="./../data/asms/"

mkdir -p $DATA_DIR"/processed/data/presampled_filtered_training_data_biased/logs"

cd ../batch_assembly
DATA_DIR="../../data/asms/"

bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_train_split.pickle" \
                                                 --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                                 --save_dir $DATA_DIR"/processed/data/presampled_filtered_training_data_biased_03_20" \
                                                 --num_epochs 150 \
                                                 --memory_efficent true \
                                                 -with-report presampling_nf_log.html \
                                                 -c nextflow.config \
                                                 --num_bins 20 \
                                                 --exponential_bins 0.3

mkdir -p $DATA_DIR"/processed/data/presampled_filtered_validation_data_biased/logs"
echo "Running presample_and_assemble_data.nf for validation data"
bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                                                 --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                                 --save_dir $DATA_DIR"/processed/data/presampled_filtered_validation_data_biased_03_20" \
                                                 --num_epochs 1 \
                                                 --num_turns 10 \
                                                 --memory_efficent true \
                                                 --force_one_epoch true \
                                                 -c nextflow.config \
                                                 --num_bins 20 \
                                                 --exponential_bins 0.3