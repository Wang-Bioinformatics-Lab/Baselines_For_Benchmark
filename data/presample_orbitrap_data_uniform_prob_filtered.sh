#!/bin/bash

BIN_DIR="../src/shared"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

curr_dir=$(pwd)

DATA_DIR="./../data/structural_similarity_basic/"

mkdir -p $DATA_DIR"/processed/data/presampled_orbitrap_basic_training_data_filtered/logs"
mkdir -p $DATA_DIR"/processed/data/presampled_orbitrap_basic_validation_data_filtered/logs"

# conda deactivate

cd ./batch_assembly
DATA_DIR="../../data/structural_similarity_basic/"


echo "Running presample_and_assemble_data.nf for training data"
bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_train_split.pickle" \
                                                 --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                                 --save_dir $DATA_DIR"/processed/data/presampled_orbitrap_basic_training_data_filtered" \
                                                 --num_epochs 1800 \
                                                 --memory_efficent true \
                                                 --mass_analyzer_lst "orbitrap" \
                                                 -c nextflow.config

echo "Running presample_and_assemble_data.nf for validation data"
bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                                                 --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                                 --save_dir $DATA_DIR"/processed/data/presampled_orbitrap_basic_validation_data_filtered" \
                                                 --num_epochs 1 \
                                                 --num_turns 10 \
                                                 --memory_efficent true \
                                                 --force_one_epoch true \
                                                 --mass_analyzer_lst "orbitrap" \
                                                 -c nextflow.config