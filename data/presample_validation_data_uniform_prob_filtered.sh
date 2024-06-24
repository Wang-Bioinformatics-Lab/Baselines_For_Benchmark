#!/bin/bash

BIN_DIR="../src/shared"

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

curr_dir=$(pwd)

DATA_DIR="../../data/structural_similarity_basic/"

mkdir -p $DATA_DIR"/processed/data/presampled_basic_validation_data_filtered/logs"

# It turns out this is quite slow, so we will parallelize it with nextflow
# conda activate "$BIN_DIR/conda_env"
# python3 ../presample_pairs_generic.py  --validation_path $DATA_DIR"/processed/data/ALL_GNPS_positive_validation_split.pickle" \
#                                                     --validation_pairs_path $DATA_DIR"/processed/pairs_validation_filtered.feather" \
#                                                     --save_dir $curr_dir"/presampled_basic_validation_data_filtered" \
#                                                     --num_epochs 1800

# Instead, we first convert the pairs to an hdf5 file, and then use nextflow to parallelize the process

# conda activate "$BIN_DIR/conda_env"
# python3 pre_generate_hdf5_pairs.py  --pairs_path "./structural_similarity_basic/processed/pairs_validation_filtered.feather" \
#                                     --output_path "./structural_similarity_basic/processed/pairs_validation_filtered.hdf5"
# conda deactivate

cd ./batch_assembly
# bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
#                                                  --train_pairs_path $DATA_DIR"/processed/pairs_validation_filtered.hdf5" \
#                                                  --save_dir $DATA_DIR"/processed/data/presampled_basic_validation_data_filtered" \
#                                                  --num_epochs 1 \
#                                                  --num_turns 10 \
#                                                  --force_one_epoch true \
#                                                  -c nextflow.config

bash nextflow run presample_and_assemble_data.nf --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                                                 --tanimoto_scores_path $DATA_DIR"/processed/train_tanimoto_df.csv" \
                                                 --save_dir $DATA_DIR"/processed/data/presampled_basic_validation_data_filtered" \
                                                 --num_epochs 1 \
                                                 --num_turns 10 \
                                                 --memory_efficent true \
                                                 --force_one_epoch true \
                                                 -c nextflow.config