#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="../../data/fresh_dataset/"

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Training for MS2DeepScore with simple sampling split."

cd $BIN_DIR


python3 train_presampled.py --train_path $DATA_DIR"/processed/data/ALL_GNPS_positive_train_split.pickle" \
                            --val_path $DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                            --tanimoto_path $DATA_DIR"/raw/train_similarities.csv" \
                            --presampled_train_data_dir $DATA_DIR"/raw/prebatched_data/train_filtered_biased.hdf5" \
                            --presampled_val_data_dir $DATA_DIR"/raw/prebatched_data/val_filtered_biased.hdf5" \
                            --save_dir "./fresh_dataset/"

