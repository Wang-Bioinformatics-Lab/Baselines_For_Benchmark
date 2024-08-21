#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="../../data/asms/"
OLD_DATA_DIR="../../data/asms/"

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Training for MS2DeepScore with simple sampling split."

cd $BIN_DIR


python3 train_presampled.py --train_path $OLD_DATA_DIR"/processed/data/ALL_GNPS_positive_train_split.pickle" \
                            --val_path $OLD_DATA_DIR"/processed/data/ALL_GNPS_positive_val_split.pickle" \
                            --tanimoto_path $OLD_DATA_DIR"/processed/train_tanimoto_df.csv" \
                            --presampled_train_data_dir $DATA_DIR"processed/data/presampled_basic_training_data/data.hdf5" \
                            --presampled_val_data_dir $DATA_DIR"/processed/data/presampled_basic_validation_data/data.hdf5" \
                            --save_dir "./asms/"

