#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../data/structural_similarity_positive/"
DATA_DIR="../../data/structural_similarity/"
BIN_DIR="../../../../src/ms2deepscore/"

conda activate "$BIN_DIR/conda_env"

echo "Performing Preprocessing for MS2DeepScore with a basic split"

cd $BIN_DIR

python3 preprocess_data.py --train_spectra $POSITIVE_DATA_DIR"/train_rows.mgf" \
                            --train_metadata $DATA_DIR"/train_rows.csv" \
                            --test_spectra $POSITIVE_DATA_DIR"/test_rows.mgf" \
                            --test_metadata $DATA_DIR"/test_rows.csv" \
                            --save_dir "./basic_similarity/"
