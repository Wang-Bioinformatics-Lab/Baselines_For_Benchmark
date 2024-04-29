#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../data/structural_similarity_positive/"
DATA_DIR="../../data/structural_similarity/"
BIN_DIR="../../../../src/ms2deepscore/"

SIMILARITY_THRESHOLD="0.7"
SPLIT_STYLE="structural"

conda activate "$BIN_DIR/conda_env"

echo "Performing Preprocessing for MS2DeepScore with ${SIMILARITY_THRESHOLD} similarity threshold"

cd $BIN_DIR

python3 preprocess_data.py --train_spectra $POSITIVE_DATA_DIR"/train_rows_${SPLIT_STYLE}_${SIMILARITY_THRESHOLD}.mgf" \
                            --train_metadata $DATA_DIR"/train_rows_${SPLIT_STYLE}_${SIMILARITY_THRESHOLD}.csv" \
                            --test_spectra $POSITIVE_DATA_DIR"/test_rows_${SPLIT_STYLE}.mgf" \
                            --test_metadata $DATA_DIR"/test_rows_${SPLIT_STYLE}.csv" \
                            --save_dir "./${SPLIT_STYLE}_similarity/${SIMILARITY_THRESHOLD}/"
