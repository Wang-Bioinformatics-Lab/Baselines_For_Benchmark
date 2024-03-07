#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../../data/structural_similarity_positive/"
DATA_DIR="../../../data/structural_similarity/"
BIN_DIR="../../../src/spec2vec/src/"

SIMILARITY_THRESHOLD="0.8"

conda activate "$BIN_DIR/conda_env"

echo "Running spec2vec with ${SIMILARITY_THRESHOLD} similarity threshold"

cd $BIN_DIR

python3 runner.py --train_spectra $POSITIVE_DATA_DIR"/train_rows_structural_${SIMILARITY_THRESHOLD}.mgf" \
                 --train_metadata $DATA_DIR"/train_rows_structural_${SIMILARITY_THRESHOLD}.csv" \
                 --test_spectra $POSITIVE_DATA_DIR"/test_rows_structural.mgf" \
                 --test_metadata $DATA_DIR"/test_rows_structural.csv" \
                 --num_cpus 8 \
                 --withold_test True \
                 --model_save_name "./spec2vec_models/spec2vec_${SIMILARITY_THRESHOLD}_structural_similarity_split.model" \
                 --save_path "./structural_similarity/${SIMILARITY_THRESHOLD}/"
