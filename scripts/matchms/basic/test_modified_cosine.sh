#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../../data/structural_similarity_positive/"
DATA_DIR="../../../data/structural_similarity/processed"
BIN_DIR="../../../src/spec2vec/src/"

SIMILARITY_THRESHOLD="0.6"

# conda activate "$BIN_DIR/new_conda_env"

echo "Running modified_cosine with a basic split"

cd $BIN_DIR

conda activate ../../shared/conda_env

python3 eval.py --method 'modified_cosine' \
                --data $DATA_DIR \
                --split_type 'basic' \
                # --test_pairs_path $DATA_DIR/pairs_testing_test_filtered.csv \