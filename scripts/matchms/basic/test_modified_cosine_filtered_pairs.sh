#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../../data/structural_similarity_positive/"
DATA_DIR="../../../data/structural_similarity/processed"
BIN_DIR="../../../src/spec2vec/src/"
PAIRS_PATH=$DATA_DIR"/pairs_test_filtered.feather"

SIMILARITY_THRESHOLD="0.6"

conda activate "$BIN_DIR/new_conda_env"

echo "Running modified_cosine with a basic split and using pairs from the filtered dataset."
echo "Pairs Path:" $PAIRS_PATH

cd $BIN_DIR

python3 eval.py --method 'modified_cosine' \
                --data $DATA_DIR \
                --split_type 'basic' \
                --n_most_recent 1 \
                --pairs_path $PAIRS_PATH
