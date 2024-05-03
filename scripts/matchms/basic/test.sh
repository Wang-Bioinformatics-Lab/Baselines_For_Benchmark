#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

POSITIVE_DATA_DIR="../../../data/structural_similarity_positive/"
DATA_DIR="../../../data/structural_similarity/processed"
BIN_DIR="../../../src/spec2vec/src/"

SIMILARITY_THRESHOLD="0.6"

conda activate "$BIN_DIR/new_conda_env"

echo "Running spec2vec with a basic split"

cd $BIN_DIR

# python3 eval.py --method 'spec2vec' \
#                 --data $DATA_DIR \
#                 --split_type 'basic' \
#                 --n_most_recent 1

python3 eval.py --method 'spec2vec' \
                --data $DATA_DIR \
                --split_type 'basic' \
                --model "26_04_2024_12_59_34"
