#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

PROCESSED_DIR="../../../data/structural_similarity/processed/data"
DATA_DIR="../../../data/structural_similarity/"
BIN_DIR="../../../src/spec2vec/src/"

SIMILARITY_THRESHOLD="0.6"

conda activate "$BIN_DIR/new_conda_env"

echo "Running spec2vec with a basic split, not witholding test set."

cd $BIN_DIR

python3 runner.py --spectra_dir $PROCESSED_DIR \
                 --num_cpus 28 \
                 --withold_test False \
                 --save_path "./basic/"
