#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

SIMILARITY_MODE="spectral"
SIMILARITY_THRESHOLD="0.8"

conda activate "$BIN_DIR/conda_env"

cd $BIN_DIR

python3 test.py --test_path "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/data/ALL_GNPS_positive_test_split_01082024.pickle" \
                 --tanimoto_path "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/tanimoto_df.csv" \
                 --save_dir "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/"