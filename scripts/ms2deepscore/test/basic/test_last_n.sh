#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

SIMILARITY_MODE=$1
SIMILARITY_THRESHOLD=$2

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Inverence for MS2DeepScore with a basic split"

cd $BIN_DIR

python3 test.py --test_path "./basic_similarity/data/ALL_GNPS_positive_test_split_01082024.pickle" \
                 --tanimoto_path "./basic_similarity/test_tanimoto_df.csv" \
                 --train_test_similarities "./basic_similarity/train_test_tanimoto_df.csv" \
                 --save_dir "./basic_similarity/" \
                 --n_most_recent $1 \