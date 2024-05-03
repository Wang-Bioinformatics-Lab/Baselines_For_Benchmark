#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed"
PAIRS_PATH=$DATA_DIR"/pairs_testing_test_filtered.csv"

SIMILARITY_MODE=$1
SIMILARITY_THRESHOLD=$2

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Inference for MS2DeepScore with a basic split, filtered by pairs"

cd $BIN_DIR

python3 test.py --test_path $DATA_DIR"/data/ALL_GNPS_positive_test_split.pickle" \
                 --tanimoto_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/test_tanimoto_df.csv" \
                 --train_test_similarities "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/train_test_tanimoto_df.csv" \
                 --save_dir "./basic_similarity" \
                 --pairs_path $PAIRS_PATH \
                 --n_most_recent 2