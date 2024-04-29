#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

SIMILARITY_MODE=$1
SIMILARITY_THRESHOLD=$2
DATA_DIR="/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed"


conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Inference for MS2DeepScore with a basic split"

cd $BIN_DIR

python3 test.py --test_path $DATA_DIR"/data/ALL_GNPS_positive_test_split.pickle" \
                 --tanimoto_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/test_tanimoto_df.csv" \
                 --train_test_similarities "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/train_test_tanimoto_df.csv" \
                 --save_dir "./basic_similarity/"