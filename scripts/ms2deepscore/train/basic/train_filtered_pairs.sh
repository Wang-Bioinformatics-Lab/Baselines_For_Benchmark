#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Training for MS2DeepScore with a basic split"


cd $BIN_DIR

python3 train.py --train_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/data/ALL_GNPS_positive_train_split.pickle" \
                 --val_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/data/ALL_GNPS_positive_val_split.pickle" \
                 --tanimoto_path "./basic_similarity/train_tanimoto_df.csv" \
                 --train_pairs_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/pairs_train_filtered.feather" \
                 --val_pairs_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/pairs_val_filtered.feather" \
                 --save_dir "./basic_similarity/"