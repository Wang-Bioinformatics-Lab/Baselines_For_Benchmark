#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

BIN_DIR="../../../../src/ms2deepscore/"

SIMILARITY_MODE="structural"
SIMILARITY_THRESHOLD="0.4"

conda activate "$BIN_DIR/conda_env"

cd $BIN_DIR

python3 train.py --train_path "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/data/ALL_GNPS_positive_train_split.pickle" \
                 --val_path "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/data/ALL_GNPS_positive_val_split.pickle" \
                 --tanimoto_path "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/train_tanimoto_df.csv" \
                 --save_dir "./"$SIMILARITY_MODE"_similarity/${SIMILARITY_THRESHOLD}/"