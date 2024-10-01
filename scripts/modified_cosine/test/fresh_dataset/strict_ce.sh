#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

FOLDER="fresh_dataset"
DATASET="test_pairs_filtered_strict_ce"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="../../data/$FOLDER/"

conda activate "$BIN_DIR/../shared/new_conda_env"

echo "Performing Inference for MS2DeepScore with a basic split, filtered by pairs"

cd $BIN_DIR

python3 test_presampled.py --test_path $DATA_DIR"/raw/test_rows.json" \
                            --metadata_path $DATA_DIR"/raw/test_rows.csv" \
                            --presampled_pairs_path $DATA_DIR"/raw/test_pairs/"$DATASET".parquet" \
                            --save_dir "./$FOLDER" \
                            --n_jobs 32 \
                            --modified_cosine

DIR_NAME=$FOLDER"/"
BASE_NAME_WITHOUT_EXT="modified_cosine"

echo "Looking for predictions in" "$DIR_NAME/$BASE_NAME_WITHOUT_EXT/"$DATASET"/dask_output.parquet"

python3 ../shared/metrics_and_plotting.py --prediction_path "$DIR_NAME/$BASE_NAME_WITHOUT_EXT/"$DATASET"/dask_output.parquet" \
                                            --save_dir "$DIR_NAME/$BASE_NAME_WITHOUT_EXT/"$DATASET"/" \
                                            --n_jobs 4
