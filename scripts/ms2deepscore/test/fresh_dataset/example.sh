#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

FOLDER="fresh_dataset"
DATASET="test_pairs_filtered"

BIN_DIR="../../../../src/ms2deepscore/"

DATA_DIR="../../data/$FOLDER/"

echo "Using model: $MODEL_PATH"

conda activate "$BIN_DIR/../shared/conda_env"

echo "Performing Inference for MS2DeepScore with a basic split, filtered by pairs"

cd $BIN_DIR

# Get first arg as model name e.g., model_30_08_2024_14_27_50.hdf5
if [ "$#" -eq 1 ]; then
    MODEL_PATH="./$FOLDER/$1"
else
    echo "No model file provided. Searching for a model file in the folder: $FOLDER"
    # Get the first file that starts with 'model_' and ends with '.hdf5'
    MODEL_PATH=$(find "$FOLDER" -type f -name 'model_*.hdf5' | head -n 1)
    
    if [ -z "$MODEL_PATH" ]; then
        echo "No model file found."
        exit 1
    fi
    echo "Using model: $MODEL_PATH"
fi

# Expect n * 16 GB of memory usage
python3 test_presampled.py --test_path $DATA_DIR"/processed/data/ALL_GNPS_positive_test_split.pickle" \
                            --presampled_pairs_path $DATA_DIR"/raw/test_pairs/"$DATASET".parquet" \
                            --save_dir "./$FOLDER" \
                            --n_jobs 2 \
                            --model_path $MODEL_PATH

DIR_NAME=$(dirname "$MODEL_PATH")
BASE_NAME_WITHOUT_EXT=$(basename "$MODEL_PATH" .hdf5)

# Expect n * 4 GB of memory usage 
python3 ../shared/metrics_and_plotting.py --prediction_path "$DIR_NAME/$BASE_NAME_WITHOUT_EXT/"$DATASET"/dask_output.parquet" \
                                            --save_dir "$DIR_NAME/$BASE_NAME_WITHOUT_EXT/"$DATASET"/" \
                                            --n_jobs 12 \
                                            --skip_titrated_rank_metrics