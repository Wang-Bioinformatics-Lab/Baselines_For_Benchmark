#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "../../src/shared/conda_env/"

# Get data to reproduce structural similarity models from asms

# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output/Structural_Similarity_Prediction/structure_smart/ \
          ./raw/

# Copy unsplit data for posterity
# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/prototyping_data/Structural_Similarity_Prediction.csv ./raw/org_data/
# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/prototyping_data/Structural_Similarity_Prediction.mgf ./raw/org_data/

cd ..
echo "Getting positive data"
# python3 get_positive.py --input_dir "./asms/raw/" \
#                         --output_dir "./asms/raw/positive/"

### PREPROCESS NEW DATA
echo "----------------------------------------"
echo "PREPROCESSING NEW DATA"
POSITIVE_DATA_DIR="../../data/asms/raw/positive/"
DATA_DIR="../../data/asms/raw"
SAVE_DIR="../../data/asms/processed/"
BIN_DIR="../src/shared"

cd $BIN_DIR

python3 structural_similarity_preprocessing.py  --train_spectra $POSITIVE_DATA_DIR"/train_rows.mgf" \
                                                --train_metadata $DATA_DIR"/train_rows.csv" \
                                                --test_spectra $POSITIVE_DATA_DIR"/test_rows.mgf" \
                                                --test_metadata $DATA_DIR"/test_rows.csv" \
                                                --save_dir $SAVE_DIR