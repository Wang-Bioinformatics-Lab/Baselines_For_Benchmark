#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "../../src/shared/conda_env/"

# Get data to reproduce structural similarity models from asms

# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output/Structural_Similarity_Prediction/sample_structures_smart_inchikey/ \
#           ./raw/

# # Copy unsplit data for posterity
# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output/summary/Structural_Similarity_Prediction.csv ./raw/org_data/
# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output/spectra/Structural_Similarity_Prediction.mgf ./raw/org_data/
# rsync -rP labvm:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output/json_outputs/Structural_Similarity_Prediction.json ./raw/org_data/

cd ..

### PREPROCESS NEW DATA
echo "----------------------------------------"
echo "PREPROCESSING NEW DATA"
DATA_DIR="../../data/fresh_dataset/raw"
SAVE_DIR="../../data/fresh_dataset/processed/"
BIN_DIR="../src/shared"

cd $BIN_DIR

python3 structural_similarity_preprocessing.py  --train_spectra $DATA_DIR"/train_rows.json" \
                                                --train_metadata $DATA_DIR"/train_rows.csv" \
                                                --validation_spectra $DATA_DIR"/val_rows.json" \
                                                --validation_metadata $DATA_DIR"/val_rows.csv" \
                                                --test_spectra $DATA_DIR"/test_rows.json" \
                                                --test_metadata $DATA_DIR"/test_rows.csv" \
                                                --save_dir $SAVE_DIR