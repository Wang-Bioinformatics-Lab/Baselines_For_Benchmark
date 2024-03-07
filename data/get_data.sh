#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "../src/ms2deepscore/conda_env/"

# Structural Similarity
rsync -rP user@mstrobel.cs.ucr.edu:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output_structural_similarity/ ./structural_similarity
# rsync -rP user@mstrobel.cs.ucr.edu:/home/user/LabData/michael_s/DS_Split/5d/80e8f038bd7f397c7b02e55afbcb06/ ./structural_similarity

python3 get_positive.py --input_dir "./structural_similarity/" \
                        --output_dir "./structural_similarity_positive/"

# # Orbitrap Fragmentation Prediction
# rsync -rP user@mstrobel.cs.ucr.edu:/home/user/SourceCode/GNPS_ML_Processing_Workflow/Train_Test_Splits/nf_output_orbitrap_fragmentation_prediction/ ./orbitrap_fragmentation_prediction

# python3 get_positive.py --input_dir "./orbitrap_fragmentation_prediction/" \
#                         --output_dir "./orbitrap_fragmentation_prediction_positive/"