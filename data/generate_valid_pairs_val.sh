#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "../src/shared/conda_env/"

python3 generate_valid_pairs.py --input_pickle_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/data/ALL_GNPS_positive_val_split.pickle" \
                                --output_feather_path "./structural_similarity/processed/pairs_val_filtered.feather" \
                                --summary_plot_dir "./pairs_testing_val_filtered_plots/" \
                                --no_cosine