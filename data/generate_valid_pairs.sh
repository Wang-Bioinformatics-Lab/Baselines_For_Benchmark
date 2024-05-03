#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "../src/shared/conda_env/"

# python3 generate_valid_pairs.py --input_pickle_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/raw/test_rows.mgf" --output_csv_path "./pairs_testing_test_filtered" --summary_plot_dir "./pairs_test_filtered_plots." 
python3 generate_valid_pairs.py --input_pickle_path "/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/data/structural_similarity/processed/data/ALL_GNPS_positive_test_split.pickle" \
                                --output_csv_path "./structural_similarity/processed/pairs_testing_test_filtered.feather" \
                                --summary_plot_dir "./pairs_testing_test_filtered_plots/" \
                                --prune_duplicate_structures