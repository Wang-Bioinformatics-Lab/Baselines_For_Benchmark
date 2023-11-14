#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "/scratch/mstro016/Baselines for Benchmark/src/spec2vec/src/conda_env"

echo "Running spec2vec with 0.9 similarity threshold"
python3 runner.py --train_spectra '../../../data/nf_output/train_rows_spectral_0.9.mgf' \
                 --train_metadata '../../../data/nf_output/train_rows_spectral_0.9.csv' \
                 --test_spectra '../../../data/nf_output/test_rows_spectral.mgf' \
                 --test_metadata '../../../data/nf_output/test_rows_spectral.csv' \
                 --num_cpus 8 \
                 --withold_test False

# echo "Running spec2vec with 0.8 similarity threshold"
# python3 runner.py --train_spectra '../../../data/nf_output/train_rows_spectral_0.8.mgf' \
#                  --train_metadata '../../../data/nf_output/train_rows_spectral_0.8.csv' \
#                  --num_cpus 8 \
#                  --withold_test False

# echo "Running spec2vec with 0.7 similarity threshold"
# python3 runner.py --train_spectra '../../../data/nf_output/train_rows_spectral_0.7.mgf' \
#                  --train_metadata '../../../data/nf_output/train_rows_spectral_0.7.csv' \
#                  --num_cpus 8 \
#                  --withold_test False

# echo "Running spec2vec with 0.6 similarity threshold"
# python3 runner.py --train_spectra '../../../data/nf_output/train_rows_spectral_0.6.mgf' \
#                  --train_metadata '../../../data/nf_output/train_rows_spectral_0.6.csv' \
#                  --num_cpus 8 \
#                  --withold_test False