#!/usr/bin/env bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

conda activate "/scratch/mstro016/Baselines for Benchmark/src/spec2vec/src/conda_env"

mkdir -p models

echo "Training an MLP on 0.9 similarity threshold data"
python3 MLP_structural_sim.py --train_data './embedded_spectra/train_rows_spectral_0.9' \
                            --test_data './embedded_spectra/test_rows_spectral' \
                            --similarity_path '../../../data/Structural_Similarity_Prediction_Pairs.json' \
                            --model_path './models/MLP_structural_sim_0.9' \
                            --num_epochs 10 \
                            --device 'cuda' \
                            --batch_size 10 \
                            --mode 'train'