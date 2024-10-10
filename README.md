
### Download Raw and Preprocessed Input Data
```
./download_data.sh
```

## Build the Environment (Mamba Recomended)
```
./build_environment.sh
```

### Example Training
```
cd ./scripts/ms2deepscore/train/fresh_dataset/
./train_all_pairs.sh
```

### Example Testing and Evaluation
Performs inference, and runs evaluations script over predictions.

Please note that this script has an n_jobs parameters to modulate memory and cpu usage.
```
cd scripts/ms2deepscore/test/fresh_dataset/ 
./example.sh
```