import argparse
import os
from datetime import datetime

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import pickle

def main():
    parser = argparse.ArgumentParser(description='Train MS2DeepScore on the original data')
    parser.add_argument('--train_path', type=str, help='Path to the training data')
    parser.add_argument('--val_path', type=str, help='Path to the validation data')
    parser.add_argument('--tanimoto_path', type=str, help='Path to the tanimoto scores')
    parser.add_argument('--save_dir', type=str, help='Path to the save directory')
    args = parser.parse_args()
        
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
    
    print("Loading data...", flush=True)
    spectrums_training = pickle.load(open(args.train_path, "rb"))
    spectrums_val = pickle.load(open(args.val_path, "rb"))
    tanimoto_df = pd.read_csv(args.tanimoto_path, index_col=0)
    print("\tDone.", flush=True)
    
    print("Binning Spectra...", flush=True)
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(spectrums_training)
    binned_spectrums_val = spectrum_binner.transform(spectrums_val)
    print("\tDone.", flush=True)

    training_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_training])
    print(len(training_inchikeys))

    # tanimoto_scores_df = pickle.load(open("./data/similarities_train_pos.pkl", "rb"))
    # print("Total number of inchikey14s:", len(tanimoto_scores_df.columns))

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
    dimension = len(spectrum_binner.known_bins)
    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training, tanimoto_df, spectrum_binner=spectrum_binner, selected_inchikeys=training_inchikeys,
        same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)

    validation_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_val])

    # tanimoto_scores_df = pickle.load(open("./data/similarities_test_pos.pkl", "rb"))

    validation_generator = DataGeneratorAllInchikeys(
        binned_spectrums_val, tanimoto_df, spectrum_binner=spectrum_binner, selected_inchikeys=validation_inchikeys, same_prob_bins=same_prob_bins,
        num_turns=10, augment_removal_max=0, augment_removal_intensity=0, 
        augment_intensity=0, augment_noise_max=0, use_fixed_set=True)

    model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                            dropout_rate=0.2)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    # Save best model and include earlystopping
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1, restore_best_weights=True)

    history = model.model.fit(training_generator, validation_data=validation_generator,
                            epochs=150, verbose=1, callbacks=[earlystopper_scoring_net])

    # history = model.model.fit(training_generator, validation_data=validation_generator,
    #                         epochs=50, verbose=1)

    # Get current time in dd_mm_yyyy_hh_mm_ss format
    current_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")   

    model_file_name = os.path.join(args.save_dir, f"model_{current_time}.hdf5")
    model.save(model_file_name)

    # Save history to pickle
    pickle.dump(history.history, open(os.path.join(args.save_dir, f"model_{current_time}_history.pkl"), "wb"))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join(args.save_dir, "loss.png"))

if __name__ == "__main__":
    main()