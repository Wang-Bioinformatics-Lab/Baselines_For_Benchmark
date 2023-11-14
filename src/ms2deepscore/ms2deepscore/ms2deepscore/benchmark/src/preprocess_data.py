from pathlib import Path
import os

from matchms.importing import load_from_mgf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import sys

# sys.path.append("../../../ms2deepscore")

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

import argparse
import vaex
from matchms import Spectrum

def load_data(spectra_path:str, metadata_path:str):
    spectra = vaex.open(spectra_path)
    metadata = vaex.open(metadata_path)

    unique_names = spectra.spectrum_id.unique()
    
    for name in unique_names:
        this_spectrum = spectra[spectra.spectrum_id == name]
        this_metadata = metadata[metadata.spectrum_id == name]
        # metadata will be read as pyarrow types, must be converted to python types
        metadata_dict = {
            "precursor_mz": this_metadata['Precursor_MZ'].values[0].as_py(),
            "collision_energy": this_metadata['collision_energy'].values[0].as_py(),
            "charge": this_metadata['Charge'].values[0].as_py(),
            "smiles": this_metadata.Smiles.values[0].as_py(),
            "InChIKey_inchi": this_metadata.InChIKey_inchi.values[0].as_py(),
            
        }
        matchms_spectrum = Spectrum(mz=this_spectrum.mz.to_numpy(), intensities=this_spectrum.i.to_numpy(), metadata=metadata_dict)
        yield matchms_spectrum

"""
Spectrum(mz=np.array([100, 150, 200.]),
                    intensities=np.array([0.7, 0.2, 0.1]),
                    metadata={'id': 'spectrum1',
                              "peak_comments": {200.: "the peak at 200 m/z"}})

"""

def main():
    parser = argparse.ArgumentParser(description='Train MS2DeepScore model')
    parser.add_argument("--train_data_spectra_path", type=str, help="Path to parquet file containing the spectra", required=True)
    parser.add_argument("--train_data_metadata_path", type=str, help="Path to csv file containing the spectra metadata", required=True) 
    parser.add_argument("--test_data_spectra_path", type=str, help="Path to parquet file containing the spectra", required=True)
    parser.add_argument("--test_data_metadata_path", type=str, help="Path to csv file containing the spectra metadata", required=True)
    args = parser.parse_args()
    
    spectrums_training  = list(load_data(args.train_data_spectra_path, args.train_data_metadata_path))
    spectrums_test      = list(load_data(args.test_data_spectra_path, args.test_data_metadata_path))
    
    # Preprocessing
    spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100.0)
    binned_spectrums_training = spectrum_binner.fit_transform(spectrums_training)
    binned_spectrums_test = spectrum_binner.transform(spectrums_test)
    
    training_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_training])

    same_prob_bins = list(zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1, 10)))
    dimension = len(spectrum_binner.known_bins)
    training_generator = DataGeneratorAllInchikeys(
        binned_spectrums_training, training_inchikeys, tanimoto_scores_df, dim=dimension,
        same_prob_bins=same_prob_bins, num_turns=2, augment_noise_max=10, augment_noise_intensity=0.01)

    test_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectrums_test])
    test_generator = DataGeneratorAllInchikeys(
        binned_spectrums_test, test_inchikeys, tanimoto_scores_df, dim=dimension, same_prob_bins=same_prob_bins,
        num_turns=10, augment_removal_max=0, augment_removal_intensity=0, 
        augment_intensity=0, augment_noise_max=0, use_fixed_set=True)
    
    model = SiameseModel(spectrum_binner, base_dims=(500, 500), embedding_dim=200,
                         dropout_rate=0.2)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    # Save best model and include earlystopping
    earlystopper_scoring_net = EarlyStopping(monitor='val_loss', mode="min", patience=10, verbose=1, restore_best_weights=True)
    
    
if __name__ == "__main__":
    main()