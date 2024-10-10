import os
import argparse

from matchms import Spectrum as matchms_Spectrum
from matchms.importing import load_from_mgf, load_from_msp
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import json
import pickle

from matchms.filtering import select_by_mz, default_filters, add_fingerprint
from matchms.filtering import normalize_intensities, add_parent_mass
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity

from matchms.filtering.metadata_processing.harmonize_undefined_inchikey import harmonize_undefined_inchikey
from matchms.filtering.metadata_processing.harmonize_undefined_inchi import harmonize_undefined_inchi
from matchms.filtering.metadata_processing.harmonize_undefined_smiles import harmonize_undefined_smiles

def load_spectrums(path):
    """Load spectrums from path"""
    if path.endswith(".mgf"):
        spectrums = list(load_from_mgf(path, metadata_harmonization=False))
    elif path.endswith(".msp"):
        spectrums = list(load_from_msp(path, metadata_harmonization=False))
    elif path.endswith(".json"):
        with open(path, "r") as f:
            spectrums = json.load(f)
            # Transform spectra to matchms spectrum objects
            spectrums = [matchms_Spectrum(mz=np.array(v["m/z array"]), 
                                          intensities=np.array(v["intensity array"]), 
                                          metadata={"title": k,
                                                    "spectrum_id": k,
                                                    "precursor_mz": v["precursor mz"]})
                                    for k,v in spectrums.items()]
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return spectrums

def inject_metadata(path, spectra):
    df = pd.read_csv(path)
    
    # We want to get the following columns: Smiles,INCHI,InChIKey_smiles
    df = df[["spectrum_id","Smiles", "INCHI", "InChIKey_smiles", "collision_energy", "GNPS_library_membership", "Ion_Mode"]]
    for spectrum in spectra:
        spectrum_id = spectrum.metadata.get("title")
        row = df.loc[df['spectrum_id'] == spectrum_id]
        if len(row) != 1:
            raise ValueError(f"Expected one row for {spectrum_id} but got {len(row)}") 
        smiles = row["Smiles"].values[0]
        inchi = row["INCHI"].values[0]
        inchikey = row["InChIKey_smiles"].values[0]
        collision_energy = row["collision_energy"].values[0]
        library_membership = row['GNPS_library_membership'].values[0]
        ion_mode = row['Ion_Mode'].values[0]
        
        if smiles is not None:
            spectrum.set("smiles", smiles)
        if inchi is not None:
            spectrum.set("inchi", inchi)
        if inchikey is not None:
            spectrum.set("inchikey", inchikey)
        if collision_energy is not None:
            spectrum.set("collision_energy", collision_energy)
        if library_membership is not None:
            spectrum.set("library_membership", library_membership)
        if ion_mode is not None:
            spectrum.set("ionmode", ion_mode)
        
    return spectra

def high_level_summary(spectra):
    number_of_peaks = np.array([len(s.peaks) for s in spectra])

    print(f"{np.sum(number_of_peaks < 10)} spectra have < 10 peaks")
    print(f"{np.sum(number_of_peaks < 5)} spectra have < 5 peaks")
    print(f"{np.sum(number_of_peaks < 2)} spectra have < 2 peaks")
    print(f"{np.sum(number_of_peaks < 1)} spectra have < 1 peaks")

def minimal_processing(spectrum):
    """
    We will not use this processing because this data should be cleaned
    """
    spectrum = default_filters(spectrum)
    
    spectrum = harmonize_undefined_inchikey(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_smiles(spectrum)
    if spectrum.get("inchikey") == "nan" or spectrum.get("inchi") == "nan" or spectrum.get("smiles") == "nan":
        # In the future, this should raise an error. These should not be in the dataset
        return None
    if pd.isna(spectrum.get("inchikey")) or pd.isna(spectrum.get("inchi")) or pd.isna(spectrum.get("smiles")):
        # In the future, this should raise an error. These should not be in the dataset
        return None
    
    # spectrum = repair_not_matching_annotation(spectrum)
    spectrum = add_parent_mass(spectrum)
    
    if spectrum.get("adduct") in ['[M+CH3COO]-/[M-CH3]-',
                            '[M-H]-/[M-Ser]-',
                            '[M-CH3]-']:
        if spectrum.get("ionmode") != "negative":
            spectrum.set("ionmode", "negative")

    # Check for inchikey
    if not spectrum.get("inchikey"):
        return  None
    spectrum = add_fingerprint(spectrum)
    # If a fingerprint couldn't be generated, discard spectrum
    if spectrum is None:
        return None
    if not spectrum.get("fingerprint") is not None:
        spectrum = None
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_spectra', type=str, default=None)
    parser.add_argument('--train_metadata', type=str, default=None)
    parser.add_argument('--validation_spectra', type=str, default=None)
    parser.add_argument('--validation_metadata', type=str, default=None)    
    parser.add_argument('--test_spectra', type=str, default=None)
    parser.add_argument('--test_metadata', type=str, default=None)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()
    
    training_spectra = list(load_spectrums(args.train_spectra))
    val_spectra = list(load_spectrums(args.validation_spectra))
    test_spectra = list(load_spectrums(args.test_spectra))
    
    # Inject metadata
    training_spectra = inject_metadata(args.train_metadata, training_spectra)
    val_spectra = inject_metadata(args.validation_metadata, val_spectra)
    test_spectra = inject_metadata(args.test_metadata, test_spectra)
    
    # Process the Spectra
    print("Begining preprocessing of spectra")
    print("Starting with", len(training_spectra), "train spectra")
    print("Starting with", len(val_spectra), "validation spectra")
    print("Starting with", len(test_spectra), "test spectra")
    training_spectra = Parallel(n_jobs=-1)(
        delayed(minimal_processing)(s) for s in tqdm(training_spectra)
    )
    val_spectra = Parallel(n_jobs=-1)(
        delayed(minimal_processing)(s) for s in tqdm(val_spectra)
    )
    test_spectra = Parallel(n_jobs=-1)(
        delayed(minimal_processing)(s) for s in tqdm(test_spectra)
    )

    training_spectra = [s for s in training_spectra if s is not None]
    val_spectra = [s for s in val_spectra if s is not None]
    test_spectra = [s for s in test_spectra if s is not None]
    print("Done processing spectra")
    print("Ending with", len(training_spectra), "train spectra")
    print("Ending with", len(val_spectra), "validation spectra")
    print("Ending with", len(test_spectra), "test spectra")

    test_spectrua_represent = {s.get("inchikey")[:14]: s for s in test_spectra}    # Store as a map rather than a set to retain info for debugging
    print(f"Test set contains {len(test_spectrua_represent)} unique inchikeys")
    
    # Must get all inchikeys sims first!
    print("Calculating representative spectra")
    train_spectra_represent = {s.get('inchikey')[:14]: s for s in training_spectra}
    val_spectra_represent = {s.get('inchikey')[:14]: s for s in val_spectra}
        
    # Test data will share tanimoto
    # Checking for overlap, turns out there will be overlap with spectral similarity
    overlaping_inchis = []
    
    # if 'spectral' not in args.train_spectra:
    for key, value in test_spectrua_represent.items():
        if key[:14] in train_spectra_represent:
            overlaping_inchis.append(key)
            
    if len(overlaping_inchis) > 0:
        print("Overlapping inchis:")
        print(pd.Series(overlaping_inchis).value_counts())
    
    train_inchikeys = list(train_spectra_represent.keys())
    val_inchikeys = list(val_spectra_represent.keys())
    test_inchikeys = list(test_spectrua_represent.keys())
    print(f"Found {len(train_inchikeys)} unique train inchikeys")
    print(f"Found {len(val_inchikeys)} unique validation inchikeys")
    print(f"Found {len(test_inchikeys)} unique test inchikeys")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    data_save_path = os.path.join(args.save_dir, "data")
    if not os.path.isdir(data_save_path):
        os.makedirs(data_save_path, exist_ok=True)

    print("Writing data to pickle files...")
    pickle.dump(training_spectra, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_train_split.pickle'), "wb"))
    pickle.dump(val_spectra, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_val_split.pickle'), "wb"))
    pickle.dump(test_spectra, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_test_split.pickle'), "wb"))
    print("/tDone")


    
    
if __name__=="__main__":
    main()
