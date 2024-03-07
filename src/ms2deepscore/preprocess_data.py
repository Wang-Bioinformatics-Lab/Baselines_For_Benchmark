from pathlib import Path
import os
import argparse

import matchms
from matchms.importing.load_from_msp import parse_msp_file
from matchms.importing import load_from_mgf, load_from_msp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from matchms.filtering import metadata_processing

from matchms.filtering import select_by_mz, default_filters, add_fingerprint
from matchms.filtering import normalize_intensities, add_parent_mass, derive_adduct_from_name
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity

from matchms.filtering.metadata_processing.repair_not_matching_annotation import repair_not_matching_annotation
from matchms.filtering.metadata_processing.derive_inchi_from_smiles import derive_inchi_from_smiles
from matchms.filtering.metadata_processing.derive_inchi_from_smiles  import derive_inchi_from_smiles
from matchms.filtering.metadata_processing.derive_inchikey_from_inchi import derive_inchikey_from_inchi
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import is_valid_inchikey
from matchms.filtering.metadata_processing.harmonize_undefined_inchikey import harmonize_undefined_inchikey
from matchms.filtering.metadata_processing.harmonize_undefined_inchi import harmonize_undefined_inchi
from matchms.filtering.metadata_processing.harmonize_undefined_smiles import harmonize_undefined_smiles
from matchms.filtering.metadata_processing.repair_inchi_inchikey_smiles import repair_inchi_inchikey_smiles

from matchms.similarity import FingerprintSimilarity

import tensorflow as tf
print("Is GPU Available:", tf.test.is_gpu_available())

def load_spectrums(path):
    """Load spectrums from path"""
    if path.endswith(".mgf"):
        spectrums = list(load_from_mgf(path, metadata_harmonization=False))
    elif path.endswith(".msp"):
        spectrums = list(load_from_msp(path, metadata_harmonization=False))
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return spectrums

def inject_metadata(path, spectra):
    df = pd.read_csv(path)
    
    # We want to get the following columns: Smiles,INCHI,InChIKey_smiles
    df = df[["spectrum_id","Smiles", "INCHI", "InChIKey_smiles"]]
    for spectrum in spectra:
        spectrum_id = spectrum.metadata.get("title")
        row = df.loc[df['spectrum_id'] == spectrum_id]
        if len(row) != 1:
            raise ValueError(f"Expected one row for {spectrum_id} but got {len(row)}") 
        smiles = row["Smiles"].values[0]
        inchi = row["INCHI"].values[0]
        inchikey = row["InChIKey_smiles"].values[0]
        
        # if not isinstance(smiles, str):
        #     print("Smiles is not a string", smiles)
        #     smiles = None
        # if not isinstance(inchi, str):
        #     print("Inchi is not a string", inchi)
        #     inchi = None
        # if not isinstance(inchikey, str) :
        #     print("Inchikey is not a string", inchikey)
        #     inchikey = None
        if smiles is not None:
            spectrum.set("smiles", smiles)
        if inchi is not None:
            spectrum.set("inchi", inchi)
        if inchikey is not None:
            spectrum.set("inchikey", inchikey)
    
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
    # # If spectrum charge > 0, set ion mode to positive
    # if '+' in spectrum.get("charge"):
    #     spectrum.set("ionmode", "positive")
    # # If spectrum charge < 0, set ion mode to negative
    # if '-' in spectrum.get("charge"):
    #     spectrum.set("ionmode", "negative")
    # if '+' in spectrum.get("charge") and '-' in spectrum.get("charge"):
    #     raise ValueError("Spectrum has both positive and negative charges")
    
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
    
    # Make sure ion mode is positive
    if spectrum.get("ionmode") != "positive":
        print(f"Ion mode is not positive, got {spectrum.get('ionmode')} instead, charge is {spectrum.get('charge')}")
        return None
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

def more_minimal_processing(spectrum):
    spectrum = add_fingerprint(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    
    if spectrum is None:
        return spectrum
    
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return  None
    
    if spectrum.get("adduct") in ['[M+CH3COO]-/[M-CH3]-',
                            '[M-H]-/[M-Ser]-',
                            '[M-CH3]-']:
        if spectrum.get("ionmode") != "negative":
            spectrum.set("ionmode", "negative")
            
    if spectrum.get("ionmode") != "positive":
       return None
    return spectrum

def main():
    # TODO List
    # * Validate which pre-processing to use
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_spectra', type=str, default=None)
    parser.add_argument('--train_metadata', type=str, default=None)
    # This script will overwrite the test spectra if run for each split, but they're all the same so we'll ignore for now
    parser.add_argument('--test_spectra', type=str, default=None)
    parser.add_argument('--test_metadata', type=str, default=None)
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()
    
    if args.train_spectra and not args.test_spectra:
        print("When using training spectra, test spectra must be supplied to ensure datasets are disjoint in inchikey.")
        
    if args.train_metadata and not args.test_metadata:
        print("When using training metadata, test metadata must be supplied to ensure datasets are disjoint in inchikey.")
    
    training_spectra = list(load_spectrums(args.train_spectra))     # If not converted to a list, in-place operations will not apply
    
    # Inject metadata
    training_spectra = inject_metadata(args.train_metadata, training_spectra)
    
    # Process the Spectra
    print("Begining preprocessing of spectra")
    spectrums_pos_processing = Parallel(n_jobs=-1)(
        delayed(minimal_processing)(s) for s in tqdm(training_spectra)
    )

    spectrums_pos_processing = [s for s in spectrums_pos_processing if s is not None]
    print(f"{len(spectrums_pos_processing)} spectrums after processing")
    # print(dir(spectrums_pos_processing[0]))
    # print(spectrums_pos_processing[0].to_dict())

    assert all([s.get("ionmode") == "positive" for s in spectrums_pos_processing]), "Not all spectrums are positive"
    
    test_path = args.test_spectra

    test_spectra = list(load_from_mgf(test_path, metadata_harmonization=False))
    
    # Inject metadata
    test_spectra = inject_metadata(args.test_metadata, test_spectra)
    
    print("Processing Test Set")
    print("Original Spectra Count:", len(test_spectra))
    # Apply basic filters to rectify smiles and inchikey 
    test_spectra = Parallel(n_jobs=-1)(
        delayed(minimal_processing)(s) for s in tqdm(test_spectra)
    )
    test_spectra = [s for s in test_spectra if s is not None]
    print(f"Test set contains {len(test_spectra)} spectra after processing")
    test_mapping = {s.get("inchikey")[:14]: s for s in test_spectra}    # Store as a map rather than a set to retain info for debugging
    print(f"Test set contains {len(test_mapping)} unique inchikeys")
    
    # Must get all inchikeys sims first!
    print("Calculating representative spectra")
    spectrums_represent = {s.get('inchikey')[:14]: s for s in spectrums_pos_processing}
    # Test data will share tanimoto
    # Checking for overlap, turns out there will be overlap with spectral similarity
    overlaping_inchis = []
    
    # if 'spectral' not in args.train_spectra:
    for key, value in test_mapping.items():
        # assert not (key[:14] in spectrums_represent), f"Found overlap for innchikey {key}. Test ID {value.metadata['title']}, Train ID {spectrums_represent[key[:14]].metadata['title']}"
        # We use blank spectra here, which is intentional. The test set is allowed to contain multiple spectra for each inchikey
        # spectrums_represent[key[:14]] = matchms.Spectrum(mz=np.array([], dtype=float), intensities=np.array([], dtype=float), metadata={"inchikey": key})
        overlaping_inchis.append(key)
            
    if len(overlaping_inchis) > 0:
        print(pd.Series(overlaping_inchis).value_counts())
            
    all_inchis = list(spectrums_represent.keys())
    spectrums_represent = list(spectrums_represent.values())
    print(f"Found {len(spectrums_represent)} unique inchikeys")

    similarity_measure = FingerprintSimilarity(similarity_measure="jaccard")
    scores_mol_similarity = similarity_measure.matrix(spectrums_represent, spectrums_represent)
    tanimoto_df = pd.DataFrame(scores_mol_similarity, columns=all_inchis, index=all_inchis)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    tanimoto_df_output_path = os.path.join(args.save_dir, "tanimoto_df.csv")
    tanimoto_df.to_csv(tanimoto_df_output_path)
    
    # Create our own validation split
    np.random.seed(100)
    inchikeys14  = tanimoto_df.index.to_numpy()
    testIDs      = np.where(np.isin(inchikeys14, list(test.mapping.keys())))[0].tolist()
    inchikey_ids = set(np.arange(inchikeys14.shape[0])) - set(testIDs)

    # Original Split was 14061 500 500

    # Select training, validation, and test IDs:
    # We'll take 500 inchikeys for validation
    n_val = 500
    valIDs = list(set(np.random.choice(list(set(inchikey_ids)), n_val, replace=False)))
    trainIDs = list(set(inchikey_ids)  - set(valIDs))
    
    # quick check to see if there's indeed no overlap
    for idx in trainIDs:
        assert not (idx in valIDs), f"Found overlap for ID {idx}"
        assert not (idx in testIDs), f"Found overlap for ID {idx}"
    
    inchikeys14_training = tanimoto_df.index.to_numpy()[trainIDs]

    spectrums_training = [s for s in spectrums_pos_processing if s.get("inchikey")[:14] in inchikeys14_training]
    print(f"{len(spectrums_training)} spectrums in training data")

    inchikeys14_val = tanimoto_df.index.to_numpy()[valIDs]

    spectrums_val = [s for s in spectrums_pos_processing if s.get("inchikey")[:14] in inchikeys14_val]
    print(f"{len(spectrums_val)} spectrums in validation data.")

    inchikeys14_test = tanimoto_df.index.to_numpy()[testIDs]

    # Checking for overlap, turns out there will be overlap with spectral similarity
    if 'spectral' not in args.train_spectra:
        # Check no test overlap
        for inchi in inchikeys14_test:
            assert not (inchi in inchikeys14_training), f"Found overlap for ID {inchi}"
            assert not (inchi in inchikeys14_val), f"Found overlap for ID {inchi}"
            
        known_test_inchis = [s.get("inchikey")[:14] for s in test_spectra]
        for inchi in known_test_inchis:
            assert not (inchi in inchikeys14_training), f"Found overlap for ID {inchi}"
            assert not (inchi in inchikeys14_val), f"Found overlap for ID {inchi}"
            
        for inchi in inchikeys14_test:
            assert inchi in known_test_inchis, f"Found overlap for ID {inchi}"
    

    spectrums_test = test_spectra #[s for s in spectrums_pos_processing if s.get("inchikey")[:14] in inchikeys14_test]
    print(f"{len(spectrums_test)} spectrums in test data.")

    spectrums_wo_test = spectrums_training + spectrums_val
    print(f"{len(spectrums_wo_test)} spectrums in data w/o test")

    data_save_path = os.path.join(args.save_dir, "data")
    if not os.path.isdir(data_save_path):
        os.makedirs(data_save_path, exist_ok=True)

    import pickle
    print("Writing data to pickle files...")
    pickle.dump(spectrums_wo_test, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_wo_test_split.pickle'), "wb"))

    pickle.dump(spectrums_training, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_train_split.pickle'), "wb"))

    pickle.dump(spectrums_val, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_val_split.pickle'), "wb"))

    pickle.dump(spectrums_test, 
                open(os.path.join(data_save_path,'ALL_GNPS_positive_test_split_01082024.pickle'), "wb"))
    print("/tDone")


    
    
if __name__=="__main__":
    main()


