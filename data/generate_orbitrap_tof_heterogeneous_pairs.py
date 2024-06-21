import argparse
import os
import gc
import pickle
import pandas as pd
import numpy as np
from matchms.Spectrum import Spectrum
from matchms.filtering.metadata_processing.add_fingerprint import add_fingerprint
from matchms.similarity import FingerprintSimilarity, ModifiedCosine, CosineGreedy
from matchms.filtering import remove_peaks_around_precursor_mz, remove_peaks_outside_top_k
from matchms.filtering.metadata_processing.derive_inchi_from_smiles import derive_inchi_from_smiles 
from matchms.filtering.metadata_processing.derive_inchikey_from_inchi import derive_inchikey_from_inchi
from pandarallel import pandarallel
from parallel_pandas import ParallelPandas
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
from joblib import Parallel, delayed

from generate_valid_pairs import enrich_pair_dict, produce_summary_plots

def generate_pairs(spectra:list, output_feather_path:str, prune:bool=False, no_cosine:bool=False):
    """ This function takes a list of matchms.Spectrum objects whose metadata contains
    'ms_mass_analyzer', and 'ms_ionisation' keys. It generates a feather file with all possible
    pairs of spectra whose mass analyzer and ionisation match.
    
    Parameters:
    - spectra: list of matchms.Spectrum objects
    - output_feather_path: str, path to the output feather file
    - prune: bool, whether to remove duplicate structures
    
    Returns:
    None
    """
    
    output_lst = []
    
    filtered_by_mass_analyzer = 0
    filtered_by_ionisation = 0
    filtered_by_adduct = 0
    filtered_by_precursor_mz = 0
    filtered_by_no_ce = 0
    filtered_by_mismatched_ce = 0
    
    # TEMP
    # Get unique structures only
    # spectra = {s.metadata.get('inchikey')[:14]: s for s in spectra}
    # spectra = list(spectra.values())    
    
    # Filter spectra for Cosine
    # spectra = [remove_peaks_around_precursor_mz(s, 17) for s in spectra]
    # spectra = [remove_peaks_outside_top_k(s, k=6, mz_window=25) for s in spectra]
    # spectra = [x for x in spectra if x is not None]
    spectra = [remove_peaks_around_precursor_mz(s, 0.1) for s in spectra]
        
    # Create a dataframe with the metadata and spectrum objects
    spectrum_df = pd.DataFrame([{'spectrum_id': s.metadata['spectrum_id'],
                                'ms_mass_analyzer': s.metadata.get('ms_mass_analyzer'), 
                                'ms_ionisation': s.metadata.get('ms_ionisation'), 
                                'adduct': s.metadata['adduct'], 
                                'collision_energy': s.metadata.get('collision_energy'), 
                                'precursor_mz': s.metadata.get('precursor_mz'), 
                                } for s in spectra])
    
    spectrum_df.collision_energy = spectrum_df.collision_energy.astype('float16')
    
    spectrum_df['ms_mass_analyzer'] = spectrum_df['ms_mass_analyzer'].str.lower()

    # Remove everything without a ms_mass_analyzer, ms_ionisation
    print("Removing Spectra Without Mass Analyzer or Ionisation...", flush=True)
    print("Original Length: ", len(spectrum_df))
    spectrum_df = spectrum_df.dropna(subset=['ms_mass_analyzer', 'ms_ionisation'])
    print("New Length: ", len(spectrum_df))
    
    # Get only tof, qtof, and orbitrap spectra
    print("Filtering for TOF, QTOF, and Orbitrap...", flush=True)
    spectrum_df = spectrum_df.loc[spectrum_df['ms_mass_analyzer'].isin(['tof', 'qtof', 'orbitrap'])]
    print("New Length: ", len(spectrum_df))
    print("Mass analyzer value counts: ", spectrum_df['ms_mass_analyzer'].value_counts())
    
    # Cast columns to categorical to save memory
    spectrum_df['adduct'] = spectrum_df['adduct'].astype('category')
    spectrum_df['ms_mass_analyzer'] = spectrum_df['ms_mass_analyzer'].astype('category')
    spectrum_df['ms_ionisation'] = spectrum_df['ms_ionisation'].astype('category')

    # Note that we omit mass analyzer from the groupby, this is for the heterogeneous pairs
    spectrum_df = spectrum_df.groupby(['ms_ionisation', 'adduct'], observed=True)
        
    print("Generating pairs...", flush=True)

    # For each subset of spectra with the same mass analyzer, ionisation, and adduct, generate pairs
    output_lst = []
    print(f"Total Groups: {len(spectrum_df)}")
    
    for _, group in tqdm(spectrum_df):
        # print(group)
        # Get all pairs with precursor mz difference less than 200 using vectorized operations
        pairs = group.merge(group, on=['ms_ionisation', 'adduct'], suffixes=('_1', '_2'))   # Will perform outer product
        pairs = pairs.loc[pairs['spectrum_id_1'] < pairs['spectrum_id_2']]                                      # Get upper diagonal
        pairs['precursor_mass_difference'] = (pairs['precursor_mz_1'] - pairs['precursor_mz_2']).abs().astype('float16')
        pairs = pairs.loc[abs(pairs['precursor_mz_1'] - pairs['precursor_mz_2']) < 200]
        # If they have collision energy, filter it out if the difference is greater than 5, if they don't just let them through
        # DEBUG TO SEE IF WE CAN JUST ONLY USE COLLISION ENERGY PAIRS
        org_len = len(pairs)
        ce_pairs    = pairs.loc[(pairs['collision_energy_1'].notna() & pairs['collision_energy_2'].notna())]
        ce_pairs    = ce_pairs.loc[(pairs['collision_energy_1'] - ce_pairs['collision_energy_2']).abs() <= 5]
        other_pairs = pairs.loc[pairs['collision_energy_1'].isna() | pairs['collision_energy_2'].isna()]
        pairs = pd.concat((ce_pairs, other_pairs))
        # pairs = ce_pairs
        new_len = len(pairs)
        filtered_by_mismatched_ce += org_len - new_len
        del ce_pairs, other_pairs
        
        output_lst.extend(pairs[['spectrum_id_1', 'spectrum_id_2', 'ms_mass_analyzer_1', 'ms_mass_analyzer_2', 'ms_ionisation', 'precursor_mass_difference',]].to_dict('records'))
        del pairs
        gc.collect()
    
    del spectrum_df
    gc.collect()
    
    # print(f"Filtered by mass analyzer: {filtered_by_mass_analyzer}")
    # print(f"Filtered by ionisation: {filtered_by_ionisation}")
    # print(f"Filtered by adduct: {filtered_by_adduct}")
    # print(f"Filtered by precursor mz: {filtered_by_precursor_mz}")
    # print(f"Filtered by no collision energy: {filtered_by_no_ce}")
    print(f"Filtered by mismatched collision energy: {filtered_by_mismatched_ce}")
    
    columns=['spectrum_id_1', 'spectrum_id_2', 'ms_mass_analyzer_1', 'ms_mass_analyzer_2', 'ms_ionisation', 'precursor_mass_difference']
    output_df = pd.DataFrame(output_lst, columns=columns)
    output_df['spectrum_id_1'] = output_df['spectrum_id_1'].astype('category')
    output_df['spectrum_id_2'] = output_df['spectrum_id_2'].astype('category')
    output_df['ms_mass_analyzer_1'] = output_df['ms_mass_analyzer_1'].astype('category')
    output_df['ms_mass_analyzer_2'] = output_df['ms_mass_analyzer_2'].astype('category')
    output_df['ms_ionisation'] = output_df['ms_ionisation'].astype('category')
    output_df['precursor_mass_difference'] = output_df['precursor_mass_difference'].astype('float16')
    
    del output_lst
    gc.collect()
    
    # Prune pairs
    # We want to keep one spectrum id for each structure so we'll count take one with the highest pair count
    if prune:
        print("Pruning pairs...", flush=True)
        org_len = len(output_df)
        spectra = [derive_inchi_from_smiles(s) for s in spectra]
        spectra = [derive_inchikey_from_inchi(s) for s in spectra]
        spectrum_inchikey14_mapping = {s.metadata['spectrum_id']: s.metadata['inchikey'][:14] for s in spectra}
        
        all_spectrum_ids = pd.concat((output_df['spectrum_id_1'], output_df['spectrum_id_2']))
        spectrum_counts = all_spectrum_ids.value_counts()
        print(spectrum_counts)
        spectrum_counts = spectrum_counts.reset_index()
        print(spectrum_counts)
        spectrum_counts.columns = ['spectrum_id', 'count']
        spectrum_counts['inchikey_14'] = spectrum_counts['spectrum_id'].map(spectrum_inchikey14_mapping)
        spectrum_counts = spectrum_counts.groupby('inchikey_14').apply(lambda x: x.sort_values('count', ascending=False).iloc[0])
        
        print(spectrum_counts)
        
        output_df = output_df.loc[output_df['spectrum_id_1'].isin(spectrum_counts['spectrum_id']) & output_df['spectrum_id_2'].isin(spectrum_counts['spectrum_id'])]
        print(f"Pruned {org_len - len(output_df)} pairs", flush=True)
        del spectrum_counts, all_spectrum_ids, spectrum_inchikey14_mapping
        gc.collect()
    
    # Before enrichment, add fingerprints to the spectra
    print("Adding fingerprints to spectra...", flush=True)
    spectra = [add_fingerprint(s) for s in spectra]

    spectrum_mapping = {s.metadata['spectrum_id']: s for s in spectra}
    for s in spectrum_mapping.values():
        assert isinstance(s, Spectrum)
    
    output_df = enrich_pair_dict(output_df, spectrum_mapping, no_cosine)
    
    print("Saving pairs...", flush=True)
    # Ensure path exists
    if not os.path.exists(os.path.dirname(output_feather_path)):
        os.makedirs(os.path.dirname(output_feather_path), exist_ok=True)
    
    output_df.to_feather(output_feather_path)
    

def main():
    parser = parser = argparse.ArgumentParser(description='Generate valid pairs')
    parser.add_argument('--input_pickle_path', type=str, help='Input file')
    parser.add_argument('--output_feather_path', type=str, help='Output file')
    parser.add_argument('--summary_plot_dir', type=str, help='Path to the directory where summary plots will be saved', default=None)
    parser.add_argument('--prune_duplicate_structures', action='store_true', help='Prune duplicate structures', default=False)
    parser.add_argument('--no_cosine', action='store_true', help='Do not compute cosine similarity', default=False)
    args = parser.parse_args()
    
    print("Loading spectra...", flush=True)
    if args.input_pickle_path.endswith('.pickle'):
        spectra = pickle.load(open(args.input_pickle_path, 'rb'))
    elif args.input_pickle_path.endswith('.mgf'):
        from matchms.importing import load_from_mgf

        spectra = list(load_from_mgf(args.input_pickle_path))
    else:
        raise ValueError("Input file must be a pickle or mgf file")
    
    generate_pairs(spectra, args.output_feather_path, args.prune_duplicate_structures, args.no_cosine)
    
    if not os.path.exists(args.summary_plot_dir):
        os.makedirs(args.summary_plot_dir, exist_ok=True)
    
    if args.summary_plot_dir is not None:
        produce_summary_plots(args.output_feather_path, args.summary_plot_dir, args.no_cosine)
        

if __name__ == "__main__":
    gc.enable()
    main()