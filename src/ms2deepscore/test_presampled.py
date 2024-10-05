from pathlib import Path
import os
import gc
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import json
import tensorflow as tf
import pandas as pd
import numpy as np

from ms2deepscore import MS2DeepScore
from custom_model_loader import load_model  # Allows kwargs
from ms2deepscore.vector_operations import cosine_similarity
from matchms.similarity.ModifiedCosine import ModifiedCosine
from matchms.importing import load_from_mgf, load_from_msp
from matchms import Spectrum as matchms_Spectrum

import dask

import dask.dataframe as dd
import dask.array as da
from dask.distributed import LocalCluster
from time import time

from train_presampled import biased_loss

from dask.diagnostics import ProgressBar
PBAR = ProgressBar()
PBAR.register()

sys.path.append('../shared')
from utils import train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot, \
                    fixed_tanimoto_train_test_similarity_dependent, \
                    pairwise_train_test_dependent_heatmap
                    
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18, #
    "ytick.labelsize": 18, #
    "legend.fontsize": 18, #
    "figure.titlesize": 20,
    "legend.title_fontsize": 20, #
    "figure.autolayout": True,
    "figure.dpi": 300,
    })


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
    if path.endswith(".pkl") or path.endswith(".pickle"):
        with open(path, "rb") as f:
            spectrums = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    return spectrums


def _dask_cosine(df, embedding_dict):
    def _helper(row):
        if row.spectrumid1 not in embedding_dict or row.spectrumid2 not in embedding_dict:
            return -1.0
        return cosine_similarity(embedding_dict[row.spectrumid1], embedding_dict[row.spectrumid2])
    out = df.apply(_helper, axis=1)
    # For some reason out is a dataframe with a single column, rather than a series
    out = out.rename('predicted_similarity')
    if isinstance(out, pd.DataFrame):
        return pd.Series(out.predicted_similarity)
    else:
        return pd.Series(out)

def _dask_modified_cosine(df, spectrum_dict):
    metric = ModifiedCosine(tolerance=0.10, mz_power=0.0, intensity_power=0.50)
    def _helper(row):
        if row.spectrumid1 not in spectrum_dict or row.spectrumid2 not in spectrum_dict:
            return -1.0, -1.0
        # Tuple of "score", "matched_peaks"
        o = metric.pair(spectrum_dict[row.spectrumid1], spectrum_dict[row.spectrumid2])
        return o['score'].item(), o['matches'].item()
    out = df.apply(_helper, axis=1)
    df['predicted_similarity'] = out.apply(lambda x: x[0])
    df['matched_peaks'] = out.apply(lambda x: x[1])
    return df

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

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')
    parser.add_argument('--metadata_path', type=str, help='Path to the metadata file (optional)')
    parser.add_argument('--presampled_pairs_path', type=str, help='Path to dask parquet file of presampled pairs')
    parser.add_argument("--save_dir", type=str, help="Path to the model")
    parser.add_argument('--modified_cosine', action='store_true', help='Use modified cosine rather than a provided model')
    parser.add_argument("--model_path", type=str, help="Path to the model, overrides n_most_recent", default=None)
    parser.add_argument("--save_dir_insert", type=str, help="Appended to save dir, to help organize test sets", default="")
    parser.add_argument("--n_most_recent", type=int, help="Number of most recent models to evaluate", default=None)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    args = parser.parse_args()
    
    grid_fig_size = (10,10)

    if args.model_path is not None and args.modified_cosine:
        raise ValueError("Cannot use modified cosine and a model at the same time.")
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
    # Initalize dask cluster
    # cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    cluster = LocalCluster(n_workers=int(args.n_jobs), threads_per_worker=1, memory_limit='16GB')
    client = cluster.get_client()
    # Print out the dashboard link
    print(f"Dask Dashboard: {client.dashboard_link}")
    
    def _biased_loss(y_true, y_pred):
        return biased_loss(y_true, y_pred, multiplier=args.loss_bias_multiplier)

    if not args.modified_cosine:
        n_most_recent = args.n_most_recent
        # List available models
        print("Available models:")
        if args.n_most_recent is None:
            available_models = [Path(args.model_path)]
            assert os.path.isfile(available_models[0]), f"Model path {available_models[0]} is not a file."
            n_most_recent = 1
        else:
            available_models = [model for model in Path(args.save_dir).rglob("*.hdf5")][:n_most_recent]
        print(available_models)
        
        print(f"The most recent {n_most_recent} models will be evaluated.")
        
        # Sort models by datetime:  dd_mm_yyyy_hh_mm_ss format "%d_%m_%Y_%H_%M_%S"
        available_models = sorted(available_models, key=lambda x: datetime.strptime(x.stem.split('model_')[1], "%d_%m_%Y_%H_%M_%S"), reverse=True)
    else:
        available_models = [Path('modified_cosine')]
        n_most_recent = 1
    
    print("Testing the following models:", available_models[:n_most_recent], flush=True)
    for model_index, model_name in enumerate(available_models[:n_most_recent]):
        if model_name.stem != 'modified_cosine':
            print(f"Loading model ({model_index+1}/{min(len(available_models), n_most_recent)}: {model_name.stem})...", flush=True)
            model = load_model(model_name, custom_objects={'_biased_loss': _biased_loss})
            print("\tDone.", flush=True)

        spectra_test = load_spectrums(args.test_path)

        if args.metadata_path is not None:
            spectra_test = inject_metadata(args.metadata_path, spectra_test)

        print("Loading Presampled Pairs", flush=True)
        presampled_pairs_path = Path(args.presampled_pairs_path)
        presampled_pairs = dd.read_parquet(presampled_pairs_path).repartition(partition_size="100MB")
        print("\tDone.", flush=True)
                
        metric_dir = os.path.join(args.save_dir, model_name.stem, presampled_pairs_path.stem, args.save_dir_insert)
            
        print("Saving Metrics to:", metric_dir)
        if not os.path.isdir(metric_dir):
            os.makedirs(metric_dir, exist_ok=True)
        print("\tDone.", flush=True)
               
        all_test_inchikeys = [s.get("inchikey")[:14] for s in spectra_test]
        test_inchikeys = np.unique(all_test_inchikeys)
        print(f"Got {len(test_inchikeys)} inchikeys in the test set.")

        dask_output_path = os.path.join(metric_dir, "dask_output.parquet")

        if not os.path.exists(dask_output_path):
            if not args.modified_cosine:
                print("Performing Inference...", flush=True)
                similarity_score = MS2DeepScore(model,)

                # Use MS2DeepScore to embed the vectors
                if not os.path.exists(os.path.join(metric_dir, "embeddings.pkl")):
                    print("Embeddings not found, creating embeddings...", flush=True)
                    embedded_spectra = similarity_score.calculate_vectors(spectra_test)
                    # Create a dictionary of spectrum_ids to embeddings
                    spectrumid_to_embedding_dict = {s.get("title"): e for s, e in zip(spectra_test, embedded_spectra)}
                    pickle.dump(spectrumid_to_embedding_dict, open(os.path.join(metric_dir, "embeddings.pkl"), "wb"))
                else:
                    print("Loading embeddings...", flush=True)
                    spectrumid_to_embedding_dict = pickle.load(open(os.path.join(metric_dir, "embeddings.pkl"), "rb"))       

                # Compute parallel scores for all pairs
                meta = _dask_cosine(presampled_pairs.head(2),
                                    embedding_dict=spectrumid_to_embedding_dict)
                presampled_pairs['predicted_similarity'] = presampled_pairs.map_partitions(_dask_cosine,
                                                                    embedding_dict=spectrumid_to_embedding_dict,
                                                                    meta=meta   # Must provide metadata to avoid fake inputs used for type prediction
                                                                    )
            else:
                # Use Modified Cosine to calculate the similarity
                spectrum_dict = {s.get("title"): s for s in spectra_test}
                meta = _dask_modified_cosine(presampled_pairs.head(2),
                                    spectrum_dict=spectrum_dict)

                presampled_pairs = presampled_pairs.map_partitions(_dask_modified_cosine,
                                                                    spectrum_dict=spectrum_dict,
                                                                    meta=meta
                                                                    )

            # Replace -1 with nan for predictions
            presampled_pairs.predicted_similarity = presampled_pairs.predicted_similarity.replace(-1, np.nan)
        
            # Add the ground_true value to the predictions
            print("Calculating Error")
            presampled_pairs['error'] = (presampled_pairs['predicted_similarity'] - presampled_pairs['ground_truth_similarity']).abs()

            # Save the dataframe to disk
            print("Saving Dataframe to Disk...", flush=True)

            presampled_pairs.to_parquet(dask_output_path)
            print("\tDone.", flush=True)


if __name__ == "__main__":
    main()