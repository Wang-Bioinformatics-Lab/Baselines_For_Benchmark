from pathlib import Path
import os
import gc
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
import pandas as pd
import numpy as np

from ms2deepscore import MS2DeepScore
from custom_model_loader import load_model  # Allows kwargs
from ms2deepscore.vector_operations import cosine_similarity

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

# def _parallel_cosine(dask_partition,
#                      embedding_dict,
#                      buffer_size=7_900_000): 
#     # Buffer Size Calculation: 
#     # 80 gb (leaving 40 for overhead) / 32 / 254 bytes per row * 0.75 for extra safety


#     hdf_store = None
#     hdf_path = None
#     try:
#         # Tempfile for hdf storage
#         hdf_path = tempfile.NamedTemporaryFile(delete=False)

#         # Calculate the similarity list in a memory-efficent manner
#         hdf_store = pd.HDFStore(hdf_path.name)
        
#         max_main_train_test_sim  = train_test_sim_dict[main_inchikey]['max']
#         min_main_train_test_sim  = train_test_sim_dict[main_inchikey]['min']
#         mean_main_train_test_sim = train_test_sim_dict[main_inchikey]['mean']

#         columns=['inchikey1', 'inchikey2','spectrumid1', 'spectrumid2', 'ground_truth_similarity', 'mean_mean_train_test_sim',
#                 'max_max_train_test_sim', 'max_mean_train_test_sim', 'max_min_train_test_sim', 'pred', 'error'] # Only error and pred are new here


#         output_list = []
#         curr_buffer_size = buffer_size

#         def _helper(spectrumid1, spectrumid2):
#             return cosine_similarity(embedding_dict[spectrumid1]['embedding'], embedding_dict[spectrumid2]['embedding'])
        
#         spectrumid1_lst = dask_partition['spectrum_id1'].values
#         spectrumid2_lst = dask_partition['spectrum_id2'].values

        

#         # Dump the remaining content
#         start_time = time()
#         output_frame = pd.DataFrame(output_list, columns=columns)
#         # print(f"Creating dataframe took {time() - start_time} seconds")
#         start_time = time()
#         hdf_store.put(f"{main_inchikey}", output_frame, format='table', append=True, track_times=False,
#                         min_itemsize={'spectrumid1': 50, 'spectrumid2': 50})
#         # print(f"Storing took {time() - start_time} seconds")
#         curr_buffer_size = buffer_size
#         output_list = []

#         hdf_store.close()
#         hdf_path.close()
#     except Exception as e:
#         # Close the hdf store if needed and exists
#         if hdf_store is not None and hdf_store.is_open:
#             hdf_store.close()

#         # Delete the hdf file if needed
#         if os.path.exists(hdf_path.name):
#             os.remove(hdf_path.name)

#         raise e
#     except KeyboardInterrupt as ki:
#         # Close the hdf store if needed and exists
#         if hdf_store is not None and hdf_store.is_open:
#             hdf_store.close()

#         # Delete the hdf file if needed
#         if os.path.exists(hdf_path.name):
#             os.remove(hdf_path.name)

#         raise ki

#     return hdf_path.name # Cleanup in serial process after concat

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

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')      # Path to pickle file of spectra
    parser.add_argument('--presampled_pairs_path', type=str, help='Path to dask parquet file of presampled pairs')
    parser.add_argument("--save_dir", type=str, help="Path to the model")
    parser.add_argument("--model_path", type=str, help="Path to the model, overrides n_most_recent", default=None)
    parser.add_argument("--save_dir_insert", type=str, help="Appended to save dir, to help organize test sets", default="")
    parser.add_argument("--n_most_recent", type=int, help="Number of most recent models to evaluate", default=None)
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    args = parser.parse_args()
    
    grid_fig_size = (10,10)
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
    # Initalize dask cluster
    # cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    cluster = LocalCluster(n_workers=int(args.n_jobs/2), threads_per_worker=2)
    client = cluster.get_client()
    # Print out the dashboard link
    print(f"Dask Dashboard: {client.dashboard_link}")
    
    def _biased_loss(y_true, y_pred):
        return biased_loss(y_true, y_pred, multiplier=args.loss_bias_multiplier)

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
    
    
    print("Testing the following models:", available_models[:n_most_recent], flush=True)
    for model_index, model_name in enumerate(available_models[:n_most_recent]):
        print(f"Loading model ({model_index+1}/{min(len(available_models), n_most_recent)}: {model_name.stem})...", flush=True)
        model = load_model(model_name, custom_objects={'_biased_loss': _biased_loss})
        print("\tDone.", flush=True)

        spectra_test = pickle.load(open(args.test_path, "rb"))

        print("Loading Presampled Pairs", flush=True)
        presampled_pairs_path = Path(args.presampled_pairs_path)
        presampled_pairs = dd.read_parquet(presampled_pairs_path)
        print("\tDone.", flush=True)
                
        metric_dir = os.path.join(args.save_dir, model_name.stem, presampled_pairs_path.stem, args.save_dir_insert)
            
        print("Saving Metrics to:", metric_dir)
        if not os.path.isdir(metric_dir):
            os.makedirs(metric_dir, exist_ok=True)
        print("\tDone.", flush=True)
               
        all_test_inchikeys = [s.get("inchikey")[:14] for s in spectra_test]
        test_inchikeys = np.unique(all_test_inchikeys)
        print(f"Got {len(test_inchikeys)} inchikeys in the test set.")
        
        print("Performing Inference...", flush=True)
        similarity_score = MS2DeepScore(model,)
        df_labels = [s.get("spectrum_id") for s in spectra_test]

        dask_output_path = os.path.join(metric_dir, "dask_output.parquet")

        if not os.path.exists(dask_output_path):
            # Use MS2DeepScore to embed the vectors
            if not os.path.exists(os.path.join(metric_dir, "embeddings.pkl")):
                print("Embeddings not found, creating embeddings...", flush=True)
                embedded_spectra = similarity_score.calculate_vectors(spectra_test)
                # embedded_spectra = similarity_score.calculate_vectors(spectra_test[0:500])        # DEBUG USING ONE EMBEDDING
                # embedded_spectra = np.tile(embedded_spectra, (int(len(spectra_test)/499), 1))
                # Create a dictionary of spectrum_ids to embeddings
                spectrumid_to_embedding_dict = {s.get("title"): e for s, e in zip(spectra_test, embedded_spectra)}
                pickle.dump(spectrumid_to_embedding_dict, open(os.path.join(metric_dir, "embeddings.pkl"), "wb"))
            else:
                print("Loading embeddings...", flush=True)
                spectrumid_to_embedding_dict = pickle.load(open(os.path.join(metric_dir, "embeddings.pkl"), "rb"))       
            
            # Get first two partitons #DEBUG
            # presampled_pairs = presampled_pairs.partitions[:3]

            # Compute parallel scores for all pairs
            meta = _dask_cosine(presampled_pairs.head(2),
                                embedding_dict=spectrumid_to_embedding_dict)
            presampled_pairs['predicted_similarity'] = presampled_pairs.map_partitions(_dask_cosine,
                                                                embedding_dict=spectrumid_to_embedding_dict,
                                                                meta=meta   # Must provide metadata to avoid fake inputs used for type prediction
                                                                )

            # Replace -1 with nan for predictions
            presampled_pairs.predicted_similarity = presampled_pairs.predicted_similarity.replace(-1, np.nan)
        
            # Add the ground_true value to the predictions
            print("Calculating Error")
            presampled_pairs['error'] = (presampled_pairs['predicted_similarity'] - presampled_pairs['ground_truth_similarity']).abs()


            # Save the dataframe to disk
            print("Saving Dataframe to Disk...", flush=True)
            with dask.config.set(num_workers=min(os.cpu_count()-4, 1)): # This is very low memory, but very slow, so let's give it a boost
                presampled_pairs.repartition(partition_size='100MB').to_parquet(dask_output_path)
            print("\tDone.", flush=True)


if __name__ == "__main__":
    main()