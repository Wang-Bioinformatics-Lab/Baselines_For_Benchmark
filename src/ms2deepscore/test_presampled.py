from pathlib import Path
from glob import glob
import os
import gc
import sys
import argparse
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

from matchms.importing import load_from_mgf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

from ms2deepscore import SpectrumBinner
from ms2deepscore.data_generators import DataGeneratorAllInchikeys
from ms2deepscore.models import SiameseModel
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

from ms2deepscore.vector_operations import cosine_similarity

from tqdm import tqdm
from joblib import Parallel, delayed
import tempfile

import dask.dataframe as dd
import dask.array as da
from dask.distributed import LocalCluster
from time import time

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
    cluster = LocalCluster(n_workers=4, threads_per_worker=2)
    client = cluster.get_client()
    

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
        model = load_model(model_name)
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
            presampled_pairs.repartition(partition_size='100MB').to_parquet(dask_output_path)
            print("\tDone.", flush=True)
        # Reload with the error column already computed
        presampled_pairs = dd.read_parquet(dask_output_path)

        # Persist the dataframe into memory, can be very memory intensive
        # print("Persisting Dataframe...", flush=True)
        # presampled_pairs = presampled_pairs.persist()

        print(presampled_pairs.head(10))
        
        ref_score_bins = np.linspace(0.2,1.0, 17)
        ref_score_bins = np.linspace(0.2,1.0, 17)   # TODO: CHANGE THIS?
        
        overall_rmse = da.sqrt(da.mean(da.square(presampled_pairs['error']))).compute()
        print("Overall RMSE (from evaluate()):", overall_rmse)
        overall_mae =  da.mean(da.abs(presampled_pairs['error'])).compute()
        print("Overall MAE (from evaluate()):", overall_mae)
        PBAR.unregister()

        # Nan Safe
        print("Overall NAN RMSE (from evaluate()):", da.sqrt(da.nanmean(da.square(presampled_pairs['error'].values))).compute())
        print("Overall NAN MAE (from evaluate()):", da.nanmean(da.abs(presampled_pairs['error'])).compute())
        print("Nan Count:", presampled_pairs['error'].isna().sum().compute())

        if False:   # Not currently implemented for dask
            train_test_grid = train_test_similarity_heatmap(predictions, train_test_similarities, ref_score_bins)
            # Returns {'bin_content':bin_content_grid, 'bounds':bound_grid, 'rmses':rmse_grid, 'maes':mae_grid}
            
            grid_fig_size = (10,10)
            plt.figure(figsize=grid_fig_size)
            plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
            plt.imshow(train_test_grid['rmses'], vmin=0)
            plt.colorbar()
            
            tick_labels = [f'({x[1][0]:.2f}, {x[1][1]:.2f})' for x in train_test_grid['bounds'][0]]
            
            plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
            plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
            plt.xlabel("Max Test-Train Stuctural Similarity")
            plt.ylabel("Max Test-Train Stuctural Similarity")
            plt.savefig(os.path.join(metric_dir, 'heatmap.png'))
            # Train-Test Similarity Dependent Counts
            plt.figure(figsize=grid_fig_size)
            plt.title('Train-Test Dependent Counts')
            plt.imshow(train_test_grid['bin_content'], vmin=0)
            plt.colorbar()
        
            plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
            plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
            plt.xlabel("Max Test-Train Stuctural Similarity")
            plt.ylabel("Max Test-Train Stuctural Similarity")
            # If the number of samples in a bin is less than 30, put text in the bin
            for i in range(len(train_test_grid['rmses'])):
                for j in range(len(train_test_grid['rmses'])):
                    if train_test_grid['bin_content'][i,j] < 30 and not pd.isna(train_test_grid['bin_content'][i,j]):
                        plt.text(j, i, f'{train_test_grid["bin_content"][i,j]:.0f}', ha="center", va="center", color="white")

            plt.savefig(os.path.join(metric_dir, 'heatmap_counts.png'))
            # Train-Test Similarity Dependent Nan-Counts
            plt.figure(figsize=grid_fig_size)
            plt.title('Train-Test Dependent Nan-Counts')
            plt.imshow(train_test_grid['nan_count'], vmin=0)
            plt.colorbar()
            plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
            plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
            plt.savefig(os.path.join(metric_dir, 'heatmap_nan_counts.png'))
        
            
        # Train-Test Similarity Dependent Losses Aggregated (Max)
        print("Creating Train-Test Similarity Dependent Losses Aggregated Plot", flush=True)
        similarity_dependent_metrics_max = train_test_similarity_dependent_losses(presampled_pairs, ref_score_bins, mode='max')
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(len(similarity_dependent_metrics_max["rmses"]),), similarity_dependent_metrics_max["rmses"],)
        # Add labels on top of bars
        for i, v in enumerate(similarity_dependent_metrics_max["rmses"]):
            plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
        plt.xlabel("Max Test-Train Stuctural Similarity")
        plt.ylabel("RMSE")
        plt.xticks(np.arange(len(similarity_dependent_metrics_max["rmses"])), [f"{a:.2f} to < {b:.2f}" for (a, b) in similarity_dependent_metrics_max["bounds"]], rotation='vertical')
        plt.grid(True)
        plt.savefig(os.path.join(metric_dir, 'train_test_rmse_max.png'))
        train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_max.pkl")
        pickle.dump(similarity_dependent_metrics_max, open(train_test_metric_path, "wb"))
        del similarity_dependent_metrics_max
        gc.collect()

        # Train-Test Similarity Dependent Losses Aggregated (Mean)
        print("Creating Train-Test Similarity Dependent Losses Aggregated Plot", flush=True)
        similarity_dependent_metrics_mean = train_test_similarity_dependent_losses(presampled_pairs, ref_score_bins, mode='mean')
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(len(similarity_dependent_metrics_mean["rmses"]),), similarity_dependent_metrics_mean["rmses"],)
        # Add labels on top of bars
        for i, v in enumerate(similarity_dependent_metrics_mean["rmses"]):
            plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
        plt.xlabel("Mean(Max(Test-Train Stuctural Similarity))")
        plt.ylabel("RMSE")
        plt.xticks(np.arange(len(similarity_dependent_metrics_mean["rmses"])), [f"{a:.2f} to < {b:.2f}" for (a, b) in similarity_dependent_metrics_mean["bounds"]], rotation='vertical')
        plt.grid(True)
        plt.savefig(os.path.join(metric_dir, 'train_test_rmse_mean.png'))
        train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_mean.pkl")
        pickle.dump(similarity_dependent_metrics_mean, open(train_test_metric_path, "wb"))
        del similarity_dependent_metrics_mean
        gc.collect()

        # Train-Test Similarity Dependent Losses Aggregated (ASMS - Left structure based aggregation)
        print("Creating Train-Test Similarity Dependent Losses Aggregated Plot", flush=True)
        similarity_dependent_metrics_asms = train_test_similarity_dependent_losses(presampled_pairs, ref_score_bins, mode='asms')
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(len(similarity_dependent_metrics_asms["rmses"]),), similarity_dependent_metrics_asms["rmses"],)
        # Add labels on top of bars
        for i, v in enumerate(similarity_dependent_metrics_asms["rmses"]):
            plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
        plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
        plt.xlabel("Mean(Max(Test-Train Stuctural Similarity))")
        plt.ylabel("RMSE")
        plt.xticks(np.arange(len(similarity_dependent_metrics_asms["rmses"])), [f"{a:.2f} to < {b:.2f}" for (a, b) in similarity_dependent_metrics_asms["bounds"]], rotation='vertical')
        plt.grid(True)
        plt.savefig(os.path.join(metric_dir, 'train_test_rmse_asms.png'))
        train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_asms.pkl")
        pickle.dump(similarity_dependent_metrics_asms, open(train_test_metric_path, "wb"))
        del similarity_dependent_metrics_asms
        
       
        # MS2DeepScore Tanimoto Dependent Losses Plot

        print("Computing Tanimoto Dependent Losses...", flush=True)
        tanimoto_dependent_dict = tanimoto_dependent_losses(presampled_pairs, np.linspace(0,1.0, 11))

        metric_dict = {}
        metric_dict["bin_content"]      = tanimoto_dependent_dict["bin_content"]
        metric_dict["nan_bin_content"]  = tanimoto_dependent_dict["nan_bin_content"]
        metric_dict["bounds"]           = tanimoto_dependent_dict["bounds"]
        metric_dict["rmses"]            = tanimoto_dependent_dict["rmses"]
        metric_dict["maes"]             = tanimoto_dependent_dict["maes"]
        metric_dict["rmse"] = overall_rmse
        metric_dict["mae"]  = overall_mae
        

        metric_path = os.path.join(metric_dir, "metrics.pkl")
        pickle.dump(metric_dict, open(metric_path, "wb"))
        # Fixed-Tanimoto, Train-Test Dependent Plot
        if False:
            ref_score_bins = np.linspace(0,1.0, 11)
            ftttsd = fixed_tanimoto_train_test_similarity_dependent(predictions, train_test_similarities, ref_score_bins)
            
            # Save to pickle
            metric_path = os.path.join(metric_dir, "metrics.pkl")
            print(f"Saving metrics to {metric_path}")
            fixed_tanimoto_train_test_similarity_dependent_path = os.path.join(metric_dir, "fixed_tanimoto_train_test_similarity_dependent_path.pkl")
            pickle.dump(ftttsd, open(fixed_tanimoto_train_test_similarity_dependent_path, "wb"))
            pickle.dump(metric_dict, open(metric_path, "wb"))
            train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_mean.pkl")
            pickle.dump(similarity_dependent_metrics_mean, open(train_test_metric_path, "wb"))
            train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_max.pkl")
            pickle.dump(similarity_dependent_metrics_max, open(train_test_metric_path, "wb"))
            train_test_metric_path = os.path.join(metric_dir, "train_test_grid.pkl")
            pickle.dump(train_test_grid, open(train_test_metric_path, "wb"))

        print("Creating Tanimoto Dependent Losses Plot", flush=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))
        
        ax1.plot(np.arange(len(metric_dict["rmses"])), metric_dict["rmses"], "o:", color="crimson")
        ax1.set_title('RMSE')
        ax1.set_ylabel("RMSE")
        ax1.grid(True)

        ax2.plot(np.arange(len(metric_dict["rmses"])), metric_dict["bin_content"], "o:", color="teal")
        ax2.plot(np.arange(len(metric_dict["rmses"])), metric_dict["nan_bin_content"], "o:", color="grey")
        if sum(metric_dict["nan_bin_content"]) > 0:
            ax2.legend(["# of valid spectrum pairs", "# of nan spectrum pairs"])
        ax2.set_title('# of spectrum pairs')
        ax2.set_ylabel("# of spectrum pairs")
        ax2.set_xlabel("Tanimoto score bin")
        plt.yscale('log')
        plt.xticks(np.arange(len(metric_dict["rmses"])), [f"{a:.1f} to < {b:.1f}" for (a, b) in metric_dict["bounds"]], rotation='vertical')
        ax2.grid(True)
        
        # Save figure
        fig_path = os.path.join(metric_dir, "metrics.png")
        plt.savefig(fig_path)
        
        # spec2vec_percentile_plot(predictions, scores_ref, metric_dir)

        # Pairwise Similarity, Train-Test Distance Heatmap
        train_test_similarity_bins = np.linspace(0.2, 1, 17)    # to match other figures
        pairwise_similarity_bins   = np.linspace(0.0, 1, 21)    # to match other figures
        print("Creating Pairwise & Train-Test Similarity Heatmap", flush=True)
        pw_tt_metrics = pairwise_train_test_dependent_heatmap(presampled_pairs, pairwise_similarity_bins, train_test_similarity_bins)
        # print('pw_tt_metrics', pw_tt_metrics)
        pw_tt_metric_path = os.path.join(metric_dir, "pairwise_train_test_metrics.pkl")
        pickle.dump(pw_tt_metrics, open(pw_tt_metric_path, "wb"))
        # Set any -1 values to nan
        pw_tt_metrics['rmse_grid'] = np.where(pw_tt_metrics['rmse_grid'] == -1, np.nan, pw_tt_metrics['rmse_grid'])
        plt.figure(figsize=grid_fig_size)
        # pw_tt_metrics['rmse_grid'] first index modulates the pairwise similarity, second index modulates the train-test similarity
        # imshow is transposed so that the pairwise similarity is on the y-axis
        plt.imshow(pw_tt_metrics['rmse_grid'], origin='lower')
        plt.colorbar()
        plt.title('Pairwise & Train-Test Dependent RMSE')
        plt.ylabel('Pairwise Structural Similarity')
        plt.xlabel('Max Test-Train Structural Similarity')
        bounds = np.array(pw_tt_metrics['bounds'])
        print('bounds.shape', bounds.shape)
        x_ticks = bounds[0,:,1,:]
        plt.xticks(range(x_ticks.shape[0]), [f"({x_ticks[i][0].item():.2f}-{x_ticks[i][1].item():.2f})" for i in range(x_ticks.shape[0])], rotation=90)
        plt.xlim(0, bounds.shape[1])
        y_ticks = bounds[:,0,0,:]
        plt.yticks(range(y_ticks.shape[0]), [f"({y_ticks[i][1].item():.2f}-{y_ticks[i][1].item():.2f})" for i in range(y_ticks.shape[0])])
        plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap.png'), bbox_inches="tight")

        # Pairwise Similarity, Train-Test Distance Heatmap (Counts)
        plt.figure(figsize=grid_fig_size)
        counts = pw_tt_metrics['count']
        # Transform counts by log base 10 +1
        counts = np.log10(counts + 1)
        plt.imshow(counts, vmin=0, origin='lower')
        plt.colorbar()
        plt.title('Pairwise & Train-Test Dependent Counts')
        plt.ylabel('Pairwise Structural Similarity')
        plt.xlabel('Max Test-Train Structural Similarity')
        x_ticks = bounds[0,:,1,:]
        plt.xticks(range(x_ticks.shape[0]), [f"({x_ticks[i][0].item():.2f}-{x_ticks[i][1].item():.2f})" for i in range(x_ticks.shape[0])], rotation=90)
        plt.xlim(0, bounds.shape[1])
        y_ticks = bounds[:,0,0,:]
        plt.yticks(range(y_ticks.shape[0]), [f"({y_ticks[i][0].item():.2f}-{y_ticks[i][1].item():.2f})" for i in range(y_ticks.shape[0])])
        plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap_counts.png'), bbox_inches="tight")
        
        # Train-Test Similarity Bar Plot
        # Pretty sure this is the same as 'Train-Test Dependent RMSE' plot
        if False:
            train_test_sim = train_test_similarity_bar_plot(presampled_pairs, ref_score_bins)
            bin_content, bounds = train_test_sim['bin_content'], train_test_sim['bounds']
            plt.figure(figsize=(12, 9))
            plt.title("Number of Structures in Similarity Bins (Max Similarity to Train Set)")
            plt.bar(range(len(bin_content)), bin_content, label='Number of Structures')
            plt.xlabel('Similarity Bin (Max Similarity to Train Set)')
            plt.ylabel('Number of Structures')
            plt.xticks(range(len(bin_content)), [f"({bounds[i][0]:.2f}-{bounds[i][1]:.2f})" for i in range(len(bounds))], rotation=45)
            plt.legend()
            plt.savefig(os.path.join(metric_dir, 'train_test_similarity_bar_plot.png'), bbox_inches="tight")
        
        r2 = np.corrcoef(presampled_pairs['predicted_similarity'], presampled_pairs['ground_truth_similarity'])[0,1]**2
        print(f"R squared: {r2:.2f}", flush=True)

        # Skip the scatter plots and hexbin plots for now, not particularly useful
        return 0
        # Score, Tanimoto Scatter Plot
        print("Creating Scatter Plot")
        plt.figure(figsize=grid_fig_size)
        plt.scatter(presampled_pairs['predicted_similarity'].values.compute(), presampled_pairs['ground_truth_similarity'].values.compute(), alpha=0.2)
        # Show R Squared
        r2 = np.corrcoef(presampled_pairs['predicted_similarity'], presampled_pairs['ground_truth_similarity'])[0,1]**2
        # Plot y=x line
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('Predicted Spectral Similarity Score')
        plt.ylabel('Tanimoto Score')
        # Make square
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'Predicted vs Reference Spectral Similarity Scores \n(R squared = {r2:.2f})')
        plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference.png'))
        
        # Score, Tanimoto Binned Histograms
        # This should show 10 histograms vertically stacked that correspond to 10 different tanimoto similarity bins
        #plt.figure(figsize=(20,20))
        # TODO    
        
        # Score, Tanimoto Scatter Plot (Hexbin)
        print("Creating Hexbin Plot")
        plt.figure(figsize=grid_fig_size)    
        hb = plt.hexbin(presampled_pairs['predicted_similarity'].values.compute(), presampled_pairs['ground_truth_similarity'].values.compute(), gridsize=50, cmap='inferno', bins='log')
        cb = plt.colorbar(hb)
        cb.set_label('log counts')
        plt.xlabel('Predicted Spectral Similarity Score')
        plt.ylabel('Tanimoto Score')
        # Make square
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title('Predicted vs Reference Spectral Similarity Scores')
        plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference_hexbin.png'))

    
if __name__ == "__main__":
    main()