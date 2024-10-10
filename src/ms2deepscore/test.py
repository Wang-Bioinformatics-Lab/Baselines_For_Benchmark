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
from time import time

sys.path.append('../shared')
from utils import train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot, \
                    fixed_tanimoto_train_test_similarity_dependent
                    
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

def _parallel_cosine(main_inchikey,
                    inchikey_list,
                    tanimoto_df,
                    inchikey_to_embedding_dict,
                    train_test_sim_dict,
                    pairs_generator=None,
                    buffer_size=7_900_000): 
    # Buffer Size Calculation: 
    # 80 gb (leaving 40 for overhead) / 32 / 254 bytes per row * 0.75 for extra safety

    hdf_store = None
    hdf_path = None
    try:
        # Tempfile for hdf storage
        hdf_path = tempfile.NamedTemporaryFile(delete=False)

        # Calculate the similarity list in a memory-efficent manner
        hdf_store = pd.HDFStore(hdf_path.name)
        predictions_i = inchikey_to_embedding_dict[main_inchikey]
        
        max_main_train_test_sim  = train_test_sim_dict[main_inchikey]['max']
        min_main_train_test_sim  = train_test_sim_dict[main_inchikey]['min']
        mean_main_train_test_sim = train_test_sim_dict[main_inchikey]['mean']

        columns=['spectrumid1', 'spectrumid2', 'ground_truth_similarity', 'predicted_similarity', 'mean_mean_train_test_sim',
                'max_max_train_test_sim', 'max_mean_train_test_sim', 'max_min_train_test_sim', 'error']


        output_list = []
        curr_buffer_size = buffer_size
        for inchikey_j in inchikey_list:

            max_j_train_test_sim  = train_test_sim_dict[inchikey_j]['max']
            min_j_train_test_sim  = train_test_sim_dict[inchikey_j]['min']
            mean_j_train_test_sim = train_test_sim_dict[inchikey_j]['mean']
            mean_mean_train_test_sim = np.mean([mean_main_train_test_sim, mean_j_train_test_sim])
            max_max_train_test_sim = max(max_main_train_test_sim, max_j_train_test_sim)
            max_mean_train_test_sim = max(mean_main_train_test_sim, mean_j_train_test_sim)
            max_min_train_test_sim = max(min_main_train_test_sim, min_j_train_test_sim)
            gt = tanimoto_df.loc[main_inchikey, inchikey_j]

            predictions_j = inchikey_to_embedding_dict[inchikey_j]

            valid_pairs = None
            if pairs_generator is not None:
                # Filter by valid pairs only
                # Returns list of spectrum_id1, spectrumid2
                valid_pairs = pairs_generator.get_spectrum_pair_with_inchikey(main_inchikey, inchikey_j,
                                                                              return_type='ids',
                                                                              return_all=True)
                # Structure as dataframe
                valid_pairs = pd.DataFrame(valid_pairs, columns=['spectrum_id_1', 'spectrum_id_2'])
                # Use a set since it's significantly faster than querying the index
                spectra_1_ids = set(valid_pairs['spectrum_id_1'].values)
                valid_pairs = valid_pairs.set_index('spectrum_id_1')

            # For each pair of spectra in i, j, calculate the similarity
            for dict_a in predictions_i:
                if valid_pairs is not None:
                    if dict_a['spectrum_id'] not in spectra_1_ids:
                        continue
                for dict_b in predictions_j:
                    if valid_pairs is not None:
                        if dict_b['spectrum_id'] not in valid_pairs[dict_a['spectrum_id']]['spectrum_id_2'].values:
                            continue
                    # The only time this computation will cross below the main diagonal.
                    # (in terms of spectrum_ids) is when the inchikeys are the same.
                    # When this happens, we only want to compute the similarity once so
                    # these cases are not weighted differently
                    if main_inchikey == inchikey_j:
                        if dict_a['spectrum_id'] < dict_b['spectrum_id']:
                            continue
                    start_time = time()
                    pred = cosine_similarity(dict_a['embedding'], dict_b['embedding'])
                    # print(f"Calculating cosine similarity took {time() - start_time} seconds")
                    start_time = time()
                    output_list.append((dict_a['spectrum_id'],
                                        dict_b['spectrum_id'],
                                        gt,
                                        pred,
                                        mean_mean_train_test_sim,
                                        max_max_train_test_sim,
                                        max_mean_train_test_sim,
                                        max_min_train_test_sim,
                                        np.abs(gt - pred)))
                    # print(f"Appending took {time() - start_time} seconds")

                    curr_buffer_size -= 1
                    if curr_buffer_size == 0:
                        # Store the similarity at /main_inchikey/inchikey_j
                        # Should have spectrumid1, spectrumid2, ground_truth_similarity, predicted_similarity

                        start_time = time()
                        output_frame = pd.DataFrame(output_list, columns=columns)
                        # print(f"Creating dataframe took {time() - start_time} seconds")
                        start_time = time()
                        hdf_store.put(f"{main_inchikey}", output_frame, format='table', append=True, track_times=False,
                                        min_itemsize={'spectrumid1': 50, 'spectrumid2': 50})
                        # print(f"Storing took {time() - start_time} seconds")
                        curr_buffer_size = buffer_size
                        output_list = []

        # Dump the remaining content
        start_time = time()
        output_frame = pd.DataFrame(output_list, columns=columns)
        # print(f"Creating dataframe took {time() - start_time} seconds")
        start_time = time()
        hdf_store.put(f"{main_inchikey}", output_frame, format='table', append=True, track_times=False,
                        min_itemsize={'spectrumid1': 50, 'spectrumid2': 50})
        # print(f"Storing took {time() - start_time} seconds")
        curr_buffer_size = buffer_size
        output_list = []

        hdf_store.close()
        hdf_path.close()
    except Exception as e:
        # Close the hdf store if needed and exists
        if hdf_store is not None and hdf_store.is_open:
            hdf_store.close()

        # Delete the hdf file if needed
        if os.path.exists(hdf_path.name):
            os.remove(hdf_path.name)

        raise e
    except KeyboardInterrupt as ki:
        # Close the hdf store if needed and exists
        if hdf_store is not None and hdf_store.is_open:
            hdf_store.close()

        # Delete the hdf file if needed
        if os.path.exists(hdf_path.name):
            os.remove(hdf_path.name)

        raise ki

    return hdf_path.name # Cleanup in serial process after concat

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')      # Path to pickle file of spectra
    parser.add_argument("--tanimoto_path", type=str, help="Path to the tanimoto scores")        # Path to a csv file with columns, indexs of inchikey[:14]
    parser.add_argument("--train_test_similarities", type=str, help="Path to the train-test tanimoto scores")
    parser.add_argument("--save_dir", type=str, help="Path to the model")
    parser.add_argument("--model_path", type=str, help="Path to the model, overrides n_most_recent", default=None)
    parser.add_argument("--save_dir_insert", type=str, help="Appended to save dir, to help organize test sets", default="")
    parser.add_argument("--n_most_recent", type=int, help="Number of most recent models to evaluate", default=None)
    parser.add_argument("--test_pairs_path", type=str, default=None, help="Path to the test pairs file")    # Path to filtered train pairs
    parser.add_argument("--train_pairs_path", type=str, default=None, help="Path to the train pairs file")  # Path to filtered test pairs
    parser.add_argument("--subsample", type=int, default=None, help="Caps the number of spectra associated with each structure.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    args = parser.parse_args()
    
    # # Check if it's a file
    # if not os.path.isfile(args.test_path):
    #     if not os.path.isdir(args.save_dir):
    #         os.makedirs(args.save_dir, exist_ok=True)
    # else:
    #     # Get without extention    
    #     if not os.path.isdir(os.path.join(Path(args.save_dir).parent, Path(args.save_dir).stem)):
    #         os.makedirs(os.path.join(Path(args.save_dir).parent, Path(args.save_dir).stem), exist_ok=True)
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
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
                
        if args.test_pairs_path is not None:
            print(f"Filtering spectra based on pairs in {args.test_pairs_path}")
            print(f"Began with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
            pairs_df = pd.read_feather(args.test_pairs_path)
            valid_spectra_ids = np.unique(pairs_df[['spectrum_id_1', 'spectrum_id_2']].values)
            spectra_test = [s for s in spectra_test if s.get('spectrum_id') in valid_spectra_ids]
            print(f"Ended with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
        
        if args.subsample is not None:
            print(f"Subsampling {args.subsample} spectra per inchikey")
            print(f"Began with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
            test_dict = {s.get("inchikey")[:14]: [] for s in spectra_test}
            for s in spectra_test:
                test_dict[s.get("inchikey")[:14]].append(s)
            # sort all spectra by spectrum_id for reproducibility
            for inchikey, spectra in test_dict.items():
                test_dict[inchikey] = sorted(spectra, key=lambda x: x.get("spectrum_id"))
            spectra_test = []
            for inchikey, spectra in test_dict.items():
                spectra_test.extend(spectra[:args.subsample])
            print(f"Ended with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")

        tanimoto_df = pd.read_csv(args.tanimoto_path, index_col=0)
        train_test_similarities = pd.read_csv(args.train_test_similarities, index_col=0)
        if args.test_pairs_path is not None:
            metric_dir = os.path.join(args.save_dir, model_name.stem, 'metrics_filtered_pairs', args.save_dir_insert)
        else:
            if args.save_dir_insert != "":
                raise ValueError("--save_dir_insert should only be provieded for filtered pairs")
            metric_dir = os.path.join(args.save_dir, model_name.stem, 'metrics')
            
        print("Saving Metrics to:", metric_dir)
        if not os.path.isdir(metric_dir):
            os.makedirs(metric_dir, exist_ok=True)
        print(tanimoto_df)
        print("\tDone.", flush=True)
               
        all_test_inchikeys = [s.get("inchikey")[:14] for s in spectra_test]
        test_inchikeys = np.unique(all_test_inchikeys)
        print(f"Got {len(test_inchikeys)} inchikeys in the test set.")
        
        print("Performing Inference...", flush=True)
        similarity_score = MS2DeepScore(model,)
        df_labels = [s.get("spectrum_id") for s in spectra_test]

        # Use MS2DeepScore to embed the vectors
        # embedded_spectra = similarity_score.calculate_vectors(spectra_test)
        # DEBUG
        embedded_spectra = similarity_score.calculate_vectors([spectra_test[0]])
        embedded_spectra = [embedded_spectra[0] for _ in range(len(spectra_test))]

        # Group embeddings by inchikey
        inchikey_to_embedding_dict = {}
        for i, embedding in enumerate(embedded_spectra):
            inchikey = all_test_inchikeys[i]
            curr_list = inchikey_to_embedding_dict.get(inchikey, [])
            curr_list.append({'spectrum_id': spectra_test[i].get("spectrum_id"),
                               'embedding':embedding,})
            inchikey_to_embedding_dict[inchikey] = curr_list

        # Precompute, mean, max, min train-test similarities for each inchikey
        train_test_sim_dict = {}
        print("Precomputing train-test similarities...", flush=True)
        for inchikey in tqdm(test_inchikeys):
            main_train_test_sim = train_test_similarities.loc[:, inchikey]
            train_test_sim_dict[inchikey] = {'max': main_train_test_sim.max(),
                                            'min': main_train_test_sim.min(),
                                            'mean': main_train_test_sim.mean()}

        # Compute parallel scores using _parallel_cosine
        all_inchikeys = list(inchikey_to_embedding_dict.keys())
        output_hdf_files = []
        # output_hdf_path  = os.path.join(metric_dir, "predictions.hdf5")
        output_hdf_path  = os.path.join(metric_dir, "predictions.parquet")
        hdf_store = None
        temp_store = None
        try:
            print("Calculating pairwise cosine similarities...", flush=True)
            output_hdf_files = Parallel(n_jobs=args.n_jobs)(delayed(_parallel_cosine)(main_inchikey,
                                                                            all_inchikeys[i:],
                                                                            tanimoto_df,
                                                                            inchikey_to_embedding_dict,
                                                                            train_test_sim_dict)
                                                                            for i, main_inchikey in enumerate(tqdm(all_inchikeys)))
            # Concatenate the hdf files
            # print("Concatenating the hdf files...", flush=True)
            # hdf_store = pd.HDFStore(output_hdf_path)
            # for hdf_file in tqdm(output_hdf_files):
            #     temp_store = pd.HDFStore(hdf_file)
            #     for key in temp_store.keys():
            #         hdf_store.put(key, temp_store.get(key), format='table')
            #     temp_store.close()

            from time import time
            start_time = time()
            dask_df = dd.read_hdf(output_hdf_files, key='/*')
            print(f"Reading the hdf file lazily took {time() - start_time:.2f} seconds")
            start_time = time()
            # Will concatenate to single table, but use parallelism
            # dask_df.to_hdf(output_hdf_path, key='/data-*', mode='w', scheduler='processes', compute=True)
            dask_df.to_parquet(output_hdf_path, overwrite=True, compute=True)
            print(f"Writing the hdf file took {time() - start_time:.2f} seconds")
        finally:
            if hdf_store is not None and hdf_store.is_open:
                hdf_store.close()
            
            if temp_store is not None and temp_store.is_open:
                temp_store.close()

            for hdf_file in output_hdf_files:
                if os.path.exists(hdf_file):
                    os.remove(hdf_file)





        # # Calculate the similarity list in a memory-efficent manner
        # hdf_path  = os.path.join(metric_dir, "predictions.hdf5")
        # hdf_store = pd.HDFStore(hdf_path)
        # try:
        #     all_inchikeys = list(inchikey_to_embedding_dict.keys())
        #     for i, inchikey_i in enumerate(tqdm(all_inchikeys)):
        #         predictions_i = inchikey_to_embedding_dict[inchikey_i]
        #         for inchikey_j in all_inchikeys[i:]:
        #             predictions_j = inchikey_to_embedding_dict[inchikey_j]

        #             # TODO: Check if pair is valid

        #             # TODO: Add train-test similarity to the output

        #             # TODO: Profile to find why this is so slow
        #             gt = tanimoto_df.loc[inchikey_i, inchikey_j]

        #             output_list = []
        #             # For each pair of spectra in i, j, calculate the similarity
        #             for dict_a in predictions_i:
        #                 for dict_b in predictions_j:
        #                     # The only time this computation will cross below the main diagonal.
        #                     # (in terms of spectrum_ids) is when the inchikeys are the same.
        #                     # When this happens, we only want to compute the similarity once so 
        #                     # these cases are not weighted differently
        #                     if inchikey_i == inchikey_j:
        #                         if dict_a['spectrum_id'] < dict_b['spectrum_id']:
        #                             continue
        #                     pred = cosine_similarity(dict_a['embedding'], dict_b['embedding'])
        #                     output_list.append({'spectrumid1': dict_a['spectrum_id'],
        #                                         'spectrumid2': dict_b['spectrum_id'],
        #                                         'ground_truth_similarity': gt,
        #                                         'predicted_similarity': pred})
                    
        #             # Store the similarity at /inchikey_i/inchikey_j
        #             # Should have spectrumid1, spectrumid2, ground_truth_similarity, predicted_similarity
        #             output_frame = pd.DataFrame(output_list)
        #             hdf_store.put(f"{inchikey_i}/{inchikey_j}", output_frame, format='table')

        # finally:
        #     hdf_store.close()

        # DEBUG
        sys.exit(0)

        predictions = similarity_score.matrix(spectra_test, spectra_test, is_symmetric=True)
        # Convert to dataframe
        predictions = pd.DataFrame(predictions, index=df_labels, columns=df_labels)
        print("\tDone.", flush=True)
        
        print("Evaluating...", flush=True)
        inchikey_idx_test = np.zeros(len(spectra_test))
        ordered_prediction_index = np.zeros(len(spectra_test))
        
        for i, spec in enumerate(tqdm(spectra_test)):
            try:
                inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0].item()
            except ValueError as value_error:
                if not spec.get("inchikey")[:14] in tanimoto_df.index.values:
                    raise ValueError (f"InChI Key '{spec.get('inchikey')[:14]}' is not found in the provided Stuctural similarity matrix.")
                raise value_error
            try:
                ordered_prediction_index[i] = np.where(predictions.index.values == spec.get("spectrum_id"))[0].item()
            except ValueError as value_error:
                if not spec.get("spectrum_id") in predictions.index.values:
                    raise ValueError (f"spectrum_id' {spec.get('spectrum_id')}' is not found in the provided predictions.")
                raise value_error

        # Reorder both preds and ground truth based on test spectra order

        print("Shape of predictions:", predictions.shape)
        predictions = predictions.iloc[ordered_prediction_index, ordered_prediction_index]
        if args.test_pairs_path is not None:
            print("Shape of predictions after filtering:", predictions.shape)


        # inchikey_idx_test = inchikey_idx_test.astype("int")
        
        assert len(predictions.index.values) == len(predictions.index.unique())
        assert all(predictions.index.values == predictions.columns.values)

        # Flatten the prediction and reference matrices
        # columns should be ['spectrum_id_1', 'spectrum_id_2', 'score']
        predictions = predictions.stack().reset_index()
        predictions.columns = ['spectrum_id_1', 'spectrum_id_2', 'score']
        # set string columns to categorical to reduce repeat
        predictions['spectrum_id_1'] = pd.Categorical(predictions['spectrum_id_1'])
        predictions['spectrum_id_2'] = pd.Categorical(predictions['spectrum_id_2'])
        
        # Only get valid pairs
        if args.test_pairs_path is not None:
            pairs_df = pd.read_feather(args.test_pairs_path)

            # Create sets of the pairs
            pair_set = set(map(tuple, pairs_df[['spectrum_id_1', 'spectrum_id_2']].values))
            reverse_pair_set = set(map(tuple, pairs_df[['spectrum_id_2', 'spectrum_id_1']].values))

            # Check if pairs_df is symmetric
            if pair_set != reverse_pair_set:
                print("Pairs file is not symmetric. Making it symmetric.")
                pairs_df = pd.concat([pairs_df, pairs_df.rename(columns={'spectrum_id_1':'spectrum_id_2',
                                                                        'spectrum_id_2':'spectrum_id_1',
                                                                        'inchikey_1':'inchikey_2',
                                                                        'inchikey_2':'inchikey_1'})], axis=0)
                print("Pairs are now symmetric.")
                       
            pairs_df['pair'] = list(zip(pairs_df.spectrum_id_1, pairs_df.spectrum_id_2))
            predictions['pair'] = list(zip(predictions.spectrum_id_1, predictions.spectrum_id_2))
            
            print("Filtering Predictions based on pairs")
            valid_pairs = set(predictions['pair'])
            predictions = predictions.loc[predictions['pair'].isin(valid_pairs)]
            
            print("Joining Predictions and pairs")
            predictions = predictions.merge(pairs_df, on=['spectrum_id_1','spectrum_id_2','pair'])
            
            # Rename 'structural_similarity' -> 'tanimoto'
            predictions = predictions.rename(columns={'structural_similarity':'tanimoto'})
        else:
            # We'll have to get the actual tanimoto df 
            print("Shape of tanimoto_df:", tanimoto_df.shape)
            # inchikey_idx_test = inchikey_idx_test.astype("int")
            tanimoto_df = tanimoto_df.iloc[inchikey_idx_test, inchikey_idx_test]
                
            # Flatten the tanimoto matrix
            print("Reshape the tanimoto matrix")
            tanimoto_df = tanimoto_df.stack().reset_index()
            tanimoto_df.columns = ['inchikey_1', 'inchikey_2', 'tanimoto']
            tanimoto_df['inchi_pair'] = list(zip(tanimoto_df.inchikey_1, tanimoto_df.inchikey_2))
            tanimoto_mapping = dict(zip(tanimoto_df.inchi_pair, tanimoto_df.tanimoto))
            del tanimoto_df
            gc.collect()
                    
            # Add Inchis to the predictions
            print("Adding InChis to the predictions")
            structure_mapping = {spectrum.get("spectrum_id"): spectrum.get("inchikey")[:14] for spectrum in spectra_test}
            predictions['inchikey_1'] = predictions['spectrum_id_1'].map(structure_mapping)
            # Convert to categorical
            predictions['inchikey_1'] = pd.Categorical(predictions['inchikey_1'])
            predictions['inchikey_2'] = predictions['spectrum_id_2'].map(structure_mapping)
            # Convert to categorical
            predictions['inchikey_2'] = pd.Categorical(predictions['inchikey_2'])
            predictions['inchi_pair'] = list(zip(predictions.inchikey_1, predictions.inchikey_2))
            del structure_mapping
            gc.collect()
            
            # For each prediction, get the tanimoto
            print("Adding Tanimoto to the predictions")
            predictions['tanimoto'] = predictions['inchi_pair'].map(tanimoto_mapping)
            del tanimoto_mapping
            # Remove uneeded columns
            predictions.drop(columns=['inchi_pair'], inplace=True)
            gc.collect()
            
        if args.train_pairs_path is not None:
            # We need to remove the spectrum ids from the train-test similarity matrix that are not in train pairs
            print("Filtering Train-Test Similarity Matrix based on train pairs")
            train_inchikeys= np.unique(pd.read_feather(args.train_pairs_path)[['inchikey_1','inchikey_2']].values)
            print(f"Found {len(train_inchikeys)} unique inchikeys in the train pairs")
            print(f"Found {len(train_test_similarities)} unique inchikeys in the train-test similarity matrix")
            train_test_similarities = train_test_similarities.loc[train_inchikeys, :]
            print(f"Filtered to {len(train_test_similarities)} unique inchikeys in the train-test similarity matrix")
        else: 
            train_test_similarities = None
        
        # Add the ground_true value to the predictions
        print("Calculating Error")
        predictions['error'] = (predictions['tanimoto'] - predictions['score']).abs()
        
        
        # Some handy code to check memory usage, if needed
        # for var_name, var in locals().items():
        #         print(f"{var_name}: {sys.getsizeof(var)}")
        #         if isinstance(var, pd.DataFrame):
        #             # Call memory_usage
        #             print(var.memory_usage(deep=True))
        
        # Check for symmetry, this is quite memory intensive, so it would be better run as an independent test
        # print("Performing Symmetry Sanity Check")
        # pairs_list = list(map(tuple, predictions[['spectrum_id_1', 'spectrum_id_2']].values))
        # pair_set = set(pairs_list)
        # assert len(pair_set) == len(pairs_list) # Ensure no duplices
        # del pairs_list
        # gc.collect()
        # reverse_pair_set = set(map(tuple, predictions[['spectrum_id_2', 'spectrum_id_1']].values))
        # # Check for symmetry
        # if pair_set != reverse_pair_set:
        #     asymmetric_pairs = pair_set.symmetric_difference(reverse_pair_set)
        #     print(f"Found asymmetric pairs: {asymmetric_pairs}")
        #     raise ValueError("Prediction pairs are not symmetric")
        # del pair_set, reverse_pair_set
        # gc.collect()
        
        ref_score_bins = np.linspace(0.2,1.0, 17)
        ### TEMP COMMENTS:
        
        overall_rmse = np.sqrt(np.mean(np.square(predictions['error'].values)))
        print("Overall RMSE (from evaluate()):", overall_rmse)
        overall_mae =  np.mean(np.abs(predictions['error'].values))

        # Train-Test Similarity Dependent Losses
        similarity_dependent_metrics_mean = train_test_similarity_dependent_losses(predictions, train_test_similarities, ref_score_bins, mode='mean')
        similarity_dependent_metrics_max = train_test_similarity_dependent_losses(predictions, train_test_similarities, ref_score_bins, mode='max')
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
        
            
        # Train-Test Similarity Dependent Losses Aggregated
        plt.figure(figsize=(12, 9))
        plt.bar(np.arange(len(similarity_dependent_metrics_max["rmses"]),), similarity_dependent_metrics_max["rmses"],)
        plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
        plt.xlabel("Max Test-Train Stuctural Similarity")
        plt.ylabel("Max Test-Train Stuctural Similarity")
        plt.ylabel("RMSE")
        plt.xlabel("Tanimoto score bin")
        plt.xticks(np.arange(len(similarity_dependent_metrics_max["rmses"])), [f"{a:.2f} to < {b:.2f}" for (a, b) in similarity_dependent_metrics_max["bounds"]], rotation='vertical')
        plt.grid(True)
        plt.savefig(os.path.join(metric_dir, 'train_test_rmse.png'))
        
        
        # MS2DeepScore Tanimoto Dependent Losses Plot
        ref_score_bins = np.linspace(0,1.0, 11)

        tanimoto_dependent_dict = tanimoto_dependent_losses(predictions, ref_score_bins)

        
        metric_dict = {}
        metric_dict["bin_content"]      = tanimoto_dependent_dict["bin_content"]
        metric_dict["nan_bin_content"]  = tanimoto_dependent_dict["nan_bin_content"]
        metric_dict["bounds"]           = tanimoto_dependent_dict["bounds"]
        metric_dict["rmses"]            = tanimoto_dependent_dict["rmses"]
        metric_dict["maes"]             = tanimoto_dependent_dict["maes"]
        metric_dict["rmse"] = overall_rmse
        metric_dict["mae"]  = overall_mae
        
        # Fixed-Tanimoto, Train-Test Dependent Plot
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
        
        # Train-Test Similarity Bar Plot
        train_test_sim = train_test_similarity_bar_plot(predictions, train_test_similarities, ref_score_bins)
        bin_content, bounds = train_test_sim['bin_content'], train_test_sim['bounds']
        plt.figure(figsize=(12, 9))
        plt.title("Number of Structures in Similarity Bins (Max Similarity to Train Set)")
        plt.bar(range(len(bin_content)), bin_content, label='Number of Structures')
        plt.xlabel('Similarity Bin (Max Similarity to Train Set)')
        plt.ylabel('Number of Structures')
        plt.xticks(range(len(bin_content)), [f"({bounds[i][0]:.2f}-{bounds[i][1]:.2f})" for i in range(len(bounds))], rotation=45)
        plt.legend()
        plt.savefig(os.path.join(metric_dir, 'train_test_similarity_bar_plot.png'), bbox_inches="tight")
        
        # Score, Tanimoto Scatter Plot
        plt.figure(figsize=grid_fig_size)
        plt.scatter(predictions['score'], predictions['tanimoto'], alpha=0.2)
        # Show R Squared
        r2 = np.corrcoef(predictions['score'], predictions['tanimoto'])[0,1]**2
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
        plt.figure(figsize=grid_fig_size)    
        hb = plt.hexbin(predictions['score'], predictions['tanimoto'], gridsize=50, cmap='inferno', bins='log')
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