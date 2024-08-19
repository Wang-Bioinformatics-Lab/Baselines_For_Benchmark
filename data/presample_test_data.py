import argparse
import os
from pathlib import Path

from tqdm import tqdm
from joblib import Parallel, delayed
import tempfile

import numpy  as np
import pandas as pd
import dask.dataframe as dd
import time

import pickle

from generator_filtered_pairs import FilteredPairsGenerator

def build_pairs( main_inchikey,
                inchikey_list,
                tanimoto_df,
                summary,
                train_test_sim_dict,
                pairs_generator=None,
                temp_file_dir=None,
                buffer_size=7_900_000,
                subsample=100_000):
    # Buffer Size Calculation:
    # 80 gb (leaving 40 for overhead) / 32 / 254 bytes per row * 0.75 for extra safety

    hdf_store = None
    hdf_path = None
    try:
        # Tempfile for hdf storage
        hdf_path = tempfile.NamedTemporaryFile(dir=temp_file_dir, delete=False)

        # Calculate the similarity list in a memory-efficent manner
        hdf_store = pd.HDFStore(hdf_path.name)

        spectrum_id_list_i = summary.loc[main_inchikey, ['spectrum_id']].values
        if len(spectrum_id_list_i.shape) == 2:
            spectrum_id_list_i = spectrum_id_list_i.squeeze()
        
        max_main_train_test_sim  = train_test_sim_dict[main_inchikey]['max']
        min_main_train_test_sim  = train_test_sim_dict[main_inchikey]['min']
        mean_main_train_test_sim = train_test_sim_dict[main_inchikey]['mean']

        columns=['inchikey1', 'inchikey2','spectrumid1', 'spectrumid2', 'ground_truth_similarity', 'inchikey1_max_test_sim', 'inchikey2_max_test_sim', 'mean_max_train_test_sim', 'mean_mean_train_test_sim',
                'max_max_train_test_sim', 'max_mean_train_test_sim', 'max_min_train_test_sim']


        output_list = []
        curr_buffer_size = buffer_size
        for inchikey_j in inchikey_list:

            max_j_train_test_sim  = train_test_sim_dict[inchikey_j]['max']
            min_j_train_test_sim  = train_test_sim_dict[inchikey_j]['min']
            mean_j_train_test_sim = train_test_sim_dict[inchikey_j]['mean']
            mean_mean_train_test_sim = np.mean([mean_main_train_test_sim, mean_j_train_test_sim])
            mean_max_train_test_sim = np.mean([max_main_train_test_sim, max_j_train_test_sim])
            max_max_train_test_sim = max(max_main_train_test_sim, max_j_train_test_sim)
            max_mean_train_test_sim = max(mean_main_train_test_sim, mean_j_train_test_sim)
            max_min_train_test_sim = max(min_main_train_test_sim, min_j_train_test_sim)
            gt = tanimoto_df.loc[main_inchikey, inchikey_j]

            spectrum_id_list_j = summary.loc[inchikey_j, ['spectrum_id']]
            if len(spectrum_id_list_j.shape) == 2:
                spectrum_id_list_j = spectrum_id_list_j.values.squeeze()            

            valid_pairs = None
            if pairs_generator is not None:
                # Filter by valid pairs only
                # Returns list of spectrum_id1, spectrumid2
                valid_pairs = pairs_generator.get_spectrum_pair_with_inchikey(main_inchikey, inchikey_j,
                                                                              return_type='ids',
                                                                              return_all=True)
                if valid_pairs == (None, None):
                    # No valid pairs, skip
                    continue

                # Structure as dataframe
                valid_pairs = pd.DataFrame(valid_pairs, columns=['spectrum_id_1', 'spectrum_id_2'])
                # Use a set since it's significantly faster than querying the index
                valid_spectra_1_ids = set(valid_pairs['spectrum_id_1'].values)
                valid_pairs = valid_pairs.set_index('spectrum_id_1')

            # For each pair of spectra in i, j, calculate the similarity
            for spectrum_id_i in spectrum_id_list_i:
                if valid_pairs is not None:
                    if spectrum_id_i not in valid_spectra_1_ids:
                        continue
                for spectrum_id_j in spectrum_id_list_j:
                    if valid_pairs is not None:
                        if spectrum_id_j not in valid_pairs.loc[spectrum_id_i, ['spectrum_id_2']].values:
                            continue
                    # The only time this computation will cross below the main diagonal.
                    # (in terms of spectrum_ids) is when the inchikeys are the same.
                    # When this happens, we only want to compute the similarity once so
                    # these cases are not weighted differently
                    if main_inchikey == inchikey_j:
                        if spectrum_id_i < spectrum_id_j:
                            continue
                    
                    output_list.append((main_inchikey,
                                        inchikey_j,
                                        spectrum_id_i,
                                        spectrum_id_j,
                                        gt,
                                        max_main_train_test_sim,
                                        max_j_train_test_sim,
                                        mean_max_train_test_sim,
                                        mean_mean_train_test_sim,
                                        max_max_train_test_sim,
                                        max_mean_train_test_sim,
                                        max_min_train_test_sim,))
                    # print(f"Appending took {time() - start_time} seconds")

                    curr_buffer_size -= 1
                    if curr_buffer_size == 0:
                        # Store the similarity at /main_inchikey/inchikey_j
                        # Should have spectrumid1, spectrumid2, ground_truth_similarity, predicted_similarity

                        output_frame = pd.DataFrame(output_list, columns=columns)

                        hdf_store.put(f"{main_inchikey}", output_frame, format='table', append=True, track_times=False,
                                        min_itemsize={'inchikey1':14, 'inchikey2':14, 'spectrumid1': 50, 'spectrumid2': 50})
                        # print(f"Storing took {time() - start_time} seconds")
                        curr_buffer_size = buffer_size
                        output_list = []

        # Dump the remaining content
        output_frame = pd.DataFrame(output_list, columns=columns)

        hdf_store.put(f"{main_inchikey}", output_frame, format='table', append=True, track_times=False,
                        min_itemsize={'inchikey1':14, 'inchikey2':14, 'spectrumid1': 50, 'spectrumid2': 50})
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

def build_pairs_subsampled(pairwise_similarities:pd.DataFrame,
                            train_test_similarities:pd.DataFrame,
                            train_test_sim_dict:dict,       # Precompute as dict to reduce dataframe lookups
                            summary:pd.DataFrame,
                            pairs_generator:FilteredPairsGenerator=None,
                            sampling_threshold:int=100_000,
                            buffer_size:int=100_000,
                            pairwise_index_range:tuple=(0, 20),
                            train_test_index_range:tuple=(0, 16),
                            temp_file_dir=None):
                                
    similarity_counts = np.zeros((20,16)) # First dimension is pairwise, second is mean(max(train-test similarity))

    def _get_lower_bound_pw(x):
        # Returns mapping to one of 20 bins within 0.2 to 1.0
        return np.linspace(0.0, 1.0, 21)[x]
    def _get_upper_bound_pw(x):
        out =  np.linspace(0.0, 1.0, 21)[x+1]
        if x == 19:
            out = 1.1   # Include 1.0
        return out
    def _get_lower_bound_tt(x):
        # Returns mapping to one of 20 bins within 0.2 to 1.0
        return np.linspace(0.2, 1.0, 17)[x]
    def _get_upper_bound_tt(x):
        out =  np.linspace(0.2, 1.0, 17)[x+1]
        if x == 19:
            out = 1.1
        return out

    curr_buffer_size = buffer_size
    output_buffer = []
    max_tries_per_bin = sampling_threshold * 2
    hdf_store = None
    hdf_path = None
    try:
        # Tempfile for hdf storage
        hdf_path = tempfile.NamedTemporaryFile(dir=temp_file_dir, delete=False)
        # Calculate the similarity list in a memory-efficent manner
        hdf_store = pd.HDFStore(hdf_path.name)
        columns=['inchikey1', 'inchikey2','spectrumid1', 'spectrumid2', 'ground_truth_similarity', 'inchikey1_max_test_sim', 'inchikey2_max_test_sim', 'mean_max_train_test_sim', 'mean_mean_train_test_sim',
                'max_max_train_test_sim', 'max_mean_train_test_sim', 'max_min_train_test_sim']

        # Iterate over bins and subsample spectra by inchikeys
        for i in range(pairwise_index_range[0], pairwise_index_range[1]):
            # Should be an array of [(inchikey1, inchikey2, pairwise_similarity), ...]
            pairwise_relevant = pairwise_similarities[(pairwise_similarities['value'] >= _get_lower_bound_pw(i)) &
                                                        (pairwise_similarities['value'] < _get_upper_bound_pw(i))]
            # print('pairwise_relevant', pairwise_relevant)

            for j in range(train_test_index_range[0], train_test_index_range[1]):
                train_test_relevant = train_test_similarities[(train_test_similarities['average_max_similarity'] >= _get_lower_bound_tt(j)) &
                                                            (train_test_similarities['average_max_similarity'] < _get_upper_bound_tt(j))]
                # print('train_test_relevant', train_test_relevant)
                
                # Get intersection of pairwise_relevant and train_test_relevant pairs
                selected = pairwise_relevant.index.intersection(train_test_relevant.index)
                if len(selected) == 0:
                    print(f"No pairs found for bin ({i},{j})")
                    continue
                selected_as_tuples = selected.values

                # Randomly sample from tuples and add spectra to output until the bin is full
                max_tries_per_bin = sampling_threshold * 2  # reset max_tries_per_bin
                while similarity_counts[i,j] < sampling_threshold:
                    max_tries_per_bin -= 1
                    if max_tries_per_bin <= 0:
                        print(f"Warning could not fill bin ({i},{j}) with {sampling_threshold} samples. Only got {similarity_counts[i,j]}.")
                        break
                    # Randomly sample from selected_as_tuples
                    selected_tuple = selected_as_tuples[np.random.randint(0, len(selected_as_tuples))]
                    inchikey1, inchikey2 = selected_tuple

                    max_i_train_test_sim  = train_test_sim_dict[inchikey1]['max']
                    min_i_train_test_sim  = train_test_sim_dict[inchikey1]['min']
                    mean_i_train_test_sim = train_test_sim_dict[inchikey1]['mean']
                    max_j_train_test_sim  = train_test_sim_dict[inchikey2]['max']
                    min_j_train_test_sim  = train_test_sim_dict[inchikey2]['min']
                    mean_j_train_test_sim = train_test_sim_dict[inchikey2]['mean']
                    mean_mean_train_test_sim = np.mean([mean_i_train_test_sim, mean_j_train_test_sim])
                    mean_max_train_test_sim = np.mean([max_i_train_test_sim, max_j_train_test_sim])
                    max_max_train_test_sim = max(max_i_train_test_sim, max_j_train_test_sim)
                    max_mean_train_test_sim = max(mean_i_train_test_sim, mean_j_train_test_sim)
                    max_min_train_test_sim = max(min_i_train_test_sim, min_j_train_test_sim)
                    gt = pairwise_similarities.loc[inchikey1, inchikey2].item()

                    # Get the spectra for inchikey1 and inchikey2
                    if pairs_generator is not None:
                        # Filter by valid pairs only
                        # Returns list of spectrum_id1, spectrumid2
                        spectrum_id1, spectrum_id2 = pairs_generator.get_spectrum_pair_with_inchikey(inchikey1, inchikey2,
                                                                                    return_type='ids')
                        if spectrum_id1 is None or spectrum_id2 is None:
                            # No valid pairs, skip
                            continue
                        # Randomly sample one spectrum
                        if spectrum_id1 == spectrum_id2:
                            # Skip if they are the same
                            continue
                    else:
                        # Random pair index
                        inchikey_i_spectra = summary.loc[selected_tuple[0], ['spectrum_id']]
                        inchikey_j_spectra = summary.loc[selected_tuple[1], ['spectrum_id']]
                        if len(inchikey_i_spectra.shape) == 2:
                            inchikey_i_spectra = inchikey_i_spectra.squeeze()
                        if len(inchikey_j_spectra.shape) == 2:
                            inchikey_j_spectra = inchikey_j_spectra.squeeze()
                        if len(inchikey_i_spectra) == 0 or len(inchikey_j_spectra) == 0:
                            # No spectra, skip
                            continue

                        spectrum_id1 = inchikey_i_spectra.iloc[np.random.randint(0, len(inchikey_i_spectra))]
                        spectrum_id2 = inchikey_j_spectra.iloc[np.random.randint(0, len(inchikey_j_spectra))]
                        if spectrum_id1 == spectrum_id2:
                            # Skip if they are the same
                            continue
                        

                    # Add to output
                    output_buffer.append((inchikey1, inchikey2, spectrum_id1, spectrum_id2, gt, max_i_train_test_sim, max_j_train_test_sim,
                                        mean_max_train_test_sim, mean_mean_train_test_sim, max_max_train_test_sim,
                                        max_mean_train_test_sim, max_min_train_test_sim))
                    curr_buffer_size -= 1
                    similarity_counts[i,j] += 1
                    if curr_buffer_size <=0:
                        output_frame = pd.DataFrame(output_buffer, columns=columns)
                        # The name doesn't actually matter, just as long as it's unique per process (it all get's concatenated at the end and chunked by dask)
                        hdf_store.put(f"dump_{i}_{j}", output_frame, format='table', append=True, track_times=False,
                                        min_itemsize={'inchikey1':14, 'inchikey2':14, 'spectrumid1': 50, 'spectrumid2': 50})
                        curr_buffer_size = buffer_size
                        output_buffer = []
        # Dump the remaining content
        output_frame = pd.DataFrame(output_buffer, columns=columns)
        hdf_store.put(f"dump_{pairwise_index_range[1]}_{train_test_index_range[1]}", output_frame, format='table', append=True, track_times=False,
                        min_itemsize={'inchikey1':14, 'inchikey2':14, 'spectrumid1': 50, 'spectrumid2': 50})
        hdf_store.close()
        hdf_path.close()
        curr_buffer_size = buffer_size
        output_buffer = []
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

    return hdf_path.name, similarity_counts

def build_test_pairs_list(  test_summary_path:str,
                            pairwise_similarities_path:str,
                            train_test_similarity_path:str,
                            test_data_path:str,
                            output_path:str,
                            n_jobs:int=1,
                            subsample:int=None,
                            merge_on_lst=None,
                            mass_analyzer_lst=None,
                            collision_energy_thresh=5.0,
                            filter=False,
                            temp_file_dir=None,
                            inital_index=0,
                            skip_merge=False,
                            strict_collision_energy=False):

    if not filter:
        if merge_on_lst is not None or mass_analyzer_lst is not None or collision_energy_thresh != 5.0:
            print("Warning: Filtering is not enabled, but merge_on_lst, mass_analyzer_lst, or collision_energy_thresh is set. Ignoring filtering parameters.")
    
    # if subsample is not None and n_jobs != 1:
    #     print("Warning: Subsampling is enabled, but n_jobs is not 1. Only one job will be used for subsampling.")
    #     num_jobs = 1

    # Load test summary
    print("Loading test summary...", flush=True)
    test_summary = pd.read_csv(test_summary_path)
    if 'InChIKey_smiles_14' not in test_summary.columns:
        # Try to use InChIKey_smiles of InChIKey
        if 'InChIKey' in test_summary.columns:
            test_summary['InChIKey_smiles_14'] = test_summary['InChIKey'].str[:14]
        elif 'InChIKey_smiles' in test_summary.columns:
            test_summary['InChIKey_smiles_14'] = test_summary['InChIKey_smiles'].str[:14]
        else:
            raise ValueError("No InChIKey, InChIKey_smiles, or InChIKey_smiles_14 column found in test summary file.")
    # Remove anything that doesn't have spectra
    test_spectra = pickle.load(open(test_data_path, 'rb'))
    test_ids = set([spectrum.get("spectrum_id") for spectrum in test_spectra])
    test_summary = test_summary[test_summary['spectrum_id'].isin(test_ids)]
    del test_spectra, test_ids

    # Index by inchikey_14
    test_summary = test_summary.set_index('InChIKey_smiles_14')

    # Load train_test similarities
    print("Loading train-test similarities...", flush=True)
    train_test_similarities = pd.read_csv(train_test_similarity_path, index_col=0)
    # train_test_similarities = train_test_similarities.loc[:, test_summary.index.unique()]
    # DEBUG
    # train_inchis = train_test_similarities.index[~ (train_test_similarities.index.isin(test_summary.index.unique()))]
    # train_test_similarities = train_test_similarities.loc[train_inchis.unique(), test_summary.index.unique()]
    print("DEBUG -- train_test_similarities")
    print("Min", train_test_similarities.min().min())
    print("Max", train_test_similarities.max().max())
    print("Mean", train_test_similarities.mean().mean())

    # Load pairwise similarities
    tanimoto_df = pd.read_csv(pairwise_similarities_path, index_col=0)
    # tanimoto_df = tanimoto_df.loc[test_summary.index.unique(), test_summary.index.unique()]

    print("DEBUG -- pairwis similarities")
    print("Min", tanimoto_df.min().min())
    print("Max", tanimoto_df.max().max())
    print("Mean", tanimoto_df.mean().mean())

    unique_test_inchikeys = test_summary.index.unique()
    
    # Remove anything that doesn't have an inchikey in the pairwise similarities, it didn't make it through pre-processing
    unique_test_inchikeys = [inchikey for inchikey in unique_test_inchikeys if inchikey in tanimoto_df.index]
    test_summary = test_summary.loc[unique_test_inchikeys]
    # Remove bits we can with filtering
    if filter:
        local_mass_analyzer_lst = mass_analyzer_lst.split(';') if mass_analyzer_lst is not None else None
        if mass_analyzer_lst is not None:
            test_summary = test_summary[test_summary['msMassAnalyzer'].isin(local_mass_analyzer_lst)]
        local_merge_on_lst = merge_on_lst.split(';') if merge_on_lst is not None else []
        if merge_on_lst is None:
            local_merge_on_lst = ['ms_mass_analyzer', 'ms_ionisation', 'adduct']
        if 'ms_mass_analyzer' in local_merge_on_lst:
            test_summary = test_summary.dropna(subset=['msMassAnalyzer'])
        if 'ms_ionisation' in local_merge_on_lst:
            test_summary = test_summary.dropna(subset=['msIonisation'])
        if 'adduct' in local_merge_on_lst:
            test_summary = test_summary.dropna(subset=['Adduct'])

    # Remove the filtered parts + remove any inchikeys that have no spectra in the summary
    unique_test_inchikeys = test_summary.index.unique()
    tanimoto_df = tanimoto_df.loc[unique_test_inchikeys, unique_test_inchikeys]
    
    train_test_sim_dict = {}
    print("Precomputing train-test similarities...", flush=True)
    for inchikey in tqdm(unique_test_inchikeys):
        main_train_test_sim = train_test_similarities.loc[:, inchikey]
        train_test_sim_dict[inchikey] = {'max': main_train_test_sim.max(),
                                        'min': main_train_test_sim.min(),
                                        'mean': main_train_test_sim.mean()}

    # If we're filtering, create a pairs_generator
    if filter:
        # Rename col to expected
        test_summary['inchikey_14'] = test_summary.index
        test_summary['collisionEnergy'] = test_summary['collision_energy']
        pairs_generator = FilteredPairsGenerator(test_summary,
                                                tanimoto_df,
                                                ignore_equal_pairs=False,   # Will explicitly be handled in build_pairs
                                                merge_on_lst=merge_on_lst,
                                                mass_analyzer_lst=mass_analyzer_lst,
                                                collision_energy_thresh=collision_energy_thresh,
                                                strict_collision_energy=strict_collision_energy)
    else:
        pairs_generator = None

    # Compute parallel scores using _parallel_cosine
    output_hdf_files = []
    hdf_store = None
    temp_store = None
    try:
        if subsample is None:
            print("Generating pairs of test data exhausively...", flush=True)
            output_hdf_files = Parallel(n_jobs=n_jobs)(delayed(build_pairs)(unique_test_inchikeys[i],
                                                                            unique_test_inchikeys[i:],
                                                                            tanimoto_df,
                                                                            test_summary,
                                                                            train_test_sim_dict,
                                                                            pairs_generator=pairs_generator,
                                                                            temp_file_dir=temp_file_dir,)
                                                                            for i in tqdm(range(inital_index, len(unique_test_inchikeys))))
        else:
            print(f"Generating pairs of test data with subsampling of {subsample}...", flush=True)
            
            # Prepare pairwise similarities for subsampling
            pairwise_similarities = tanimoto_df.melt(ignore_index=False, var_name='InChIKey_smiles_14', value_name='value')
            print(pairwise_similarities)
            pairwise_similarities = pairwise_similarities.set_index([pairwise_similarities.index, 'InChIKey_smiles_14']) # Sets index to inchikey1, inchikey2
            
            # Prepare train-test similarities for subsampling
            prepped_train_test_similarities = train_test_similarities.max(axis=0).to_frame().reset_index() # Produces a series of max similarity values to all train inchikeys
            prepped_train_test_similarities.columns = ['InChIKey_smiles_14', 'value']
            # Calculate the cartesiant product of prepped_train_test_similarities and take the average
            first_col = prepped_train_test_similarities.assign(a=1)
            second_col = prepped_train_test_similarities.assign(a=1)
            prepped_train_test_similarities = first_col.merge(second_col, on='a', suffixes=('_1', '_2'))
            # Get upper triangle
            prepped_train_test_similarities = prepped_train_test_similarities[prepped_train_test_similarities['InChIKey_smiles_14_1'] <= prepped_train_test_similarities['InChIKey_smiles_14_2']]
            prepped_train_test_similarities['average_max_similarity'] = prepped_train_test_similarities[['value_1', 'value_2']].mean(axis=1)
            # Clean up
            prepped_train_test_similarities.drop(columns=['value_1', 'value_2', 'a'], inplace=True)
            prepped_train_test_similarities.set_index(['InChIKey_smiles_14_1', 'InChIKey_smiles_14_2'], inplace=True)
            print("Train-Test Similarities")
            print(prepped_train_test_similarities)
            
            if n_jobs > 1:
                grid = [(i, j) for i in range(0, 20) for j in range(0, 16)]
                outputs = Parallel(n_jobs=n_jobs)(delayed(build_pairs_subsampled)(pairwise_similarities,
                                                                                                prepped_train_test_similarities,
                                                                                                train_test_sim_dict,
                                                                                                test_summary,
                                                                                                pairs_generator=pairs_generator,
                                                                                                sampling_threshold=subsample,
                                                                                                temp_file_dir=temp_file_dir,
                                                                                                pairwise_index_range=(i, i+1),
                                                                                                train_test_index_range=(j, j+1))
                                                                                                for (i,j) in tqdm(grid))
                output_hdf_files = [output[0] for output in outputs]
                counts = np.sum([output[1] for output in outputs], axis=0)
            else:
                output_hdf_files, counts = build_pairs_subsampled(pairwise_similarities,
                                                          prepped_train_test_similarities,
                                                          train_test_sim_dict,
                                                          test_summary,
                                                          pairs_generator=pairs_generator,
                                                          sampling_threshold=subsample,
                                                          temp_file_dir=temp_file_dir)
            
            print(counts)
            if not isinstance(output_hdf_files, list):
                output_hdf_files = [output_hdf_files]

        if skip_merge:
            return

        print(output_hdf_files)
        start_time = time.time()
        dask_df = dd.read_hdf(output_hdf_files, key='/*')
        print(f"Reading the hdf file lazily took {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        # Will concatenate to single table, but use parallelism
        dask_df.to_parquet(output_path, overwrite=True, compute=True)
        print(f"Writing the hdf file took {time.time() - start_time:.2f} seconds")
    finally:
        if hdf_store is not None and hdf_store.is_open:
            hdf_store.close()
        
        if temp_store is not None and temp_store.is_open:
            temp_store.close()

        for hdf_file in output_hdf_files:
            if os.path.exists(hdf_file):
                os.remove(hdf_file)

def main():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata csv file.")
    parser.add_argument("--test_data_path", type=str, help="Path to the test data pickle file.")
    parser.add_argument("--pairwise_similarities_path", type=str, help="Path to the pairwise similarities csv file.")
    parser.add_argument("--train_test_similarities_path", help="Path to the train-test similarity csv file.")
    parser.add_argument("--output_path", type=str, help="Path to the output directory.")
    parser.add_argument("--temp_file_dir", type=str, help="Path to the temporary directory for storing intermediate files.", default=None)
    parser.add_argument("--inital_index", type=int, help="The initial index to start from.", default=0)
    parser.add_argument("--skip_merge", action='store_true', help="Whether to skip merging the output files.")
    parser.add_argument("--subsample", type=int, help="The number of spectra to subsample for each (train-test, pairwise) similarity bin")
    parser.add_argument("--filter", action='store_true', help="Whether to filter the pairs.")
    # Filtering parameters
    parser.add_argument('--merge_on_lst', type=str, help='A semicolon delimited list of criteria to merge on. \
                                                            Options are: ["ms_mass_analyzer", "ms_ionisation", "adduct", "library"].\
                                                            Default behavior is to filter on ["ms_mass_analyzer", "ms_ionisation", "adduct"]. \
                                                            Ignored if --filter is not set.',
                                                    default=None)
    parser.add_argument('--mass_analyzer_lst', type=str, default=None,
                                                help='A semicolon delimited list of allowed mass analyzers. All mass analyzers are \
                                                      allowed when not specified. Ignored if --filter is not set.')
    parser.add_argument('--collision_energy_thresh', type=float, default=5.0,
                                                help='The maximum difference between collision energies of two spectra to be considered\
                                                    a pair. Default is <= 5. "-1.0" means collision energies are not filtered. \
                                                    If no collision enegy is available for either spectra, both will be included. \
                                                    Ignored if --filter is not set.')
    parser.add_argument('--strict_collision_energy', action='store_true', help='Whether to strictly filter by collision energy. \
                                                                                Unless set, pairs with NaN collision energy will be included.')
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    assert metadata_path.exists(), f"Metadata file {metadata_path} does not exist."
    test_data_path = Path(args.test_data_path)
    assert test_data_path.exists(), f"Test data file {test_data_path} does not exist."
    pairwise_similarities_path = Path(args.pairwise_similarities_path)    
    assert pairwise_similarities_path.exists(), f"Pairwise similarities file {pairwise_similarities_path} does not exist."
    train_test_similarities_path = Path(args.train_test_similarities_path)
    assert train_test_similarities_path.exists(), f"Train-test similarities file {train_test_similarities_path} does not exist."
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Warning about disk usage
    print("WARNING: This script will use a significant amount of disk space to cache intermediate computation.", flush=True)
    print("Ensure that you have enough disk space available before proceeding. Continuing in 5 seconds.", flush=True)
    # time.sleep(5)

    # Build the test pairs list
    build_test_pairs_list(metadata_path,
                          pairwise_similarities_path,
                          train_test_similarities_path,
                          test_data_path,
                          output_path,
                          n_jobs=args.n_jobs,
                          subsample=args.subsample,
                          merge_on_lst=args.merge_on_lst,
                          mass_analyzer_lst=args.mass_analyzer_lst,
                          collision_energy_thresh=args.collision_energy_thresh,
                          filter=args.filter,
                          temp_file_dir=args.temp_file_dir,
                          inital_index=args.inital_index,
                          skip_merge=args.skip_merge,
                          strict_collision_energy=args.strict_collision_energy)



if __name__ == "__main__":
    main()