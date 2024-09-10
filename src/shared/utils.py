import pandas as pd
import numpy as np
from matchms.filtering import add_fingerprint
from matchms.similarity import FingerprintSimilarity
from dask import delayed, compute
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from concurrent.futures import ThreadPoolExecutor
import dask
import os
from tqdm import tqdm

def roc_curve(prediction_df:dask.dataframe, threshold_lst:str):
    """Compute the ROC curve for the given prediction dataframe and threshold list.
    Excludes identical spectrum ids from consideration.

    Parameters
    ----------
    prediction_df : dask.dataframe
        DataFrame containing the predictions and ground truth similarities.
    threshold_lst : str
        List of thresholds to evaluate the ROC curve.

    Returns
    -------
    dict
        Dictionary containing the true positive rate, false positive rate, AUC, and thresholds.
    """
    # Reduce dataframe selection to pairs with < 10 ppm
    prediction_df = prediction_df.loc[prediction_df['precursor_ppm_diff'] < 10]
    # Remove predictions with identical spectrum_ids
    prediction_df = prediction_df.loc[prediction_df['spectrumid1'] != prediction_df['spectrumid2']]
    # Remove failed predictions
    prediction_df = prediction_df.loc[prediction_df['predicted_similarity'] != -1]

    prediction_df['positive'] = prediction_df['inchikey1'] == prediction_df['inchikey2']

    tp_lst = []
    fp_lst = []
    tn_lst = []
    fn_lst = []

    for threshold in threshold_lst:
        tp = delayed(sum)(prediction_df.loc[prediction_df['predicted_similarity'] >= threshold]['positive'])
        fp = delayed(sum)(prediction_df.loc[prediction_df['predicted_similarity'] >= threshold]['positive'] == False)
        tn = delayed(sum)(prediction_df.loc[prediction_df['predicted_similarity'] < threshold]['positive'] == False)
        fn = delayed(sum)(prediction_df.loc[prediction_df['predicted_similarity'] < threshold]['positive'])

        tp_lst.append(tp)
        fp_lst.append(fp)
        tn_lst.append(tn)
        fn_lst.append(fn)

    with ProgressBar(minimum=1.0):
        tp_lst, fp_lst, tn_lst, fn_lst = compute(tp_lst, fp_lst, tn_lst, fn_lst)

    fpr = [fp/(fp + tn) for fp, tn in zip(fp_lst, tn_lst)]
    tpr = [tp/(tp + fn) for tp, fn in zip(tp_lst, fn_lst)]

    auc = np.trapz(tpr, fpr)
    print(fpr)
    print(tpr)

    return {'tp': tp_lst,
            'fp': fp_lst,
            'tn': tn_lst,
            'fn': fn_lst,
            'fpr': fpr,
            'tpr':tpr,
            'auc': auc,
            'thresholds': threshold_lst}

def get_top_k_scores(prediction_df:dask.dataframe, k=1, remove_identical_inchikeys:bool=False)->dict:
    """Get the tanimoto scores of the top-k predictions for each spectrum. Optionally, excludes pairs with identical inchikeys.
    Also returns the theoretical maximum score for each spectrum.

    Parameters
    ----------
    prediction_df : dask.dataframe
        DataFrame containing the predictions and ground truth similarities.
    k : int, optional
        The number of top predictions to consider, by default 1
    remove_identical_inchikeys : bool, optional
        If True, exclude pairs with identical inchikeys, by default False

    Returns
    -------
    dict
        Dictionary containing the top-k scores and theoretical maximum scores for each spectrum.
    """
    prediction_df = prediction_df.loc[prediction_df['spectrumid1'] != prediction_df['spectrumid2']]

    if remove_identical_inchikeys:
        prediction_df = prediction_df.loc[prediction_df['inchikey1'] != prediction_df['inchikey2']]

    # Make symmetric
    reversed_df = prediction_df.rename(columns={'spectrumid1': 'spectrumid2', 'spectrumid2': 'spectrumid1',
                                            'inchikey1': 'inchikey2', 'inchikey2': 'inchikey1'})
    prediction_df = dd.concat([prediction_df, reversed_df], axis=0)

    # Sort by predicted similarity in descending order
    prediction_df = prediction_df.sort_values('predicted_similarity', ascending=False)

    # Group by spectrum
    grouped = prediction_df.groupby('spectrumid1')

    def calculate_top_k_scores(df):
        h = df['ground_truth_similarity'].head(k).values
        # Right pad with np.nan if there are less than k values
        return np.pad(h, (0, k - len(h)), constant_values=np.nan)

    # Get top-k scores within groups
    top_k_scores = grouped.apply(calculate_top_k_scores, meta=('top_k_scores', 'f8'))

    def calculate_max_score(df):
        h = df['ground_truth_similarity'].values
        h = h[np.argsort(-h)]
        h = h[:k]
        # Right pad with np.nan if there are less than k values
        return np.pad(h, (0, k - len(h)), constant_values=np.nan)

    # Get top-k ground truth scores within groups
    max_scores = grouped.apply(calculate_max_score, meta=('max_scores', 'f8'))

    # Compute both top_k_scores and max_scores in a single compute call
    top_k_scores, max_scores = dask.compute(top_k_scores, max_scores)

    # Convert the results to dictionaries
    top_k_scores = top_k_scores.to_dict()
    max_scores = max_scores.to_dict()

    keys = list(top_k_scores.keys())
    
    difference_dict = {k: max_scores[k] - top_k_scores[k] for k in keys}
    
    top_scores_array = np.stack([top_k_scores[key] for key in keys])
    max_scores_array = np.stack([max_scores[key] for key in keys])
    difference_array = np.stack([difference_dict[key] for key in keys])

    # Get average from 0-0, 0,1,... 0-k-1
    top_scores_averages = [np.nanmean(top_scores_array[:, :i+1]) for i in range(k)]
    max_scores_averages = [np.nanmean(max_scores_array[:, :i+1]) for i in range(k)]
    difference_averages = [np.nanmean(difference_array[:, :i+1]) for i in range(k)]

    return {'top_k_scores': top_k_scores, 'max_scores': max_scores, 'difference_dict': difference_dict, 
            'top_scores_averages': top_scores_averages, 'max_scores_averages': max_scores_averages, 'difference_averages': difference_averages}

def top_k_analysis(prediction_df: dd.DataFrame, k_list=[1], remove_identical_inchikeys: bool = False) -> dict:
    """Get the rank of the most similar structure in the prediction_df for each spectrum.
    Optionally, excludes pairs with identical inchikeys simulating an analogue search setting.
    Always excludes pairs with identical spectrum ids.

    Parameters
    ----------
    prediction_df : dask.dataframe
        DataFrame containing the predictions and ground truth similarities.
    k_list : list, optional
        List of k values to consider, by default [1]
    remove_identical_inchikeys : bool, optional
        If True, exclude pairs with identical inchikeys, by default False

    Returns
    -------
    dict
        Dictionary containing the mean and standard deviation of ranks for each value of k.
    """
    
    prediction_df = prediction_df.loc[prediction_df['spectrumid1'] != prediction_df['spectrumid2']]

    if remove_identical_inchikeys:
        prediction_df = prediction_df.loc[prediction_df['inchikey1'] != prediction_df['inchikey2']]

    # Make symmetric
    reversed_df = prediction_df.rename(columns={'spectrumid1': 'spectrumid2', 'spectrumid2': 'spectrumid1',
                                                'inchikey1': 'inchikey2', 'inchikey2': 'inchikey1'})
    prediction_df = dd.concat([prediction_df, reversed_df], axis=0)

    # Sort by predicted similarity in descending order
    prediction_df = prediction_df.sort_values('predicted_similarity', ascending=False)

    # Group by spectrum
    grouped = prediction_df.groupby('spectrumid1')

    # Get ranks within groups
    prediction_df['rank'] = grouped.cumcount() + 1

    # Function to calculate normalized rank
    def calculate_rank(df, _k):
        most_similar_inchikeys_with_ties = df.drop_duplicates(subset=['inchikey2']).set_index('inchikey2')['ground_truth_similarity'].nlargest(_k, keep='all')
        max_gt_rows = df[df['inchikey2'].isin(most_similar_inchikeys_with_ties.index)]
        max_gt_rows = max_gt_rows.nsmallest(_k, keep='first', columns='rank')
        min_ranks_for_max_gt = max_gt_rows['rank'].values
        max_ranks = min(_k, len(max_gt_rows.ground_truth_similarity.unique()))
        # min(_k,max_ranks) is used because on filtered data, it is possible that there are less than k rows
        normalized_rank = min_ranks_for_max_gt.mean() - np.mean(np.arange(1, min(_k,max_ranks) + 1))

        return normalized_rank

    results = {}
    ranks_dict = {}

    for k in k_list:
        ranks = grouped.apply(calculate_rank, k, meta=('rank', 'f8'))
        ranks_dict[k] = ranks

    # Compute mean and standard deviation for each k in one go
    means_and_stds = dd.compute({k: (r.mean(), np.median(r), r.std(), r) for k, r in ranks_dict.items()})[0]

    for k, (mean, median, std, ranks) in means_and_stds.items():
        results[k] = {}
        results[k]['median_top_rank'] = median
        results[k]['mean_top_rank'] = mean
        results[k]['std_top_rank'] = std
        results[k]['all_ranks'] = ranks.to_dict()

    return results

def get_structural_similarity_matrix(a, a_labels, b=None, b_labels=None, fp_type='', similarity_measure="jaccard"):
    if b is None or b_labels is None:
        assert b is None and b_labels is None
        
    # Add the fingerprints to all of the spectra
    a = [add_fingerprint(s, fp_type=fp_type) for s in a]
    if b is None:
        b = a
        b_labels = a_labels
    else:
        b = [add_fingerprint(s, fp_type=fp_type) for s in b]

    similarity_measure = FingerprintSimilarity(similarity_measure=similarity_measure)
    scores_mol_similarity = similarity_measure.matrix(a, b)
    return pd.DataFrame(scores_mol_similarity, columns=a_labels, index=b_labels)

def train_test_similarity_dependent_losses(prediction_df, ref_score_bins, mode='max'):
    bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[-1] = np.inf
    
    assert mode in ['max', 'mean', 'asms']
    
    tasks = []
    for i in range(len(ref_scores_bins_inclusive)-1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i+1]
        bounds.append((low, high))
        if mode == 'max':
            relevant_rows = prediction_df.loc[(prediction_df['max_max_train_test_sim'] > low) &
                                              (prediction_df['max_max_train_test_sim'] <= high)]
        elif mode == 'mean':
            relevant_rows = prediction_df.loc[(prediction_df['mean_max_train_test_sim'] > low) &
                                              (prediction_df['mean_max_train_test_sim'] <= high)]
        elif mode == 'asms':    # At ASMS we used the max similarity to the test set for the left spectrum to bin RMSEs
            relevant_rows = prediction_df.loc[(prediction_df['inchikey1_max_test_sim'] > low) &
                                              (prediction_df['inchikey1_max_test_sim'] <= high)]
        else:
            raise ValueError(f"Unknown mode {mode}")

        bin_content.append(delayed(len)(relevant_rows))
        maes.append(delayed(np.nanmean)(np.abs(relevant_rows['error'].values)))
        rmses.append(delayed(np.sqrt)(delayed(np.nanmean)(np.square(relevant_rows['error'].values))))
    
    with ProgressBar(minimum=1.0):
        # with dask.config.set(pool=ThreadPoolExecutor(min(4, os.cpu_count()))):
            bin_content, maes, rmses = compute(bin_content, maes, rmses)

    # Ensure bounds are ordered
    for i in range(len(bounds)-1):
        assert bounds[i][1] == bounds[i+1][0], f"Bounds are not ordered: {bounds}"
    
    return {'bin_content': bin_content, 'bounds': bounds, 'rmses': rmses, 'maes': maes}

def train_test_similarity_heatmap(prediction_df, train_test_similarity, ref_score_bins):
    raise NotImplementedError("This function is not yet impletemented for dask")
    # TODO: Current prediction_df does not have left and right similarity, therefore we cannot make the heatmap
    # This figure will likely not be in the final publication because it's not informative anyways
    # Therefore, we will not implement this function for now

    # prediction_df is a pandas dataframe with columns spectruim_id_1, spectrum_id_2, inchikey_1, inchikey_2, score, error
    ref_scores_bins_inclusive = ref_score_bins.copy()
    # ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    
    nan_count_grid   = np.zeros((len(ref_score_bins)-1, len(ref_score_bins)-1))
    bin_content_grid = np.zeros((len(ref_score_bins)-1, len(ref_score_bins)-1))
    rmse_grid = np.zeros((len(ref_score_bins)-1, len(ref_score_bins)-1))
    mae_grid = np.zeros((len(ref_score_bins)-1, len(ref_score_bins)-1))
    bound_grid =  [[None for _ in range(len(ref_score_bins)-1)] for _ in range(len(ref_score_bins)-1)]
    
    all_predicted_inchikeys = prediction_df['inchikey_1'].unique()  # We assume square matrix
    
    # Only include if the keys are in the error matrix
    train_test_similarity = train_test_similarity.loc[:, all_predicted_inchikeys]
    
    # This will become a test inchi->max train_test similarity mapping 
    train_test_similarity = train_test_similarity.max(axis=0)
    
    for i in range(len(ref_scores_bins_inclusive)-1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i+1]
        
        relevant_test_inchis_i = train_test_similarity.loc[(train_test_similarity > low) & (train_test_similarity <= high)].index
        
        for j in range(len(ref_scores_bins_inclusive)-1):
            low_j = ref_scores_bins_inclusive[j]
            high_j = ref_scores_bins_inclusive[j+1]
            
            relevant_test_inchis_j = train_test_similarity.loc[(train_test_similarity > low_j) & (train_test_similarity <= high_j)].index
            
            relevant_values = prediction_df.loc[(prediction_df['inchikey_1'].isin(relevant_test_inchis_i)) & (prediction_df['inchikey_2'].isin(relevant_test_inchis_j))]['error']
            
            nan_count_grid[i, j] = relevant_values.isna().sum().sum()
            bin_content_grid[i, j] = relevant_values.notna().sum().sum()
            bound_grid[i][j] = ((low, high), (low_j, high_j))
            rmse_grid[i, j] = np.nanmean(np.sqrt(np.square(relevant_values).values))
            mae_grid[i, j] = np.nanmean(np.abs(relevant_values).values)
            
    # Set zeros to nan
    nan_count_grid[nan_count_grid == 0] = np.nan
    bin_content_grid[bin_content_grid == 0] = np.nan
    rmse_grid[rmse_grid == 0] = np.nan
    mae_grid[mae_grid == 0] = np.nan
    
    # Assert rmse and mae is symmetric
    # assert np.allclose(rmse_grid, rmse_grid.T, equal_nan=True), f"RMSE is not symmetric: {rmse_grid}"
    # assert np.allclose(mae_grid, mae_grid.T, equal_nan=True), f"MAE is not symmetric: {mae_grid}"
    
    return {'bin_content':bin_content_grid, 'bounds':bound_grid, 'rmses':rmse_grid, 'maes':mae_grid, 'nan_count':nan_count_grid}

def train_test_similarity_bar_plot(prediction_df, train_test_similarity, bins):
        ref_scores_bins_inclusive = bins.copy()
        # ref_scores_bins_inclusive[0] = -np.inf
        ref_scores_bins_inclusive[-1] = np.inf
        
        all_predicted_inchikeys = prediction_df['inchikey_1'].unique()  # We assume square matrix
        
        train_test_similarity = train_test_similarity.loc[:, all_predicted_inchikeys]
        train_test_similarity = train_test_similarity.max(axis=0)
        
        bin_content = []
        
        for i in range(len(ref_scores_bins_inclusive)-1):
            low = ref_scores_bins_inclusive[i]
            high = ref_scores_bins_inclusive[i+1]
            relevant_keys = train_test_similarity.loc[(train_test_similarity > low) & (train_test_similarity <= high)].index
            bin_content.append(relevant_keys.shape[0])
                    
        
        return {'bin_content':bin_content, 'bounds':[(ref_scores_bins_inclusive[i], ref_scores_bins_inclusive[i+1]) for i in range(len(ref_scores_bins_inclusive)-1)]}

def tanimoto_dependent_losses(prediction_df, ref_score_bins):
    """Compute errors (RMSE and MAE) for different bins of the reference scores (scores_ref).
    
    Parameters
    ----------
    prediction_df : DataFrame
        DataFrame containing the predictions and ground truth similarities.
        
    ref_score_bins : list
        Bins for the reference score to evaluate the performance of scores.
    
    Returns
    -------
    dict
        Dictionary containing bin contents, bounds, RMS errors, MA errors, and nan bin contents.
    """
    
    bin_content = []
    nan_bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf

    tasks = []
    for i in range(len(ref_scores_bins_inclusive) - 1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i + 1]
        bounds.append((low, high))

        relevant_rows = prediction_df.loc[(prediction_df['ground_truth_similarity'] >= low) &
                                          (prediction_df['ground_truth_similarity'] < high)]

        nan_bin_task = delayed(sum)(relevant_rows['predicted_similarity'].isna())
        bin_content_task = delayed(sum)((~relevant_rows['predicted_similarity'].isna()))
        mae_task = delayed(np.nanmean)(relevant_rows['error'].values)
        rmse_task = delayed(np.sqrt)(delayed(np.nanmean)(np.square(relevant_rows['error'].values)))

        tasks.append((nan_bin_task, bin_content_task, mae_task, rmse_task))

    # Save computation graph to image
    # dask.visualize(tasks, filename='/home/mstro016/tanimoto_dependent_losses.png')

    with ProgressBar(minimum=1.0):
        results = compute(*tasks)

    # Unpack results using tuple unpacking
    for idx in range(len(tasks)):
        nan_bin_content.append(results[idx][0])
        bin_content.append(results[idx][1])
        maes.append(results[idx][2])
        rmses.append(results[idx][3])

    print("Results:")
    print(bin_content)
    print(nan_bin_content)
    print(rmses)
    print(maes)
    print(bounds)
    print("---------------", flush=True)

    return {'bin_content': bin_content, 'bounds': bounds, 'rmses': rmses, 'maes': maes, 'nan_bin_content': nan_bin_content}

def fixed_tanimoto_train_test_similarity_dependent(prediction_df, train_test_similarity, ref_score_bins):
    raise NotImplementedError("This function is not yet impletemented for dask")
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    
    output_dict = {}
    
    all_predicted_inchikeys = prediction_df['inchikey_1'].unique()  # We assume square matrix
    #  Get keys for the current train-test similarity bin
    local_train_test_similarity = train_test_similarity.loc[:, all_predicted_inchikeys].max(axis=0)
    
    # Annotate prediction_df with averate train-test similarity
    prediction_df['train_test_similarity'] = prediction_df.apply(lambda x: np.mean([local_train_test_similarity[x['inchikey_1']], local_train_test_similarity[x['inchikey_2']]]).astype('float16'), axis=1)
    
    for i in range(len(ref_scores_bins_inclusive)-1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i+1]
              
        # local_train_test_similarity = local_train_test_similarity.loc[(local_train_test_similarity > low) & (local_train_test_similarity <= high)]
        # train_test_keys_a = local_train_test_similarity.index
        # train_test_keys_b = train_test_keys_a
        
        # Predictions that are in the current train-test similarity bin
        relevant_predictions = prediction_df.loc[(prediction_df.train_test_similarity > low) & (prediction_df.train_test_similarity <= high)].copy()
        
        if i == 0:
            key = f'({0.0},{high:.2f})'
        elif i == len(ref_scores_bins_inclusive)-1:
            key = f'({low:.2f},{1.0})'
        else:
            key = f'({low:.2f},{high:.2f})'
        
        output_dict[key] = {}
        
        for j in range(len(ref_scores_bins_inclusive)-1):
            local_low = ref_scores_bins_inclusive[j]
            local_high = ref_scores_bins_inclusive[j+1]
            
            if j == 0:
                sub_key = f'({0.0},{local_high:.2f})'
            elif j == len(ref_scores_bins_inclusive)-1:
                sub_key = f'({local_low:.2f},{1.0})'
            else:
                sub_key = f'({local_low:.2f},{local_high:.2f})'
            
            output_dict[key][sub_key] = {}
            
            # Relevant values contains the predictions that are in the current train-test similarity bin and the current reference score bin
            relevant_values = relevant_predictions.loc[(relevant_predictions.tanimoto >= low) & (relevant_predictions.tanimoto < high)].copy()
            
            output_dict[key][sub_key]['bin_content'] = relevant_values.shape[0]
            
            errors = relevant_values['error']
            output_dict[key][sub_key]['mae'] = np.nanmean(np.abs(errors).values)
            output_dict[key][sub_key]['rmse'] = np.sqrt(np.nanmean(np.square(errors).values))
    # First index is the train-test similarity bin, second index is the reference score bin
    return output_dict

def pairwise_train_test_dependent_heatmap(prediction_df, pairwise_similarity_bins, train_test_similarity_bins, mode='max'):
    count_grid = np.zeros((len(pairwise_similarity_bins)-1, len(train_test_similarity_bins)-1))
    rmse_grid = np.ones((len(pairwise_similarity_bins)-1, len(train_test_similarity_bins)-1)) * -1
    mae_grid = np.ones((len(pairwise_similarity_bins)-1, len(train_test_similarity_bins)-1)) * -1
    bounds = []
    
    tasks = []
    for pairwise_sim_index in range(len(pairwise_similarity_bins)-1):
        low_pw = pairwise_similarity_bins[pairwise_sim_index]
        high_pw = pairwise_similarity_bins[pairwise_sim_index+1]
        if high_pw >= 1.0:
            high_pw = 1.1    # To include 1.0
        bounds.append([])

        for train_test_sim_index in range(len(train_test_similarity_bins)-1):
            low_tt = train_test_similarity_bins[train_test_sim_index]
            high_tt = train_test_similarity_bins[train_test_sim_index+1]
            if high_tt >= 1.0:
                high_tt = 1.1

            bounds[pairwise_sim_index].append(((low_pw, high_pw), (low_tt, high_tt)))

            if mode == 'max':
                relevant_values = prediction_df.loc[(prediction_df['max_max_train_test_sim'] >= low_tt) &
                                                    (prediction_df['max_max_train_test_sim'] < high_tt) &
                                                    (prediction_df['ground_truth_similarity'] >= low_pw) &
                                                    (prediction_df['ground_truth_similarity'] < high_pw)]
            elif mode == 'mean':
                relevant_values = prediction_df.loc[(prediction_df['mean_max_train_test_sim'] >= low_tt) &
                                                    (prediction_df['mean_max_train_test_sim'] < high_tt) &
                                                    (prediction_df['ground_truth_similarity'] >= low_pw) &
                                                    (prediction_df['ground_truth_similarity'] < high_pw)]
            elif mode == 'asms':
                relevant_values = prediction_df.loc[(prediction_df['inchikey1_max_test_sim'] >=low_tt) &
                                                    (prediction_df['inchikey1_max_test_sim'] < high_tt) &
                                                    (prediction_df['ground_truth_similarity'] >= low_pw) &
                                                    (prediction_df['ground_truth_similarity'] < high_pw)]
            else:
                raise ValueError(f"Unknown mode {mode}")

            count_task = delayed(len)(relevant_values)
            rmse_task = delayed(np.sqrt)(delayed(np.nanmean)(np.square(relevant_values['error'].values)))
            mae_task = delayed(np.nanmean)(np.abs(relevant_values['error'].values))

            tasks.append((pairwise_sim_index, train_test_sim_index, count_task, rmse_task, mae_task))

    with ProgressBar(minimum=1.0):
        results = compute(*[task[2:] for task in tasks])    # First two elements aren't computed

    # Reconstruct grids from results
    for idx, (pairwise_sim_index, train_test_sim_index, count_task, rmse_task, mae_task) in enumerate(tasks):
        count_grid[pairwise_sim_index, train_test_sim_index] = results[idx][0]
        rmse_grid[pairwise_sim_index, train_test_sim_index] = results[idx][1]
        mae_grid[pairwise_sim_index, train_test_sim_index] = results[idx][2]

    return {'count': count_grid, 'rmse_grid': rmse_grid, 'mae_grid': mae_grid, 'bounds': bounds}