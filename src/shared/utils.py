import pandas as pd
import numpy as np
from matchms.filtering import add_fingerprint
from matchms.similarity import FingerprintSimilarity


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

def train_test_similarity_dependent_losses(prediction_df, train_test_similarity, ref_score_bins, mode='max'):
    # prediction_df is a pandas dataframe with columns spectruim_id_1, spectrum_id_2, inchikey_1, inchikey_2, score, error
    bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    # ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    
    assert mode in ['max', 'mean']
    all_predicted_inchikeys = prediction_df['inchikey_1'].unique()  # We assume square matrix
        
    # Only include if the keys are in the error matrix
    train_test_similarity = train_test_similarity.loc[:, all_predicted_inchikeys]
    
    if mode == 'max':
        train_test_similarity = train_test_similarity.max(axis=0)
    elif mode == 'mean':
        train_test_similarity = train_test_similarity.mean(axis=0)
    print(train_test_similarity)
    print(train_test_similarity.max())
    
    for inchikey14 in all_predicted_inchikeys:
        assert inchikey14 in train_test_similarity.index
    
    for i in range(len(ref_scores_bins_inclusive)-1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i+1]
        bounds.append((low, high))
        relevant_keys = train_test_similarity.loc[(train_test_similarity > low) & (train_test_similarity <= high)].index
        
        relevant_values = prediction_df.loc[(prediction_df['inchikey_1'].isin(relevant_keys)), :]['error']
        
        bin_content.append(relevant_keys.shape[0])
        
        maes.append(np.nanmean(np.abs(relevant_values).values))
        rmses.append(np.sqrt(np.nanmean(np.square(relevant_values).values)))
        rmse_errors = np.sqrt(np.square(relevant_values).values)
        mae_errors = np.abs(relevant_values).values
        
    return {'bin_content':bin_content, 'bounds':bounds, 'rmses':rmses, 'maes':maes}

def train_test_similarity_heatmap(prediction_df, train_test_similarity, ref_score_bins):
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
    assert np.allclose(rmse_grid, rmse_grid.T, equal_nan=True), f"RMSE is not symmetric: {rmse_grid}"
    assert np.allclose(mae_grid, mae_grid.T, equal_nan=True), f"MAE is not symmetric: {mae_grid}"
    
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
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).
    
    Parameters
    ----------
    
    prediction_df
        
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    
    bin_content = []
    nan_bin_content = []
    rmses = []
    maes = []
    bounds = []
    ref_scores_bins_inclusive = ref_score_bins.copy()
    ref_scores_bins_inclusive[0] = -np.inf
    ref_scores_bins_inclusive[-1] = np.inf
    for i in range(len(ref_scores_bins_inclusive)-1):
        low = ref_scores_bins_inclusive[i]
        high = ref_scores_bins_inclusive[i+1]
        bounds.append((low, high))
        
        relevant_values = prediction_df.loc[(prediction_df.tanimoto >= low) & (prediction_df.tanimoto < high)]

        nan_bin_content.append(relevant_values.score.isna().sum())
        bin_content.append(relevant_values.score.notna().sum())
        maes.append(np.nanmean(relevant_values.error.values))
        rmses.append(np.sqrt(np.nanmean(np.square(relevant_values.error.values))))

    return {'bin_content': bin_content, 'bounds':bounds, 'rmses': rmses, 'maes':maes, 'nan_bin_content':nan_bin_content}