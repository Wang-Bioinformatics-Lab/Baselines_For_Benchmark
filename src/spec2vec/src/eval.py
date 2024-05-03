import os
import gc
import sys
import argparse
import logging
from datetime import datetime
import matchms
import numpy as np
import pandas as pd
from matchms import calculate_scores
from matchms.similarity import ModifiedCosine
import pickle
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from matchms import calculate_scores
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.append('../../shared')
from utils import   get_structural_similarity_matrix, \
                    train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot
                    
                    
grid_fig_size = (10,10)
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

                    
def get_spectra(spectra_path:str)->list:
    spectra = pickle.load(open(spectra_path, 'rb'))
    return spectra

def generate_structure_mapping(spectra_path:str)->dict:
    spectra = get_spectra(spectra_path)
    structure_mapping = {spectrum.get("spectrum_id"): spectrum.get("inchikey")[:14] for spectrum in spectra}
    return structure_mapping

def spec2vec_percentile_plot(scoring_df:pd.DataFrame,metric_dir)->None:
    """This functions seeks to recreate Fig 3B from the Spec2Vec paper. The general idea is to take the top 0.10% of predicted scores and plot
    the refence for those pairs. This will allow us to see how well the model is doing on the most similar pairs. 
    
    Parameters:
    scoring_df (pd.DataFrame): A datafrane if scoring_df, and the reference values. The columns should be ['spectrum_id_1', 'spectrum_id_2', 'score', 'tanimoto']
    metric_dir (str): The directory to save the plot
    
    Returns:
    None
    """
    
    plt.figure()
    # Remove identical spectrum ids to avoid self-comparison
    scoring_df = scoring_df.loc[scoring_df['spectrum_id_1'] != scoring_df['spectrum_id_2']]
    # Calculate top N before removing NaNs
    top_percentile = 0.001
    top_n = int(len(scoring_df) * top_percentile)
    
    #### spec2vec ####
    # Remove NaNs
    scoring_df_copy = scoring_df[~scoring_df['score'].isna()].copy(deep=True)
    
    top_n_scores = scoring_df_copy['score'].nlargest(top_n).sort_values(ascending=False)
    top_n_score_idx = top_n_scores.index
    top_n_scores = top_n_scores.values
    
    top_n_reference = scoring_df_copy.loc[top_n_score_idx, 'tanimoto'].values
    print(top_n_reference)
    
    # Bin the top_n_scores into 10 bins
    hist, bin_edges = np.histogram(top_n_scores, bins=10)
    # Flip hist, bin edges so the highest similarity is on the left
    hist = np.flip(hist)
    bin_edges = np.flip(bin_edges)
    
    # Using the bins generated, get the average reference score for each bin
    avg_reference = np.zeros(10)
    for i in range(10):
        bin_indices = np.where((top_n_scores < bin_edges[i]) & (top_n_scores >= bin_edges[i+1]))[0]
        avg_reference[i] = np.mean(top_n_reference[bin_indices])
        
    # print("BIN EDGES",bin_edges)
    # print("AVG REFERENCE", avg_reference)

    # Generate the correct bins for plotting
    bins_for_bar = bin_edges[:-1]

    # Plot the average reference score for each bin
    plt.plot(np.arange(len(bins_for_bar)), avg_reference, label='Spec2Vec')
    
    #### Modified Cosine ####
    # Remove NaNs
    scoring_df_copy = scoring_df[~scoring_df['modified_cosine'].isna()].copy(deep=True)
    
    top_n_scores = scoring_df_copy['modified_cosine'].nlargest(top_n).sort_values(ascending=False)
    top_n_score_idx = top_n_scores.index
    top_n_scores = top_n_scores.values
    
    top_n_reference = scoring_df_copy.loc[top_n_score_idx, 'tanimoto'].values
    # print(top_n_reference)
    
    # Bin the top_n_scores into 10 bins
    hist, bin_edges = np.histogram(top_n_scores, bins=10)
    # Flip hist, bin edges so the highest similarity is on the left
    hist = np.flip(hist)
    bin_edges = np.flip(bin_edges)
    
    # Using the bins generated, get the average reference score for each bin
    avg_reference = np.zeros(10)
    for i in range(10):
        bin_indices = np.where((top_n_scores < bin_edges[i]) & (top_n_scores >= bin_edges[i+1]))[0]
        avg_reference[i] = np.mean(top_n_reference[bin_indices])
        
    # print("BIN EDGES",bin_edges)
    # print("AVG REFERENCE", avg_reference)
    
    # Generate the correct bins for plotting
    bins_for_bar = bin_edges[:-1]
    
    # Plot the average reference score for each bin
    plt.plot(np.arange(len(bins_for_bar)), avg_reference, color='orange', label='Modified Cosine')
    
    #### Plot the theoretical maximum ####
    top_n_struc_sim = scoring_df_copy['tanimoto'].nlargest(top_n).sort_values(ascending=False).values
    # print("top_n_struc_sim:", top_n_struc_sim)
    
    # Bin the top_n_scores into 10 bins
    hist, bin_edges = np.histogram(top_n_struc_sim, bins=10)
    # Flip hist, bin edges so the highest similarity is on the left
    hist = np.flip(hist)
    bin_edges = np.flip(bin_edges)
    
    # Using the bins generated, get the average reference score for each bin
    avg_reference = np.zeros(10)
    for i in range(10):
        bin_indices = np.where((top_n_struc_sim < bin_edges[i]) & (top_n_struc_sim >= bin_edges[i+1]))[0]
        avg_reference[i] = np.mean(top_n_struc_sim[bin_indices])

    # Generate the correct bins for plotting
    bins_for_bar = bin_edges[:-1]
    
    # print("BIN EDGES",bin_edges)
    # print("AVG REFERENCE", avg_reference)

    # Plot the average reference score for each bin
    plt.plot(np.arange(len(bins_for_bar)), avg_reference, color='grey', label='Theoretical Maximum')
    
    plt.title(f'Average Tanimoto Scores for High Spectra Similarity Scores')
    # Label x ticks with percentile
    plt.xticks(np.arange(len(bins_for_bar)), [f'{x*100:.2f}%' for x in np.linspace(0, top_percentile, len(bins_for_bar))])
    plt.xlim(0, len(bins_for_bar))
    plt.xlabel('Top Percentile of Spectral Similarity Score')
    plt.ylabel('Tanimoto Score')
    plt.legend()
    plt.savefig(os.path.join(metric_dir, 'top_percentile.png'), dpi=300, bbox_inches="tight")
        

def evaluate(predictions:pd.DataFrame, split_type:str, model:str, data_path:str, metric_dir:str, pairs_path:str=None):    
    
    print("Loading Data", flush=True)
    spectra_path = f'{data_path}/data/ALL_GNPS_positive_test_split.pickle'
    
    train_test_similarities = pd.read_csv(f'{data_path}/train_test_tanimoto_df.csv', index_col=0)
    
    if model != 'modified_cosine':
            
        tanimoto_df = pd.read_csv(f'{data_path}/test_tanimoto_df.csv', index_col=0)

        print(f"Generating structure mapping from {spectra_path}")
        
        spectra_test = list(get_spectra(spectra_path))
        
        structure_mapping = {spectrum.get("spectrum_id"): spectrum.get("inchikey")[:14] for spectrum in spectra_test}
        
        if pairs_path is not None:
            print(f"Filtering spectra based on pairs in {pairs_path}")
            print(f"Began with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
            pairs_df = pd.read_csv(pairs_path)
            valid_spectra_ids = np.unique(pairs_df[['spectrum_id_1', 'spectrum_id_2']].values)
            spectra_test = [s for s in spectra_test if s.get('spectrum_id') in valid_spectra_ids]
            print(f"Ended with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
        
        inchikey_idx_test = np.zeros(len(spectra_test))
        ordered_prediction_index = np.zeros(len(spectra_test)) # len(spectra_test) because all predictions should correspond to a spectra, allows for pairs filtering
        
        # Ensure that the ground truth order is correct
        for i, spec in enumerate(spectra_test):
            try:
                inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0].item()
            except ValueError as value_error:
                if not spec.get("inchikey")[:14] in tanimoto_df.index.values:
                    raise ValueError (f"InChI Key '{spec.get('inchikey')[:14]}' is not found in the provided strucutral similarity matrix.")
                raise value_error
            try:
                ordered_prediction_index[i] = np.where(predictions.index.values == spec.get("spectrum_id"))[0].item()
            except ValueError as value_error:
                if not spec.get("spectrum_id") in predictions.index.values:
                    raise ValueError (f"spectrum_id' {spec.get('spectrum_id')}' is not found in the provided predictions.")
                raise value_error

        # Reorder both preds and ground truth based on test spectra order
        print("Shape of tanimoto_df:", tanimoto_df.shape)
        inchikey_idx_test = inchikey_idx_test.astype("int")
        tanimoto_df = tanimoto_df.iloc[inchikey_idx_test, inchikey_idx_test]
        if pairs_path is not None:
            print("Shape of tanimoto_df after filtering:", tanimoto_df.shape)
        print("Shape of predictions:", predictions.shape)
        predictions = predictions.iloc[ordered_prediction_index, ordered_prediction_index]
        if pairs_path is not None:
            print("Shape of predictions after filtering:", predictions.shape)
        
        assert len(predictions.index.values) == len(predictions.index.unique())
        assert all(predictions.index.values == predictions.columns.values)

        
        # Flatten the prediction
        # columns should be ['spectrum_id_1', 'spectrum_id_2', 'score']
        predictions = predictions.stack().reset_index()
        predictions.columns = ['spectrum_id_1', 'spectrum_id_2', 'score']

        # Only get valid pairs
        if pairs_path is not None:
            pairs_df = pd.read_csv(pairs_path)

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
            
        # Add the ground_true value to the predictions
        predictions['error'] = (predictions['tanimoto'] - predictions['score']).abs()
        
        predictions = predictions.copy()

    elif model == 'modified_cosine':
        predictions = pd.read_csv(pairs_path)
        
        # Create sets of the pairs
        pair_set = set(map(tuple, predictions[['spectrum_id_1', 'spectrum_id_2']].values))
        reverse_pair_set = set(map(tuple, predictions[['spectrum_id_2', 'spectrum_id_1']].values))

        # Check if predictions is symmetric
        if pair_set != reverse_pair_set:
            print("Pairs file is not symmetric. Making it symmetric.")
            predictions = pd.concat([predictions, predictions.rename(columns={'spectrum_id_1':'spectrum_id_2',
                                                                    'spectrum_id_2':'spectrum_id_1',
                                                                    'inchikey_1':'inchikey_2',
                                                                    'inchikey_2':'inchikey_1'})], axis=0)
            print("Pairs are now symmetric.")
        
        predictions['score'] = predictions['modified_cosine']
        predictions['tanimoto'] = predictions['structural_similarity']
        predictions.drop(columns=['modified_cosine', 'structural_similarity'], inplace=True)
        predictions['error'] = (predictions['tanimoto'] - predictions['score']).abs()
        
        # set score, modified_cosine to nan where matched_peaks < 6
        predictions.loc[predictions['matched_peaks'] < 6, 'score'] = np.nan
        predictions.loc[predictions['matched_peaks'] < 6, 'error'] = np.nan
        
        print(f"Got columns {predictions.columns} for modified_cosine")
        
    else:
        raise ValueError(f"Model {model} not recognized")
    
    overall_rmse = np.sqrt(np.nanmean(np.square(predictions['error'].values)))
    print("Overall RMSE (from evaluate()):", overall_rmse)
    overall_mae =  np.nanmean(np.abs(predictions['error'].values))
    
    ref_score_bins = np.linspace(0,1.0, 21)
    # Train-Test Similarity Dependent Losses
    similarity_dependent_metrics_mean = train_test_similarity_dependent_losses(predictions, train_test_similarities, ref_score_bins, mode='mean')
    similarity_dependent_metrics_max = train_test_similarity_dependent_losses(predictions, train_test_similarities, ref_score_bins, mode='max')
    train_test_grid = train_test_similarity_heatmap(predictions, train_test_similarities, ref_score_bins)
    # Returns {'bin_content':bin_content_grid, 'bounds':bound_grid, 'rmses':rmse_grid, 'maes':mae_grid}
    
    plt.figure()
    plt.title('Train-Test Dependent RMSE')
    plt.imshow(train_test_grid['rmses'], vmin=0)
    plt.colorbar()
    
    tick_labels = [f'({x[1][0]:.2f}, {x[1][1]:.2f})' for x in train_test_grid['bounds'][0]]
    
    plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
    plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
    plt.savefig(os.path.join(metric_dir, 'heatmap.png'))
    # Train-Test Similarity Dependent Counts
    plt.figure()
    plt.title('Train-Test Dependent Counts')
    plt.imshow(train_test_grid['bin_content'], vmin=0)
    plt.colorbar()
    plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
    plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
    # If the number of samples in a bin is less than 30, put text in the bin
    for i in range(len(train_test_grid['rmses'])):
        for j in range(len(train_test_grid['rmses'])):
            if train_test_grid['bin_content'][i,j] < 30 and not pd.isna(train_test_grid['bin_content'][i,j]):
                plt.text(j, i, f'{train_test_grid["bin_content"][i,j]:.0f}', ha="center", va="center", color="white")
    plt.savefig(os.path.join(metric_dir, 'heatmap_counts.png'))
    # Train-Test Similarity Dependent Nan-Counts
    plt.figure()
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
    
    # Save to pickle
    metric_path = os.path.join(metric_dir, "metrics.pkl")
    print(f"Saving metrics to {metric_path}")
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
    plt.xticks(np.arange(len(metric_dict["rmses"])), [f"{a:.1f} to < {b:.1f}" for (a, b) in metric_dict["bounds"]], fontsize=9, rotation='vertical')
    ax2.grid(True)
    
    # Save figure
    fig_path = os.path.join(metric_dir, "metrics.png")
    plt.savefig(fig_path)
    
    # Only do this for spec2vec
    if model == 'spec2vec':
        spec2vec_percentile_plot(predictions, metric_dir)
    
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
    plt.savefig(os.path.join(metric_dir, 'train_test_similarity_bar_plot.png'), dpi=300, bbox_inches="tight")
    
    # Score, Tanimoto Scatter Plot
    plt.figure(figsize=grid_fig_size)
    plt.scatter(predictions['score'], predictions['tanimoto'], alpha=0.2)
    plt.xlabel('Predicted Spectral Similarity Score')
    plt.ylabel('Tanimoto Score')
    # Make square
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Predicted vs Reference Spectral Similarity Scores')
    plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference.png'), dpi=300)
    
    # Score, Tanimoto Scatter Plot
    plt.figure(figsize=grid_fig_size)
    plt.scatter(predictions['score'], predictions['tanimoto'], alpha=0.2)
    # Show R Squared
    r2 = np.corrcoef(predictions['score'][predictions['score'].notna()], predictions['tanimoto'][predictions['score'].notna()],)[0,1]**2
    # Plot y=x line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('Predicted Spectral Similarity Score')
    plt.ylabel('Tanimoto Score')
    # Make square
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f'Predicted vs Reference Spectral Similarity Scores \n(R squared = {r2:.2f})')
    plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference.png'))
    
def eval_ms2deepscore(split_type:str, model:str, data_path:str, metric_dir:str, pairs_path:str=None):
    if not os.path.isdir(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)
    
    spectra_path = f'{data_path}/data/ALL_GNPS_positive_test_split.pickle'
    spectra_test = list(get_spectra(spectra_path))
    train_test_similarities = pd.read_csv(f'{data_path}/train_test_tanimoto_df.csv', index_col=0)
    
    print("Loading MS2DeepScore model.", flush=True)
    model = load_model(model_name)
    print("Model loaded.", flush=True)
    
    print("Calculating predictions.", flush=True)
    similarity_score = MS2DeepScore(model,)
    predictions = similarity_score.matrix(spectra_test, spectra_test, is_symmetric=True)
    print("Done calculating predictions.", flush=True)
    
    print("Evaluating...", flush=True)
    
    df_labels = [s.get("spectrum_id") for s in spectra_test]
    ms2deepscore_predictions = pd.DataFrame(predictions, index=df_labels, columns=df_labels)
    evaluate(ms2deepscore_predictions, split_type, model, data_path, metric_dir, pairs_path=pairs_path)

def eval_spec2vec(split_type:str, model:str, data_path:str, metric_dir:str, pairs_path:str=None):
    if not os.path.isdir(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)
    spectra_path = glob(f'{data_path}/data/ALL_GNPS_positive_test_split*.pickle')[0]
    tanimoto_df = pd.read_csv(f'{data_path}/test_tanimoto_df.csv', index_col=0)
    train_test_similarities = pd.read_csv(f'{data_path}/train_test_tanimoto_df.csv', index_col=0)
    
    # Load vectors
    print("Retrieving vectors from", f'./{split_type}/{model}/embedded_spectra_test/')
    vector_paths = glob(f'./{split_type}/{model}/embedded_spectra_test/**/*.npy')
    print(f"Found {len(vector_paths)} vectors")
    ids = [Path(vector_path).stem for vector_path in vector_paths]
    vectors = np.array([np.load(vector_path) for vector_path in vector_paths])
    
    print("Vector shape:", vectors.shape)
    # Count the number of all nan vectors
    print(f"Number of nan vectors: {np.isnan(vectors).all(axis=1).sum()}")
    
    df_labels = ids
       
    # Calculate scores
    def _parallel_cosine(lst):
        # If there is a nan in either vector, return nan
        output_lst = []
        for i, j in lst:
            if np.isnan(vectors[i]).any() or np.isnan(vectors[j]).any():
                output_lst.append((i, j, np.nan))
            else:
                sim = cosine_similarity(vectors[i].reshape(1, -1) , vectors[j].reshape(1, -1) , dense_output=True).item()
                # Map sim to range [0,1]
                output_lst.append((i, j, (sim + 1) / 2))
        return output_lst
    
    if not os.path.exists(os.path.join(metric_dir, 'spec2vec_predictions.csv')):
        # Chunk the calculation to reduce overhead cost
        cosine_pairs = [(i, j) for i in range(len(vectors)) for j in range(i, len(vectors))]
        chunked_pairs = np.array_split(cosine_pairs, len(cosine_pairs)//1000)
        
        result = Parallel(n_jobs=-4)(delayed(_parallel_cosine)(lst) for lst in tqdm(chunked_pairs))
        # Flatten
        result = [item for sublist in result for item in sublist]
        predictions = np.zeros((len(vectors), len(vectors)))
        for i, j, score in result:
            predictions[i, j] = score
            predictions[j, i] = score
        
        # Transform to DataFrame
        spec2vec_predictions = pd.DataFrame(predictions, index=df_labels, columns=df_labels)
        spec2vec_predictions.to_csv(os.path.join(metric_dir, 'spec2vec_predictions.csv'))
    else: 
        spec2vec_predictions = pd.read_csv(os.path.join(metric_dir, 'spec2vec_predictions.csv'), index_col=0)
    
    evaluate(spec2vec_predictions, split_type, model, data_path, metric_dir, pairs_path=pairs_path)


def eval_modified_cosine(split_type:str, data_path:str, metric_dir:str, pairs_path:str):
    if not os.path.isdir(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)
    spectra_path = f'{data_path}/data/ALL_GNPS_positive_test_split.pickle'
    spectra_test = list(get_spectra(spectra_path))
    modified_cosine = ModifiedCosine(tolerance=0.005)

    assert pairs_path is not None

    evaluate(None, split_type, 'modified_cosine', data_path, metric_dir, pairs_path=pairs_path)

def main():
    parser = argparse.ArgumentParser(description='Evaluate similarity scores')
    parser.add_argument('--method', type=str, help='Method to evaluate')
    parser.add_argument('--data', type=str, help='Path to data')
    parser.add_argument('--split_type', type=str, help='Type of split')
    parser.add_argument('--model', type=str, default=None, help='Path to model')
    parser.add_argument('--n_most_recent', type=int, default=None, help='Number of most recent models to evaluate')
    parser.add_argument('--debug', type=str, help='Debug mode', default='False')
    parser.add_argument('--pairs_path', type=str, default=None,
                        help='Path to a csv file of pairs used in evaluation. Must contain columns "spectrum_id_1" and "spectrum_id_2"')
        
    method_dir_dict ={
        'spec2vec': '/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/spec2vec/src',
        'ms2deepscore': '/data/nas-gpu/SourceCode/michael_s/Baselines_For_Benchmark/src/ms2deepscore/',
    }
    method_model_ext_dict = {
        'spec2vec': '.model',
        'ms2deepscore': '.hdf5',
    }
        
    args = parser.parse_args()

    if args.method.lower() == 'ms2deepscore':
        raise NotImplementedError("MS2DeepScore evaluation is not yet implemented in this script.")

    if args.method == 'modified_cosine':
        if args.model is not None:
            raise ValueError("Model path can not be provided for modified cosine")
        if args.pairs_path is not None:
            metric_dir = f'./{args.split_type}/{args.method}/metrics_filtered_pairs/'
        else:
            metric_dir = f'./{args.split_type}/{args.method}/metrics/'
        eval_modified_cosine(args.split_type, args.data, metric_dir, args.pairs_path)
    elif args.method.lower() in ['spec2vec', 'ms2deepscore']:
        if args.split_type is None:
            raise ValueError("split_type must be provided for spec2vec")
        if args.model and args.n_most_recent:
            raise ValueError("Only one of model and n_most_recent can be provided")
        if args.model is None:
            print("Model not provided, using the most recent model")
        
        if args.n_most_recent:
            print(f"The most recent {args.n_most_recent} models will be evaluated.")
            # Sort models by datetime:  dd_mm_yyyy_hh_mm_ss format "%d_%m_%Y_%H_%M_%S"

            
            model_glob_1 = f'{method_dir_dict[args.method]}/{args.split_type}/**/*{method_model_ext_dict[args.method]}'
            model_glob_2 = f'{method_dir_dict[args.method]}/{args.split_type}/*{method_model_ext_dict[args.method]}'
            
            print(model_glob_1)
            
            available_models = [model for model in list(glob(model_glob_1)) + list(glob(model_glob_2))]
            available_models = sorted(available_models, key=lambda x: datetime.strptime(x.split('/')[-2], "%d_%m_%Y_%H_%M_%S"), reverse=True)
            print(available_models)
            available_models = available_models[:args.n_most_recent]
        elif args.model:
            available_models = [model for model in glob(f'{method_dir_dict[args.method]}/{args.split_type}/{args.model}/*.model')]
            assert len(available_models) == 1, f"Expected 1 model, got {len(available_models)}"
        else:
            raise ValueError("Either model or n_most_recent must be provided")
        
        for model_index, model_name in enumerate(available_models):
            print(f"Running model ({model_index+1}/{len(available_models)})...", flush=True)
            print('model_name', model_name)
            model_time = datetime.strptime(model_name.split('/')[-2], "%d_%m_%Y_%H_%M_%S")
            if args.pairs_path is not None:
                metric_dir = f'./{args.split_type}/{model_name.rsplit("/",2)[-2]}/metrics_filted_pairs/'
            else:
                metric_dir = f'./{args.split_type}/{model_name.rsplit("/",2)[-2]}/metrics/'
            eval_spec2vec(args.split_type, model_time.strftime("%d_%m_%Y_%H_%M_%S"), args.data, metric_dir, args.pairs_path)

if __name__ == "__main__":
    main()