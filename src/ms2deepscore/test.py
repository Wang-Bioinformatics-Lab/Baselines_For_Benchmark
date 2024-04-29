from pathlib import Path
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

sys.path.append('../shared')
from utils import train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')
    parser.add_argument("--tanimoto_path", type=str, help="Path to the tanimoto scores")
    parser.add_argument("--train_test_similarities", type=str, help="Path to the train-test tanimoto scores")
    parser.add_argument("--save_dir", type=str, help="Path to the model")
    parser.add_argument("--n_most_recent", type=int, help="Number of most recent models to evaluate", default=1)
    parser.add_argument("--pairs_path", type=str, default=None, help="Path to the pairs file")
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
    
    # List available models
    print("Available models:")
    available_models = [model for model in Path(args.save_dir).rglob("*.hdf5")]
    print(available_models)
    
    print(f"The most recent {args.n_most_recent} models will be evaluated.")
    
    # Sort models by datetime:  dd_mm_yyyy_hh_mm_ss format "%d_%m_%Y_%H_%M_%S"
    available_models = sorted(available_models, key=lambda x: datetime.strptime(x.stem.split('model_')[1], "%d_%m_%Y_%H_%M_%S"), reverse=True)
    
    
    print("Testing the following models:", available_models[:args.n_most_recent], flush=True)
    for model_index, model_name in enumerate(available_models[:args.n_most_recent]):
        print(f"Loading model ({model_index+1}/{min(len(available_models), args.n_most_recent)}: {model_name.stem})...", flush=True)
        model = load_model(model_name)
        print("\tDone.", flush=True)

        spectra_test = pickle.load(open(args.test_path, "rb"))
                
        if args.pairs_path is not None:
            print(f"Filtering spectra based on pairs in {args.pairs_path}")
            print(f"Began with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
            pairs_df = pd.read_csv(args.pairs_path)
            valid_spectra_ids = np.unique(pairs_df[['spectrum_id_1', 'spectrum_id_2']].values)
            spectra_test = [s for s in spectra_test if s.get('spectrum_id') in valid_spectra_ids]
            print(f"Ended with {len(spectra_test)} spectra, which corresponds to {len(np.unique([s.get('inchikey')[:14] for s in spectra_test]))} unique InChI Keys")
        
        tanimoto_df = pd.read_csv(args.tanimoto_path, index_col=0)
        train_test_similarities = pd.read_csv(args.train_test_similarities, index_col=0)
        if args.pairs_path is not None:
            metric_dir = os.path.join(args.save_dir, model_name.stem, 'metrics_filtered_pairs')
        else:
            metric_dir = os.path.join(args.save_dir, model_name.stem, 'metrics')
        if not os.path.isdir(metric_dir):
            os.makedirs(metric_dir, exist_ok=True)
        print(tanimoto_df)
        print("\tDone.", flush=True)
               
        test_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectra_test])
        print(f"Got {len(test_inchikeys)} inchikeys in the test set.")
        
        print("Performing Inference...", flush=True)
        similarity_score = MS2DeepScore(model,)
        df_labels = [s.get("spectrum_id") for s in spectra_test]
        predictions = similarity_score.matrix(spectra_test, spectra_test, is_symmetric=True)
        # Convert to dataframe
        predictions = pd.DataFrame(predictions, index=df_labels, columns=df_labels)
        print("\tDone.", flush=True)
        
        print("Evaluating...", flush=True)
        inchikey_idx_test = np.zeros(len(spectra_test))
        ordered_prediction_index = np.zeros(len(spectra_test))
        
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

        print("Shape of predictions:", predictions.shape)
        predictions = predictions.iloc[ordered_prediction_index, ordered_prediction_index]
        if args.pairs_path is not None:
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
        if args.pairs_path is not None:
            pairs_df = pd.read_csv(args.pairs_path)

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
    plt.savefig(os.path.join(metric_dir, 'heatmap.png'), dpi=300, bbox_inches = "tight")
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
    plt.savefig(os.path.join(metric_dir, 'heatmap_counts.png'), dpi=300, bbox_inches = "tight")
    # Train-Test Similarity Dependent Nan-Counts
    plt.figure()
    plt.title('Train-Test Dependent Nan-Counts')
    plt.imshow(train_test_grid['nan_count'], vmin=0)
    plt.colorbar()
    plt.xticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels, rotation=90)
    plt.yticks(ticks=np.arange(0,len(train_test_grid['rmses'])), labels=tick_labels,)
    plt.savefig(os.path.join(metric_dir, 'heatmap_nan_counts.png'), dpi=300, bbox_inches = "tight")
    
    # MS2DeepScore Tanimoto Dependent Losses Plot
    ref_score_bins = np.linspace(0,1.0, 11)

    tanimoto_dependent_dict = tanimoto_dependent_losses(predictions, ref_score_bins)
    
    rmse = np.sqrt(np.mean(np.square(predictions['error'].values)))
    print("Overall RMSE (from evaluate()):", rmse)
    mae =  np.mean(np.abs(predictions['error'].values))
    
    metric_dict = {}
    metric_dict["bin_content"]      = tanimoto_dependent_dict["bin_content"]
    metric_dict["nan_bin_content"]  = tanimoto_dependent_dict["nan_bin_content"]
    metric_dict["bounds"]           = tanimoto_dependent_dict["bounds"]
    metric_dict["rmses"]            = tanimoto_dependent_dict["rmses"]
    metric_dict["maes"]             = tanimoto_dependent_dict["maes"]
    metric_dict["rmse"] = rmse
    metric_dict["mae"]  = mae
    
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

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)
    
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
    plt.savefig(fig_path, dpi=300, bbox_inches = "tight")
    
    # spec2vec_percentile_plot(predictions, scores_ref, metric_dir)
    
    # Train-Test Similarity Bar Plot
    train_test_sim = train_test_similarity_bar_plot(predictions, train_test_similarities, ref_score_bins)
    bin_content, bounds = train_test_sim['bin_content'], train_test_sim['bounds']
    plt.figure(figsize=(20, 5))
    plt.title("Number of Structures in Similarity Bins (Max Similarity to Train Set)")
    plt.bar(range(len(bin_content)), bin_content, label='Number of Structures')
    plt.xlabel('Similarity Bin (Max Similarity to Train Set)')
    plt.ylabel('Number of Structures')
    plt.xticks(range(len(bin_content)), [f"({bounds[i][0]:.2f}-{bounds[i][1]:.2f})" for i in range(len(bounds))], rotation=45)
    plt.legend()
    plt.savefig(os.path.join(metric_dir, 'train_test_similarity_bar_plot.png'), dpi=300, bbox_inches="tight")
    
    # Score, Tanimoto Scatter Plot
    plt.figure(figsize=(10,10))
    plt.scatter(predictions['score'], predictions['tanimoto'], alpha=0.2)
    plt.xlabel('Predicted Spectral Similarity Score')
    plt.ylabel('Tanimoto Score')
    # Make square
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Predicted vs Reference Spectral Similarity Scores')
    plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference.png'), dpi=300)
    
    # Score, Tanimoto Scatter Plot (Hexbin)
    plt.figure(figsize=(10,10))
    hb = plt.hexbin(predictions['score'], predictions['tanimoto'], gridsize=50, cmap='inferno')
    cb = plt.colorbar(hb)
    cb.set_label('counts')
    plt.xlabel('Predicted Spectral Similarity Score')
    plt.ylabel('Tanimoto Score')
    # Make square
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('Predicted vs Reference Spectral Similarity Scores (Hexbin)')
    plt.savefig(os.path.join(metric_dir, 'predicted_vs_reference_hexbin.png'), dpi=300)

    
if __name__ == "__main__":
    main()