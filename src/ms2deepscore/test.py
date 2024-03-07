from pathlib import Path
import os
import argparse
import pickle
import matplotlib.pyplot as plt

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

def tanimoto_dependent_losses(scores, scores_ref, ref_score_bins):
    """Compute errors (RMSE and MSE) for different bins of the reference scores (scores_ref).
    
    Parameters
    ----------
    
    scores
        Scores that should be evaluated
    scores_ref
        Reference scores (= ground truth).
    ref_score_bins
        Bins for the refernce score to evaluate the performance of scores.
    """
    bin_content = []
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
        idx = np.where((scores_ref >= low) & (scores_ref < high))
        bin_content.append(idx[0].shape[0])
        maes.append(np.abs(scores_ref[idx] - scores[idx]).mean())
        rmses.append(np.sqrt(np.square(scores_ref[idx] - scores[idx]).mean()))

    return bin_content, bounds, rmses, maes

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument('--test_path', type=str, help='Path to the test data')
    parser.add_argument("--tanimoto_path", type=str, help="Path to the tanimoto scores")
    parser.add_argument("--save_dir", type=str, help="Path to the model")
    args = parser.parse_args()
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    metric_dir = os.path.join(args.save_dir, "metrics")
    if not os.path.isdir(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)
    
    # Check if GPU is available
    print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE", flush=True)
    
    print("Loading model...", flush=True)
    model = load_model(os.path.join(args.save_dir,"model.hdf5"))
    print("\tDone.", flush=True)
    
    print("Loading data...", flush=True)
    print
    spectra_test = pickle.load(open(args.test_path, "rb"))
    tanimoto_df = pd.read_csv(args.tanimoto_path, index_col=0)
    print("\tDone.", flush=True)
    
    # print("Binning Spectra...", flush=True)
    # spectrum_binner = SpectrumBinner(10000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5, allowed_missing_percentage=100.0)
    # binned_spectra_test = spectrum_binner.fit_transform(spectra_test)
    # print("\tDone.", flush=True)
    
    test_inchikeys = np.unique([s.get("inchikey")[:14] for s in spectra_test])
    print(f"Got {len(test_inchikeys)} inchikeys in the test set.")
    
    print("Performing Inference...", flush=True)
    similarity_score = MS2DeepScore(model,)
    similarities_test = similarity_score.matrix(spectra_test, spectra_test, is_symmetric=True)
    print("\tDone.", flush=True)
    
    print("Evaluating...", flush=True)
    inchikey_idx_test = np.zeros(len(spectra_test))

    for i, spec in enumerate(spectra_test):
        inchikey_idx_test[i] = np.where(tanimoto_df.index.values == spec.get("inchikey")[:14])[0]

    inchikey_idx_test = inchikey_idx_test.astype("int")

    scores_ref = tanimoto_df.values[np.ix_(inchikey_idx_test[:], inchikey_idx_test[:])].copy()

    ref_score_bins = np.linspace(0,1.0, 11)
    bin_content, bounds, rmses, maes = tanimoto_dependent_losses(similarities_test,
                                                                scores_ref, ref_score_bins)
    metric_dict = {}
    metric_dict["bin_content"] = bin_content
    metric_dict["bounds"] = bounds
    metric_dict["rmses"] = rmses
    metric_dict["maes"] = maes
    
    # Save to pickle
    metric_path = os.path.join(metric_dir, "metrics.pkl")
    pickle.dump(metric_dict, open(metric_path, "wb"))  

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4, 5), dpi=120)
    
    ax1.plot(np.arange(len(rmses)), rmses, "o:", color="crimson")
    ax1.set_title('RMSE')
    ax1.set_ylabel("RMSE")
    ax1.grid(True)

    ax2.plot(np.arange(len(rmses)), bin_content, "o:", color="teal")
    ax2.set_title('# of spectrum pairs')
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    plt.yscale('log')
    plt.xticks(np.arange(len(rmses)), [f"{a:.1f} to < {b:.1f}" for (a, b) in bounds], fontsize=9, rotation='vertical')
    ax2.grid(True)
    
    # Save figure
    fig_path = os.path.join(metric_dir, "metrics.png")
    plt.savefig(fig_path)
    
    
if __name__ == "__main__":
    main()