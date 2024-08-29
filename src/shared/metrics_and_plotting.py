import argparse
from pathlib import Path
import os

import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.distributed import LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc

TT_SIM_BINS = np.linspace(0.4,1.0, 13)
PW_SIM_BINS = np.linspace(0.,1.0, 11)

from utils import train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot, \
                    fixed_tanimoto_train_test_similarity_dependent, \
                    pairwise_train_test_dependent_heatmap, \
                    roc_curve

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

GRID_FIG_SIZE = (10,7)

def main():
    parser = argparse.ArgumentParser(description='Test MS2DeepScore on the original data')
    parser.add_argument("--prediction_path", type=str, help="Path to parquet file containing predictions")
    parser.add_argument("--save_dir", type=str, help="Path to save the output")
    parser.add_argument("--save_dir_insert", type=str, help="Appended to save dir, to help organize test sets if needed", default="")
    parser.add_argument("--n_jobs", type=int, help="Number of jobs to run in parallel", default=1)
    args = parser.parse_args()

    prediction_path = Path(args.prediction_path)

    metric_dir = os.path.join(args.save_dir, args.save_dir_insert)

    print("Saving Metrics to:", metric_dir)
    if not os.path.isdir(metric_dir):
        os.makedirs(metric_dir, exist_ok=True)

    # Initialize Dask Cluster
    cluster = LocalCluster(n_workers=int(args.n_jobs/2), threads_per_worker=2)
    client = cluster.get_client()

    presampled_pairs = dd.read_parquet(prediction_path)

    print(presampled_pairs.head(10))
    
    overall_rmse = da.sqrt(da.mean(da.square(presampled_pairs['error']))).compute()
    print("Overall RMSE (from evaluate()):", overall_rmse)
    overall_mae =  da.mean(da.abs(presampled_pairs['error'])).compute()
    print("Overall MAE (from evaluate()):", overall_mae)

    # Nan Safe
    print("Overall NAN RMSE (from evaluate()):", da.sqrt(da.nanmean(da.square(presampled_pairs['error'].values))).compute())
    print("Overall NAN MAE (from evaluate()):", da.nanmean(da.abs(presampled_pairs['error'])).compute())
    print("Nan Count:", presampled_pairs['error'].isna().sum().compute())

    # Calculate ROC Curve
    # print("Creating ROC Curve", flush=True)
    # roc_metrics = roc_curve(presampled_pairs, np.linspace(0,1.0, 21))
    # roc_path = os.path.join(metric_dir, "roc_metrics.pkl")
    # pickle.dump(roc_metrics, open(roc_path, "wb"))
    # plt.figure(figsize=(6, 6))
    # plt.plot(roc_metrics["fpr"], roc_metrics["tpr"], "o-")
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.grid(True)

    # Train-Test Similarity Dependent Losses Aggregated (rmse)
    print("Creating Train-Test Similarity Dependent Losses Aggregated Plot", flush=True)
    similarity_dependent_metrics_mean = train_test_similarity_dependent_losses(presampled_pairs, TT_SIM_BINS, mode='mean')
    plt.figure(figsize=(12, 9))
    plt.bar(np.arange(len(similarity_dependent_metrics_mean["rmses"]),), similarity_dependent_metrics_mean["rmses"],)
    # Add labels on top of bars
    for i, v in enumerate(similarity_dependent_metrics_mean["rmses"]):
        plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=14)
    plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_rmse:.2f}')
    plt.xlabel("Mean(Max(Test-Train Stuctural Similarity))")
    plt.ylabel("RMSE")
    plt.xticks(np.arange(len(TT_SIM_BINS[1:])), [f"{x:.1f}" for x in TT_SIM_BINS[1:]], rotation='vertical')
    plt.grid(True)
    plt.savefig(os.path.join(metric_dir, 'train_test_rmse_mean.png'))
    train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_mean.pkl")
    pickle.dump(similarity_dependent_metrics_mean, open(train_test_metric_path, "wb"))

    # Train-Test Similarity Dependent Losses Aggregated (mae)
    plt.figure(figsize=(12, 9))
    plt.bar(np.arange(len(similarity_dependent_metrics_mean["maes"]),), similarity_dependent_metrics_mean["maes"],)
    # Add labels on top of bars
    for i, v in enumerate(similarity_dependent_metrics_mean["maes"]):
        plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=14)
    plt.title(f'Train-Test Dependent MAE\nAverage MAE: {overall_mae:.2f}')
    plt.xlabel("Mean(Max(Test-Train Stuctural Similarity))")
    plt.ylabel("MAE")
    plt.xticks(np.arange(len(TT_SIM_BINS[1:])), [f"{x:.1f}" for x in TT_SIM_BINS[1:]], rotation='vertical')
    plt.grid(True)
    plt.savefig(os.path.join(metric_dir, 'train_test_mae_mean.png'))
    train_test_metric_path = os.path.join(metric_dir, "train_test_metrics_mean.pkl")
    pickle.dump(similarity_dependent_metrics_mean, open(train_test_metric_path, "wb"))
    del similarity_dependent_metrics_mean
    gc.collect()
    
    # Tanimoto Dependent Losses Plot (RMSE)
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

    print("Creating Tanimoto Dependent Losses Plot", flush=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    
    ax1.plot(np.arange(len(metric_dict["rmses"])), metric_dict["rmses"], "o:", color="crimson")
    ax1.set_ylabel("RMSE")
    ax1.grid(True)

    ax2.plot(np.arange(len(metric_dict["rmses"])), metric_dict["bin_content"], "o:", color="teal")
    ax2.plot(np.arange(len(metric_dict["rmses"])), metric_dict["nan_bin_content"], "o:", color="grey")
    if sum(metric_dict["nan_bin_content"]) > 0:
        ax2.legend(["# of valid spectrum pairs", "# of nan spectrum pairs"])
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    plt.yscale('log')
    plt.xticks(np.arange(len(PW_SIM_BINS[1:])), [f"{x:.1f}" for x in PW_SIM_BINS[1:]], rotation='vertical')
    ax2.grid(True)
    
    # Save figure
    fig_path = os.path.join(metric_dir, "rmse_metrics.png")
    plt.savefig(fig_path)

    # Tanimoto Dependent Losses Plot (MAE)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    ax1.plot(np.arange(len(metric_dict["maes"])), metric_dict["maes"], "o:", color="crimson")
    ax1.set_ylabel("MAE")
    ax1.grid(True)

    ax2.plot(np.arange(len(metric_dict["maes"])), metric_dict["bin_content"], "o:", color="teal")
    ax2.plot(np.arange(len(metric_dict["maes"])), metric_dict["nan_bin_content"], "o:", color="grey")
    if sum(metric_dict["nan_bin_content"]) > 0:
        ax2.legend(["# of valid spectrum pairs", "# of nan spectrum pairs"])
    ax2.set_ylabel("# of spectrum pairs")
    ax2.set_xlabel("Tanimoto score bin")
    plt.yscale('log')
    plt.xticks(np.arange(len(PW_SIM_BINS[1:])), [f"{x:.1f}" for x in PW_SIM_BINS[1:]], rotation='vertical')
    ax2.grid(True)

    fig_path = os.path.join(metric_dir, "mae_metrics.png")
    plt.savefig(fig_path)

    # RMSE and MAE Only (No Counts)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(len(metric_dict["rmses"]),), metric_dict["rmses"], "o:", color="crimson")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Tanimoto score bin")
    ax.grid(True)
    plt.xticks(np.arange(len(PW_SIM_BINS[1:])), [f"{x:.1f}" for x in PW_SIM_BINS[1:]], rotation='vertical')
    fig_path = os.path.join(metric_dir, "rmse_metrics_no_counts.png")
    plt.savefig(fig_path)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.arange(len(metric_dict["maes"]),), metric_dict["maes"], "o:", color="crimson")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Tanimoto score bin")
    ax.grid(True)
    plt.xticks(np.arange(len(PW_SIM_BINS[1:])), [f"{x:.1f}" for x in PW_SIM_BINS[1:]], rotation='vertical')
    fig_path = os.path.join(metric_dir, "mae_metrics_no_counts.png")
    plt.savefig(fig_path)

    # Pairwise Similarity, Train-Test Distance Heatmap (RMSE)
    print("Creating Pairwise & Train-Test Similarity Heatmap", flush=True)
    pw_tt_metrics = pairwise_train_test_dependent_heatmap(presampled_pairs, PW_SIM_BINS, TT_SIM_BINS, mode='mean')
    pw_tt_metric_path = os.path.join(metric_dir, "pairwise_train_test_metrics_mean.pkl")
    pickle.dump(pw_tt_metrics, open(pw_tt_metric_path, "wb"))
    # Set any -1 values to nan
    pw_tt_metrics['rmse_grid'] = np.where(pw_tt_metrics['rmse_grid'] == -1, np.nan, pw_tt_metrics['rmse_grid'])
    plt.figure(figsize=GRID_FIG_SIZE)
    # pw_tt_metrics['rmse_grid'] first index modulates the pairwise similarity, second index modulates the train-test similarity
    # imshow is transposed so that the pairwise similarity is on the y-axis
    plt.imshow(pw_tt_metrics['rmse_grid'], origin='lower')
    plt.colorbar()
    plt.title('Pairwise & Train-Test Dependent RMSE')
    plt.ylabel('Pairwise Structural Similarity')
    plt.xlabel('Max Test-Train Structural Similarity')
    plt.xticks(np.arange(len(TT_SIM_BINS))-0.5, [f"{x:.2f}" for x in TT_SIM_BINS], rotation=90)
    plt.yticks(np.arange(len(PW_SIM_BINS))-0.5, [f"{x:.1f}" for x in PW_SIM_BINS])
    plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap_rmse.png'), bbox_inches="tight")

    # Pairwise Similarity, Train-Test Distance Heatmap (MAE)
    plt.figure(figsize=GRID_FIG_SIZE)
    # Set any -1 values to nan
    pw_tt_metrics['mae_grid'] = np.where(pw_tt_metrics['mae_grid'] == -1, np.nan, pw_tt_metrics['mae_grid'])
    plt.imshow(pw_tt_metrics['mae_grid'], origin='lower')
    plt.colorbar()
    plt.title('Pairwise & Train-Test Dependent MAE')
    plt.ylabel('Pairwise Structural Similarity')
    plt.xlabel('Max Test-Train Structural Similarity')
    plt.xticks(np.arange(len(TT_SIM_BINS))-0.5, [f"{x:.2f}" for x in TT_SIM_BINS], rotation=90)
    plt.yticks(np.arange(len(PW_SIM_BINS))-0.5, [f"{x:.1f}" for x in PW_SIM_BINS])
    plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap_mae.png'), bbox_inches="tight")

    # Pairwise Similarity, Train-Test Distance Heatmap (Counts)
    plt.figure(figsize=GRID_FIG_SIZE)
    counts = pw_tt_metrics['count']
    # Transform counts by log base 10 +1
    counts = np.log10(counts + 1)
    plt.imshow(counts, vmin=0, origin='lower')
    plt.colorbar()
    plt.title('Pairwise & Train-Test Dependent Counts')
    plt.ylabel('Pairwise Structural Similarity')
    plt.xlabel('Max Test-Train Structural Similarity')
    plt.xticks(np.arange(len(TT_SIM_BINS))-0.5, [f"{x:.2f}" for x in TT_SIM_BINS], rotation=90)
    plt.yticks(np.arange(len(PW_SIM_BINS))-0.5, [f"{x:.1f}" for x in PW_SIM_BINS])
    plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap_counts.png'), bbox_inches="tight")

if __name__ == "__main__":
    main()