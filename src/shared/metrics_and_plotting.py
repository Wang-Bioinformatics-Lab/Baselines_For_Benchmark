import argparse
from pathlib import Path
import os

from matplotlib import cm
import pandas as pd
import dask.dataframe as dd
import dask.array as da
from dask.distributed import LocalCluster
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gc
import seaborn as sns

TT_SIM_BINS = np.linspace(0.4,1.0, 13)
PW_SIM_BINS = np.linspace(0.,1.0, 11)

from utils import train_test_similarity_dependent_losses, \
                    tanimoto_dependent_losses, \
                    train_test_similarity_heatmap, \
                    train_test_similarity_bar_plot, \
                    fixed_tanimoto_train_test_similarity_dependent, \
                    pairwise_train_test_dependent_heatmap, \
                    roc_curve, \
                    top_k_analysis, \
                    get_top_k_scores, \
                    get_top_k_scores_for_bins, \
                    pr_curve

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

plt.style.use('seaborn-v0_8-deep')

GRID_FIG_SIZE = (10,7)

def top_k_similarities_line_plot(top_scores,):
    # Line plot of the top score and max score
    fig = plt.figure(figsize=(6, 6))
    # Concatenate the top scores and max scores into a single array
    top_scores_array = np.stack(list(top_scores['top_k_scores'].values()))
    optimal_scores_array = np.stack(list(top_scores['optimal_scores'].values()))
    # Average 
    top_scores_mean = np.nanmean(top_scores_array, axis=0)
    optimal_scores_mean = np.nanmean(optimal_scores_array, axis=0)
    # Standard Deviation
    # top_scores_std = np.nanstd(top_scores_array, axis=0)
    # optimal_scores_std = np.nanstd(optimal_scores_array, axis=0)
    # Line Plot with Error Bars
    # plt.errorbar(np.arange(len(top_scores_mean)), top_scores_mean, yerr=top_scores_std, label="Top 10 Predictions")
    # plt.errorbar(np.arange(len(optimal_scores_mean)), optimal_scores_mean, yerr=optimal_scores_std, label="Theoretical Max")
    # Regular line plot
    plt.plot(np.arange(len(top_scores_mean)), top_scores_mean, label="Top 10 Predictions")
    plt.plot(np.arange(len(optimal_scores_mean)), optimal_scores_mean, label="Theoretical Max")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Rank")
    plt.ylabel("Tanimoto Score")
    plt.title("Top 10 Tanimoto Scores")
    plt.legend()
    return fig

def top_k_distance_line_plot(top_scores):
    fig = plt.figure(figsize=(6, 6))
    # Concatenate the difference scores into a single array
    differance_array = np.stack(list(top_scores['difference_dict'].values()))

    # Average
    difference_mean = np.nanmean(differance_array, axis=0)

    # Line Plot
    plt.plot(np.arange(len(difference_mean)), difference_mean)
    plt.ylim(0.0, 1)
    plt.xlabel("Rank")
    plt.ylabel("Difference of Tanimoto Score Means")
    plt.title("Top 10 Tanimoto Scores")
    return fig

def top_k_similarities_violin_plot(top_scores):
    data = []
    ranks = np.arange(len(next(iter(top_scores['top_k_scores'].values()))))
    top_scores_array = np.stack(list(top_scores['top_k_scores'].values()))
    optimal_scores_array = np.stack(list(top_scores['optimal_scores'].values()))
    for i, rank in enumerate(ranks):
        for score in top_scores_array[:, i]:
            data.append({'Rank': rank, 'Score': score, 'Type': 'Top 10 Predictions'})
        for score in optimal_scores_array[:, i]:
            data.append({'Rank': rank, 'Score': score, 'Type': 'Theoretical Max'})

    df = pd.DataFrame(data)

    # Create the violin plot
    fig = plt.figure(figsize=(12, 6))
    sns.violinplot(x='Rank', y='Score', hue='Type', data=df, split=True, inner="quartile")
    plt.xlabel("Rank")
    plt.ylabel("Tanimoto Score")
    plt.title("Top 10 Tanimoto Scores vs Theoretical Max")
    plt.legend(title="Score Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig

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
    overall_nan_rmse = da.sqrt(da.nanmean(da.square(presampled_pairs['error'].values))).compute()
    overall_nan_mae = da.nanmean(da.abs(presampled_pairs['error'])).compute()
    print("Overall NAN RMSE (from evaluate()):", overall_nan_rmse)
    print("Overall NAN MAE (from evaluate()):", overall_nan_mae)
    print("Nan Count:", presampled_pairs['error'].isna().sum().compute())

    # Get Top K Tanimoto Similarities (including identical inchikeys)
    print("Creating Top K Tanimoto Score Analysis Including Identical InChIKeys", flush=True)
    top_scores = get_top_k_scores(presampled_pairs, k=10)
    print("top_scores_averages:", [f"{x:.4f}" for x in top_scores['top_scores_averages']])
    print("optimal_scores_averages:", [f"{x:.4f}" for x in top_scores['optimal_scores_averages']])
    print("difference_averages:", [f"{x:.4f}" for x in top_scores['difference_averages']])
    # Save the top scores
    top_scores_path = os.path.join(metric_dir, "top_k_scores.pkl")
    pickle.dump(top_scores, open(top_scores_path, "wb"))
    # Line Plot
    fig = top_k_similarities_line_plot(top_scores)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_line.png'))
    # Violin Plot
    fig = top_k_similarities_violin_plot(top_scores)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_violin.png'))
    # Line Plot for Differences
    fig = top_k_distance_line_plot(top_scores)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_distance_line.png'))

    print("Calculating Top K Tanimoto Scores (excluding identical inchikeys)", flush=True)
    top_scores_no_identical = get_top_k_scores(presampled_pairs, k=10, remove_identical_inchikeys=True)
    print("top_scores_averages:", [f"{x:.4f}" for x in top_scores_no_identical['top_scores_averages']])
    print("optimal_scores_averages:", [f"{x:.4f}" for x in top_scores_no_identical['optimal_scores_averages']])
    print("difference_averages:", [f"{x:.4f}" for x in top_scores_no_identical['difference_averages']])
    # Save the top scores
    top_scores_path = os.path.join(metric_dir, "top_k_scores_no_identical.pkl")
    pickle.dump(top_scores_no_identical, open(top_scores_path, "wb"))
    # Line Plot
    fig = top_k_similarities_line_plot(top_scores_no_identical)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_no_identical_line.png'))
    # Violin Plot
    fig = top_k_similarities_violin_plot(top_scores_no_identical)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_no_identical_violin.png'))
    # Line Plot for Differences
    fig = top_k_distance_line_plot(top_scores_no_identical)
    fig.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_no_identical_distance_line.png'))

    # Binned Top-K Scores
    print("Creating Binned Top K Tanimoto Scores", flush=True)
    top_scores_binned_w_identical = get_top_k_scores_for_bins(presampled_pairs, TT_SIM_BINS, k=10, remove_identical_inchikeys=False)
    top_scores_binned_no_identical = get_top_k_scores_for_bins(presampled_pairs, TT_SIM_BINS, k=10, remove_identical_inchikeys=True)
    # Dumpy to pikle
    output_path = os.path.join(metric_dir, "top_k_scores_binned_w_identical.pkl")
    pickle.dump(top_scores_binned_w_identical, open(output_path, "wb"))
    output_path = os.path.join(metric_dir, "top_k_scores_binned_no_identical.pkl")
    pickle.dump(top_scores_binned_no_identical, open(output_path, "wb"))
    # Plot Mean Distance to Optimal Tanimoto Scores
    cmap = cm.get_cmap('viridis', len(top_scores_binned_w_identical.keys()))
    plt.figure(figsize=(10,7))
    for idx, k in enumerate(top_scores_binned_w_identical.keys()):
        color = cmap(idx / len(top_scores_binned_w_identical.keys()))
        data = np.array(top_scores_binned_w_identical[k]['optimal_scores_averages']) - np.array(top_scores_binned_w_identical[k]['top_scores_averages'])
        plt.plot(np.arange(len(data)), data, '--o', label=k, color=color, markersize=8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Train-Test Similarity')
    plt.ylabel("Mean Distance to Optimal Tanimoto Score")
    plt.xlabel("Prediction Rank")
    plt.xticks(np.arange(len(data)), np.arange(1, len(data) + 1))
    plt.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_binned_w_identical.png'), bbox_inches="tight")

    plt.figure(figsize=(10,7))
    for idx, k in enumerate(top_scores_binned_no_identical.keys()):
        color = cmap(idx / len(top_scores_binned_no_identical.keys()))
        data = np.array(top_scores_binned_no_identical[k]['optimal_scores_averages']) - np.array(top_scores_binned_no_identical[k]['top_scores_averages'])
        plt.plot(np.arange(len(data)), data, '--o', label=k, color=color, markersize=8)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Train-Test Similarity')
    plt.ylabel("Mean Distance to Optimal Tanimoto Score")
    plt.xlabel("Prediction Rank")
    plt.xticks(np.arange(len(data)), np.arange(1, len(data) + 1))
    plt.savefig(os.path.join(metric_dir, 'top_k_tanimoto_scores_binned_no_identical.png'), bbox_inches="tight")

    # Calculate Top k Analysis
    print("Creating Top k Analysis", flush=True)
    k_lst = [1,3,10]
    top_k_metrics = top_k_analysis(presampled_pairs, k_list=k_lst, remove_identical_inchikeys=True)
    top_k_metrics_w_identical = top_k_analysis(presampled_pairs, k_list=k_lst, remove_identical_inchikeys=False)
    # Each dict contains 'mean_top_rank' and 'std_top_rank'
    top_k_path = os.path.join(metric_dir, "top_k_metrics.pkl")
    pickle.dump(top_k_metrics, open(top_k_path, "wb"))
    top_k_path = os.path.join(metric_dir, "top_k_metrics_w_identical.pkl")
    pickle.dump(top_k_metrics_w_identical, open(top_k_path, "wb"))

    # Bar chart with error bars comparing the two for all k values
    plt.figure(figsize=(8, 6))
    mean_values = []
    std_values = []
    labels = []

    for k in k_lst:
        mean_values.append([top_k_metrics[k]['mean_top_rank'], top_k_metrics_w_identical[k]['mean_top_rank']])
        std_values.append([top_k_metrics[k]['std_top_rank'], top_k_metrics_w_identical[k]['std_top_rank']])
        labels.append(f'k={k} Non-Identical InChI')
        labels.append(f'k={k} Identical InChI')

    mean_values = np.array(mean_values).flatten()
    std_values = np.array(std_values).flatten()
    positions = np.arange(len(mean_values))

    plt.bar(positions, mean_values, yerr=std_values)
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylabel("Mean Top Rank")
    plt.title("Predicted Rank of Highest Similarity\nStructure for Various k")
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, 'top_k_analysis_bar.png'))

    # Violin Plot of Top k Analysis for all k values
    plt.figure(figsize=(8, 6))
    all_ranks = []
    labels = []

    for k in k_lst:
        all_ranks.append(list(top_k_metrics[k]['all_ranks'].values()))
        all_ranks.append(list(top_k_metrics_w_identical[k]['all_ranks'].values()))
        labels.append(f'k={k} Non-Identical InChI')
        labels.append(f'k={k} Identical InChI')

    plt.violinplot(all_ranks, showmedians=True)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("Top Rank")
    plt.title("Predicted Rank of Highest Similarity\nStructure for Various k")
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, 'top_k_analysis_violin.png'))


    # Calculate ROC Curve
    print("Creating ROC Curve Including Identical InChiKeys", flush=True)
    roc_metrics_w_identical = roc_curve(presampled_pairs, np.linspace(1.0,0.0, 41), exclude_identical_inchikeys=False)
    roc_path = os.path.join(metric_dir, "roc_metrics_w_identical.pkl")
    pickle.dump(roc_metrics_w_identical, open(roc_path, "wb"))
    plt.figure(figsize=(6, 6))
    plt.plot(roc_metrics_w_identical["fpr"], roc_metrics_w_identical["tpr"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_metrics_w_identical['auc']:.2f})")
    plt.savefig(os.path.join(metric_dir, 'roc_curve.png'))

    # Calculate ROC Curve Excluding Identical InChiKeys
    print("Creating ROC Curve Excluding Identical InChiKeys", flush=True)
    # 250_000 ppm would mean no more than a 25% mass delta
    roc_metrics_no_identical = roc_curve(presampled_pairs, np.linspace(1.0,0.0, 41), precursor_ppm_diff=250_000, exclude_identical_inchikeys=True, positive_threshold=0.6)
    roc_path = os.path.join(metric_dir, "roc_metrics_no_identical.pkl")
    pickle.dump(roc_metrics_no_identical, open(roc_path, "wb"))
    plt.figure(figsize=(6, 6))
    plt.plot(roc_metrics_no_identical["fpr"], roc_metrics_no_identical["tpr"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_metrics_no_identical['auc']:.2f})")
    plt.savefig(os.path.join(metric_dir, 'roc_curve_no_identical.png'))

    # Calculate PR Curve Including Identical InChiKeys
    print("Creating PR Curve Including Identical InChiKeys", flush=True)
    pr_metrics_w_identical = pr_curve(presampled_pairs, np.linspace(1.0,0.0, 41), exclude_identical_inchikeys=False)
    pr_path = os.path.join(metric_dir, "pr_metrics.pkl")
    pickle.dump(pr_metrics_w_identical, open(pr_path, "wb"))
    plt.figure(figsize=(6, 6))
    plt.plot(pr_metrics_w_identical["recall"], pr_metrics_w_identical["precision"])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AUC={pr_metrics_w_identical['auc']:.2f})")
    plt.savefig(os.path.join(metric_dir, 'pr_curve.png'))

    # Calculate PR Curve Excluding Identical InChiKeys
    print("Creating PR Curve Excluding Identical InChiKeys", flush=True)
    pr_metrics_no_identical = pr_curve(presampled_pairs, np.linspace(1.0,0.0, 41), precursor_ppm_diff=250_000, exclude_identical_inchikeys=True, positive_threshold=0.6)
    pr_path = os.path.join(metric_dir, "pr_metrics_no_identical.pkl")
    pickle.dump(pr_metrics_no_identical, open(pr_path, "wb"))
    plt.figure(figsize=(6, 6))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(pr_metrics_no_identical["recall"], pr_metrics_no_identical["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AUC={pr_metrics_no_identical['auc']:.2f})")
    plt.savefig(os.path.join(metric_dir, 'pr_curve_no_identical.png'))


    # Train-Test Similarity Dependent Losses Aggregated (rmse)
    print("Creating Train-Test Similarity Dependent Losses Aggregated Plot", flush=True)
    similarity_dependent_metrics_mean = train_test_similarity_dependent_losses(presampled_pairs, TT_SIM_BINS, mode='mean')
    plt.figure(figsize=(12, 9))
    plt.bar(np.arange(len(similarity_dependent_metrics_mean["rmses"]),), similarity_dependent_metrics_mean["rmses"],)
    # Add labels on top of bars
    for i, v in enumerate(similarity_dependent_metrics_mean["rmses"]):
        plt.text(i, v + 0.001, f"{v:.2f}", ha='center', va='bottom', fontsize=14)
    plt.title(f'Train-Test Dependent RMSE\nAverage RMSE: {overall_nan_rmse:.2f}')
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
    plt.title(f'Train-Test Dependent MAE\nAverage MAE: {overall_nan_mae:.2f}')
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
    metric_dict["nan_rmse"] = overall_nan_rmse
    metric_dict["nan_mae"]  = overall_nan_mae

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
    cbar = plt.colorbar()
    cbar_ticks = cbar.get_ticks()
    cbar_labels = [f'$10^{{{int(tick)}}}$' for tick in cbar_ticks]
    cbar.set_label('Log Count')
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    plt.title('Pairwise & Train-Test Dependent Counts')
    plt.ylabel('Pairwise Structural Similarity')
    plt.xlabel('Max Test-Train Structural Similarity')
    plt.xticks(np.arange(len(TT_SIM_BINS))-0.5, [f"{x:.2f}" for x in TT_SIM_BINS], rotation=90)
    plt.yticks(np.arange(len(PW_SIM_BINS))-0.5, [f"{x:.1f}" for x in PW_SIM_BINS])
    plt.savefig(os.path.join(metric_dir, 'pairwise_train_test_heatmap_counts.png'), bbox_inches="tight")

if __name__ == "__main__":
    main()