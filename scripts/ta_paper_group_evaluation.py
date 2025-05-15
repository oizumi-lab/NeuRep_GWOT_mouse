# %% [markdown]
# ## Evaluation of alignment results for pseudo-mouse
# ### Outputs to be created
# - Alignment results for the same brain area
#   - top-1 matching rate
#   - RSA
# - top-1 matching rate, RSA
#   - csv file
#   - heatmap
#   - dendrogram

# # Usage
# ```python ta_paper_group_evaluation.py 'stimulus_name' 'exp_name'``` <br />
# * Evaluates the results of experiments performed with `ta_paper_group_alignment.py`. <br />
# * The generated figures and tables are saved under `stimulus_name/img/exp_name`. <br />


#%% [markdown]
# ## Import libraries
# Standard Library
import json
import sys
import argparse
from pathlib import Path

# Third Party Library
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# First Party Library
from neurep_gwot_mouse.alignment.fig_funcs import across_area_heatmap, plot_clustering, swarm_same_areas


# %%
areas = ["VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam", "LGd", "CA1"]


def aggregate_calcuration(results_dir: Path, v_name: str = "spearman_r") -> np.ndarray:
    """Aggregate evaluation metrics from GW results.

    Args:
        results_dir (Path): Directory where the results are saved.
        v_name (str, optional): Name of the evaluation metric. "spearman_r" or "ot_top1_matching_rate". Defaults to "spearman_r".

    Returns:
        np.ndarray: Matrix containing the values of the evaluation metric. shape=(n_samples, n_areas, n_areas)
    """
    # get n_samples
    n = sum(1 for _ in results_dir.glob("condition*"))
    matrix = np.zeros((n, len(areas), len(areas)))

    # load results
    for k, p in enumerate(results_dir.glob("condition*")):
        for i, area_a in enumerate(areas):
            for j, area_b in enumerate(areas):

                if i > j:
                    continue

                # load values
                if v_name == "spearman_r":
                    with open(p / f"{area_a}_vs_{area_b}/partial_RSA.json", "r") as f:
                        d = json.load(f)
                    matrix[k, i, j] = next(iter(d.values()))

                elif v_name == "ot_top1_matching_rate":
                    matrix[k, i, j] = np.load(p / f"{area_a}_vs_{area_b}/OT_matching_rate.npy")

                if i != j:
                    matrix[k, j, i] = matrix[k, i, j]
    return matrix


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Group alignment experiment evaluator")
    parser.add_argument("stimulus_name", type=str, default=None, help="Stimulus name")
    parser.add_argument("exp_name", type=str, default=None, help="Experiment name")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory to save results")
    parser.add_argument("--fig_dir", type=str, default=None, help="Directory to save figures")

    return parser.parse_args()


if __name__ == "__main__":
    "usage: python ta_paper_group_evaluation.py 'stimulus_name' 'exp_name'"
    # Parse command line arguments
    args = parse_args()
    current_dir = Path(__file__).parent

    stimulus_name = args.stimulus_name
    exp_name = args.exp_name

    # Set directories
    results_dir = Path(args.results_dir) if args.results_dir else current_dir / f"../results/{stimulus_name}/results/{exp_name}"
    fig_dir = Path(args.fig_dir) if args.fig_dir else current_dir / f"../results/{stimulus_name}/img/{exp_name}"

    # Create directories if they do not exist
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in ["spearman_r", "ot_top1_matching_rate"]:

        # load results
        mat = aggregate_calcuration(results_dir, metric_name)

        # save results
        np.save(results_dir / f"{metric_name}_mat.npy", mat)

        # swarm plots
        df = pd.DataFrame(np.array([np.diag(mat[i]) for i in range(mat.shape[0])]), columns=areas)
        swarm_same_areas(df, metric_name, fig_dir)

        # heatmap
        mean_mat = mat.mean(axis=0)
        t = "Spearman R" if metric_name == "spearman_r" else "OT top-1 matching rate (%)"
        fmt = ".2f" if metric_name == "spearman_r" else ".1f"
        vmax = 1 if metric_name == "spearman_r" else 100
        across_area_heatmap(mean_mat, areas, t, vmax=vmax, fmt=fmt, fig_dir=fig_dir, fig_name=f"heatmap_{metric_name}")

        # dendrogram
        dist_mat = 1 - mean_mat if metric_name == "spearman_r" else 1 - mean_mat / 100
        d = squareform(dist_mat, checks=False)
        Z_ward = linkage(d, method="ward")
        t = "Spearman R" if metric_name == "spearman_r" else "OT top-1 matching rate (%)"
        plot_clustering(Z_ward, areas, title=t, fig_dir=fig_dir, fig_name=f"dendrogram_{metric_name}")

    # %%
