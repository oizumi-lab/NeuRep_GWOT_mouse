#%% [markdown]
# # 1-on-1 alignment <br>
# NormalizeScaler -> MeanTrials -> MakeRDM(metric="cosine")
# EntropicGW2Computation

# # Usage
# ```python ta_paper_ind_alignment.py <setting_file.csv>``` <br>
# <br>
# For how to create the setting file, refer to dummy_ind_alignment_setting.csv. <br>


#%% [markdown]
# ## Import libraries
# Standard Library
import os
import sys
import argparse

set_cpu = 64
os.environ["OPENBLAS_NUM_THREADS"] = str(set_cpu)
os.environ["MKL_NUM_THREADS"] = str(set_cpu)
os.environ["OMP_NUM_THREADS"] = str(set_cpu)

# Standard Library
from pathlib import Path

# Third Party Library
import pandas as pd
from neurep_gwot_mouse.alignment.ta_paper_ind import ind_alignment_experiment


#%% [markdown]
# ## Experiment settings

# %%
# general setting
#### TO BE CHANGED #############################################################################
# current_dir = Path(__file__).parent
# whole_data_dir = Path("/home/share/neuropixel_data")  # Path("/home/share/neuropixel_data/natural_scenes")
# pairs_dict_path = current_dir / f"../session_split/pairs_dict_ind.json"
# config_dir = current_dir / f"../config"
# setting_file = current_dir / "ta_paper_ind_alignment_setting.csv"  # "ta_paper_alignment_setting.csv"
################################################################################################

# %% [markdown]
# ## Contents of exp_setting_df
# exp_setting_df is a DataFrame where each row describes an experiment setting. Edit it according to your experiment.
# - exp_name:
#   - The prefix for the GW alignment study. It is recommended to include stimulus name, number of individuals, etc.
# - stimulus_name:
#   - Stimulus name (one of natural_movie_one, natural_movie_three, natural_scenes)
# - storage_name:
#   - Name of the RDB to save results. Intended to be set per experiment (e.g., 'takeda_abe_paper').
# - session_split_name:
#   - File name describing session split information for creating pseudo-mice. Assumed to be saved under session_split_dir.
# - config_name:
#   - Pipeline setting file name for converting spike_counts to RDM. Assumed to be saved under config_dir.
# <br> <br>
# ## Overview of files created in the experiment
# ```
# stimulus_name /
# ├── results/                   # Directory where GW alignment results are saved
# │   ├── exp_name/              # Directory named as specified by exp_name
# │   │   ├── condition_0/       # Results for each pseudo-mouse. The split method is determined by the file specified in "session_split_name".
# │   │   │   ├── VISp_vs_VISp/  # Alignment results for each area. Calculation results for 2_C_8 pairs are stored.
# │   │   │   ├── VISp_vs_VISrl/
# │   │   │   └── ...
# │   │   └── ...
# │   └── ...
# └── img/
#     ├── exp_name/             # Directory to save figures created after all calculations are finished.
#     └──  ...
# ```

# %%
# align representations hyperparameters
align_hyp = {
    "compute_OT": True,  # True for the first calculation
    "delete_results": True,  # True for the first calculation
    "delete_database": False,
    "delete_directory": False,
    "sim_mat_format": "default",
}

gw_main_hyp = {
    "eps_list": [1e-4, 1e-1],
    "eps_log": True,
    "num_trial": 100,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Individual alignment experiment runner")
    parser.add_argument("setting_file", type=str, help="Path to the setting CSV file")
    parser.add_argument("--current_dir", type=str, default=None, help="Current script directory (default: script location)")
    parser.add_argument("--whole_data_dir", type=str, default=None, help="Directory containing source data")
    parser.add_argument("--pairs_dict_path", type=str, default=None, help="Path to pairs_dict_ind.json")
    parser.add_argument("--config_dir", type=str, default=None, help="Directory for config files")
    return parser.parse_args()


def main():
    """usage:
    ```
    python ta_paper_ind_alignment.py <setting_file.csv> --target_dir <target_dir> --whole_data_dir <whole_data_dir> --pairs_dict_path <pairs_dict_path> --config_dir <config_dir>
    ```
    """
    args = parse_args()
    current_dir = Path(__file__).parent

    target_dir = Path(args.target_dir) if args.target_dir else current_dir / "../results"
    whole_data_dir = Path(args.whole_data_dir) if args.whole_data_dir else Path("/home/share/neuropixel_data")
    pairs_dict_path = Path(args.pairs_dict_path) if args.pairs_dict_path else current_dir / "../session_split/pairs_dict_ind.json"
    config_dir = Path(args.config_dir) if args.config_dir else current_dir / "../config"

    with open(args.setting_file) as f:
        exp_setting_df = pd.read_csv(f)

    for i in range(len(exp_setting_df)):
        exp_setting = exp_setting_df.iloc[i]
        ind_alignment_experiment(
            exp_name=exp_setting.loc["exp_name"],
            stimulus=exp_setting.loc["stimulus_name"],
            storage_name=exp_setting.loc["storage_name"],
            align_hyp=align_hyp,
            gw_main_hyp=gw_main_hyp,
            current_dir=target_dir,
            data_dir=whole_data_dir / exp_setting.loc["stimulus_name"],
            pairs_dict_path=pairs_dict_path,
            config_path=config_dir / exp_setting.loc["config_name"],
        )


if __name__ == "__main__":
    main()
