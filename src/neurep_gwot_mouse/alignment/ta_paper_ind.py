#%% [markdown]
# # Functions for analsis between individuals. <br>
# NormalizeScaler -> MeanTrials -> MakeRDM(metric="cosine")
# EntropicGW2Computation

#%% [markdown]
# Standard Library
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third Party Library
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial.distance import squareform
from sklearn.metrics import top_k_accuracy_score
from tqdm.auto import tqdm

# First Party Library
from neurep_gwot_mouse.allen_brain_toolbox.rdms.pipeline import make_pipeline
from GW_methods.src import visualization
from GW_methods.src.align_representations import AlignRepresentations, OptimizationConfig, Representation


#%% [markdown]
# ## Function definition

# %%
# Define a variable to store data.
@dataclass
class SpikeDataContainer:
    session_ids: list[str]
    spike_counts: list[xr.DataArray]
    rdms: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.session_ids)

    def get_item(self, session_id: str, item: str) -> Union[xr.DataArray, np.ndarray]:
        idx = self.session_ids.index(session_id)
        return getattr(self, item)[idx]


# loading function
def load_spike_data_container(
    data_dir: Path,
    areas: List[str],
    pipeline_config: Dict[str, Any],
) -> Dict[str, SpikeDataContainer]:
    """Function to load spike_counts from an xr.DataArray

    Args:
        data_dir (Path): Directory containing the raw data
        areas (List[str]): List of brain areas to load

    Returns:
        Dict[str, SpikeDataContainer]: Dictionary with brain areas as keys and SpikeDataContainer instances as values
    """
    spike_data_container_dic: Dict[str, SpikeDataContainer] = {}
    for area in areas:
        print(f"load {area} data=======================")
        spike_data_container = SpikeDataContainer(session_ids=[], spike_counts=[], rdms=[])

        for p in sorted(data_dir.glob(f"*/{area}_spike_counts_da.nc")):
            da = xr.open_dataarray(p)

            # データのfiltering
            num_neurons = da.shape[2]
            sum0_stim = np.sum(da.values.sum(axis=2) == 0)

            if (num_neurons < 20) or (sum0_stim >= 10):  # Exclude if the number of neurons is less than 10, or if there exists a stimulus that no neurons respond to
                print(f"session {p.parent.stem} is excluded")
                continue

            # make rdm
            pipeline = make_pipeline(pipeline_config)
            rdm = pipeline.fit_transform(da.values)

            # save data
            spike_data_container.session_ids.append(p.parent.stem)
            spike_data_container.spike_counts.append(da)
            spike_data_container.rdms.append(rdm)

        spike_data_container_dic[area] = spike_data_container

    print("\nnumber of sessions")
    for key, container in spike_data_container_dic.items():
        print(f"{key}: number of animas = {len(container)}")

    return spike_data_container_dic


# analysis
def make_align_representations(
    session_ids: List[str],
    areas: List[str],
    rdms: List[np.ndarray],
    optim_config: OptimizationConfig,
    results_dir: Path,
    exp_name: str,
) -> AlignRepresentations:
    # make representations
    representations = []
    names = [f"{session_ids[0]}_{areas[0]}", f"{session_ids[1]}_{areas[1]}"]

    # make representations
    representations = []
    for name, rdm in zip(names, rdms):
        representations.append(
            Representation(
                name=name,
                embedding=None,
                sim_mat=rdm,
                get_embedding=False,
            )
        )

    # make align representations
    align_representation = AlignRepresentations(
        config=optim_config,
        representations_list=representations,
        histogram_matching=False,
        metric="cosine",
        main_results_dir=str(results_dir),
        data_name=exp_name,
    )
    return align_representation


def do_rsa(align_representation: AlignRepresentations) -> None:
    """Execute RSA and plot similarity matrix and distribution.

    Args:
        align_representation (AlignRepresentations): AlignRepresentations object.
    """

    # show sim_mat
    axes = visualization.show_sim_mat(
        align_representation,
        sim_mat_format="default",
        fig_dir=None,
        ticks=None,
        # keyword arguments
        show_figure=False,
        fig_ext="svg",  # you can also use "png" or "pdf", and so on. Default is "png".
        figsize=(6, 5),
        title_size=15,
        cmap='rocket_r',
    )

    # show distribution
    axes = visualization.show_distribution(
        align_representation,
        fig_dir=None,
        bins=50,
        # keyword arguments
        show_figure=False,
        fig_ext="svg", # you can also use "png" or "pdf", and so on. Default is "png".
        figsize=(6, 5),
        title_size=15,
        color='C0',
    )

    # RSA
    align_representation.calc_RSA_corr(metric="spearman")

    with open(Path(align_representation.main_results_dir) / "partial_RSA.json", "w") as f:
        json.dump(align_representation.RSA_corr, f)

    pass


def do_gw_alignment(
    align_representation: AlignRepresentations,
    compute_OT: bool=False,
    delete_results: bool=False,
    delete_database: bool=False,
    delete_directory: bool=False,
    sim_mat_format: str="default",
) -> None:
    # alignment
    align_representation.gw_alignment(
        compute_OT=compute_OT,
        delete_results=delete_results,
        delete_database=delete_database,
        delete_directory=delete_directory,
        return_data=False,
        OT_format=sim_mat_format,
    )
    axes = visualization.show_OT(
        align_representation,
        OT_format="default",
        # user can re-define the parameter if necessary.
        figsize=(6, 5),
        title_size=15,
        xlabel_size=15,
        ylabel_size=15,
        xticks_rotation=0,
        cbar_ticks_size=15,
        xticks_size=20,
        yticks_size=20,
        cbar_format="%.1e",
        cbar_label_size=15,
        cmap='rocket_r',
        fig_ext="svg", # you can also use "png" or "pdf", and so on. Default is "png".
        show_figure=False,
    )

    # Show optimization log
    axes_list = visualization.show_optimization_log(
        align_representation,
        fig_dir=None,
        # keyword arguments
        show_figure=False,
        fig_ext="svg", # you can also use "png" or "pdf", and so on. Default is "png".
        figsize=(6, 5),
        title_size=15,
        plot_eps_log=True,
        cmap='rocket_r',
    )

    # Calculate the accuracy based on the OT plan.
    align_representation.calc_accuracy(top_k_list=[1, 5, 10], eval_type="ot_plan")

    ax = visualization.plot_accuracy(
        align_representation,
        eval_type="ot_plan",
        scatter=True,
        show_figure=False,
    )

    top_k_accuracy = align_representation.top_k_accuracy  # you can get the dataframe directly
    pass


def calc_ot_matching_rate(OT: np.ndarray, ks: List[int] = [1, 5, 10]) -> Tuple[List[float], List[str]]:
    """matching rate of OT

    Args:
        OT (np.ndarray): OT matrix
        ks (List[int], optional): ks. Defaults to [1, 5, 10].

    Returns:
        Tuple[List[float], List[str]]: matching rate and matching rate name
    """

    y_true = np.arange(OT.shape[0])
    matching_rate, matching_rate_name = [], []
    for k in ks:
        matching_rate.append(top_k_accuracy_score(y_true, OT, k=k, labels=y_true) * 100)
        matching_rate_name.append(f"ot_top{k}_matching_rate")
    return matching_rate, matching_rate_name


def save_data(
    align_representation: AlignRepresentations,
    spike_data_container_dic: Dict[str, SpikeDataContainer],
    session_ids: List[str],
    areas: List[str],
    cond_results_dir: Path,
):

    # definition
    session_id_a, session_id_b = str(session_ids[0]), str(session_ids[1])
    area_a, area_b = areas

    # get spike counts
    spike_counts_a = spike_data_container_dic[area_a].get_item(session_id_a, "spike_counts")
    spike_counts_b = spike_data_container_dic[area_b].get_item(session_id_b, "spike_counts")

    # get rdms
    rdm_a = spike_data_container_dic[area_a].get_item(session_id_a, "rdms")
    rdm_b = spike_data_container_dic[area_b].get_item(session_id_b, "rdms")

    # save
    spike_dir = cond_results_dir / "spike_counts"
    spike_dir.mkdir(exist_ok=True, parents=True)
    np.save(spike_dir / f"{session_id_a}_spike_counts.npy", spike_counts_a)
    np.save(spike_dir / f"{session_id_b}_spike_counts.npy", spike_counts_b)

    rdm_dir = cond_results_dir / "rdms"
    rdm_dir.mkdir(exist_ok=True, parents=True)
    np.save(rdm_dir / f"{session_id_a}_rdm.npy", rdm_a)
    np.save(rdm_dir / f"{session_id_b}_rdm.npy", rdm_b)

    np.save(cond_results_dir / "best_OT.npy", align_representation.OT_list[0])
    pass


def whole_gw_alignment(
    pairs: List[List[str]],
    spike_data_container_dic: Dict[str, SpikeDataContainer],
    areas: List[str],
    optim_config: OptimizationConfig,
    exp_name: str,
    pairs_results_dir: Path,
    compute_OT: bool = False,
    delete_results: bool = False,
    delete_database: bool = False,
    delete_directory: bool = False,
    sim_mat_format: str = "default",
):
    for tmp_session_ids in pairs:

        # path setting
        cond_results_dir = pairs_results_dir / f"{tmp_session_ids[0]}_vs_{tmp_session_ids[1]}"
        cond_results_dir.mkdir(exist_ok=True, parents=True)

        # get data
        tmp_rdms = [
            spike_data_container_dic[areas[0]].get_item(str(tmp_session_ids[0]), "rdms"),
            spike_data_container_dic[areas[1]].get_item(str(tmp_session_ids[1]), "rdms"),
        ]

        # make align representations
        align_representation = make_align_representations(
            session_ids=tmp_session_ids,
            areas=areas,
            rdms=tmp_rdms,
            optim_config=optim_config,
            results_dir=cond_results_dir,
            exp_name=exp_name
        )

        # RSA
        do_rsa(align_representation)

        # GW alignment
        do_gw_alignment(
            align_representation=align_representation,
            compute_OT=compute_OT,
            delete_results=delete_results,
            delete_database=delete_database,
            delete_directory=delete_directory,
            sim_mat_format=sim_mat_format
        )

        # evaluate
        mc, _ = calc_ot_matching_rate(align_representation.OT_list[0], ks=[1])
        np.save(cond_results_dir / "OT_matching_rate.npy", mc[0])

        # save
        save_data(
            align_representation=align_representation,
            spike_data_container_dic=spike_data_container_dic,
            session_ids=tmp_session_ids,
            areas=areas,
            cond_results_dir=cond_results_dir,
        )
    pass


# %%
# main function
def ind_alignment_experiment(
    # main variables
    exp_name: str,  # "ta_paper_nc_ind"
    stimulus: str,  # "natural_scenes"
    storage_name: str,  # "ta_paper_sc"
    align_hyp: Dict[str, Any],
    gw_main_hyp: Dict[str, Any],
    # path settings
    current_dir: Path,
    data_dir: Path,
    pairs_dict_path: Path,
    config_path: Path,
):
    print("=======================================================================")
    print(f"Start experiments ''{exp_name}''!!!")
    print("This script performs alignment b/w areas of different individual mice.\n")

    # global variable settings
    if stimulus == "natural_scenes":
        assert storage_name == "ta_paper_sc"
    elif stimulus == "natural_movie_one_120frame":
        assert storage_name == "ta_paper_mv1_120frame"
    elif stimulus == "natural_movie_one_90frame":
        assert storage_name == "ta_paper_mv1_90frame"
    elif stimulus == "natural_movie_three_480frame":
        assert storage_name == "ta_paper_mv3_480frame"
    elif stimulus == "natural_movie_three":
        assert storage_name == "ta_paper_mv3_360frame"

    # path settings
    fig_dir = current_dir / f"../{stimulus}/img/{exp_name}"
    fig_dir.mkdir(exist_ok=True, parents=True)
    results_dir = current_dir / f"../{stimulus}/results/{exp_name}"
    results_dir.mkdir(exist_ok=True, parents=True)

    # variable settings
    areas = ["VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam", "LGd", "CA1"]

    eps_list = gw_main_hyp["eps_list"]
    eps_log = gw_main_hyp["eps_log"]
    num_trial = gw_main_hyp["num_trial"]

    compute_OT = align_hyp["compute_OT"]
    delete_results = align_hyp["delete_results"]
    delete_database = align_hyp["delete_database"]
    delete_directory = align_hyp["delete_directory"]
    sim_mat_format = align_hyp["sim_mat_format"]

    optim_config = OptimizationConfig(
        gw_type="entropic_gromov_wasserstein2",
        eps_list=eps_list,
        eps_log=eps_log,
        num_trial=num_trial,
        sinkhorn_method="sinkhorn",
        to_types="numpy",  # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
        device="cpu",  # "cuda" or "cpu"; for numpy, only "cpu" can be used.
        data_type="double",
        n_jobs=1,
        multi_gpu=False,
        storage=f"mysql+pymysql://root@localhost/{storage_name}",
        db_params=None,
        init_mat_plan="random",
        n_iter=1,
        max_iter=500,
        sampler_name="tpe",
        pruner_name="hyperband",
        pruner_params={"n_startup_trials": 1, "n_warmup_steps": 2, "min_resource": 2, "reduction_factor": 3},
    )

    # loading session pairs
    with open(pairs_dict_path, "r") as f:
        pairs_dict = json.load(f)

    # loading pipeline config
    with open(config_path, "r") as f:
        pipeline_config = yaml.safe_load(f)

    # loading spike-counts
    spike_data_container_dic = load_spike_data_container(data_dir, areas, pipeline_config)

    # analysis for each areas
    for i, area_a in enumerate(areas):
        for j, area_b in enumerate(areas):

            if i > j:
                continue

            # get pairs
            pairs = pairs_dict[f"{area_a}_vs_{area_b}"]

            # make pairs directory
            pair_fig_dir = fig_dir / f"{area_a}_vs_{area_b}"
            pair_fig_dir.mkdir(exist_ok=True, parents=True)
            pair_results_dir = results_dir / f"{area_a}_vs_{area_b}"
            pair_results_dir.mkdir(exist_ok=True, parents=True)

            # analysis
            whole_gw_alignment(
                pairs,
                spike_data_container_dic=spike_data_container_dic,
                areas=[area_a, area_b],
                optim_config=optim_config,
                exp_name=exp_name,
                pairs_results_dir=pair_results_dir,
                compute_OT=compute_OT,
                delete_results=delete_results,
                delete_database=delete_database,
                delete_directory=delete_directory,
                sim_mat_format=sim_mat_format,
            )


# # general variable settings
# #### TO BE CHANGED #############
# eps_list = [1e-4, 1e-1]
# eps_log = True
# num_trial = 100

# optim_config = OptimizationConfig(
#     gw_type="entropic_gromov_wasserstein2",
#     eps_list=eps_list,
#     eps_log=eps_log,
#     num_trial=num_trial,
#     sinkhorn_method='sinkhorn',
#     to_types="numpy",  # user can choose "numpy" or "torch". please set "torch" if one wants to use GPU.
#     device="cpu",  # "cuda" or "cpu"; for numpy, only "cpu" can be used.
#     data_type="double",
#     n_jobs=1,
#     multi_gpu=False,
#     storage=f"mysql+pymysql://root@localhost/{storage_name}",
#     db_params=None,
#     init_mat_plan="random",
#     n_iter=1,
#     max_iter=500,
#     sampler_name='tpe',
#     pruner_name='hyperband',
#     pruner_params={
#         'n_startup_trials': 1,
#         'n_warmup_steps': 2,
#         'min_resource': 2,
#         'reduction_factor' : 3
#     },
# )

# compute_OT = True
# delete_results = True
# delete_database = False
# delete_directory = False
# sim_mat_format = "default"



# for i, area_a in enumerate(areas):
#     for j, area_b in enumerate(areas):

#         if i > j:
#             continue

#         # get pairs
#         pairs = pairs_dict[f"{area_a}_vs_{area_b}"]

#         for tmp_session_ids in pairs:

#             # path setting
#             cond_results_dir = results_dir / f"{area_a}_vs_{area_b}/{tmp_session_ids[0]}_vs_{tmp_session_ids[1]}"
#             cond_results_dir.mkdir(exist_ok=True, parents=True)

#             # make align representations
#             align_representation = make_align_representations(
#                 spike_data_container_dic=spike_data_container_dic,
#                 session_ids=tmp_session_ids,
#                 areas=[area_a, area_b],
#                 optim_config=optim_config,
#                 results_dir=cond_results_dir,
#                 exp_name=exp_name
#             )

#             # RSA
#             do_rsa(align_representation)

#             # GW alignment
#             do_gw_alignment(
#                 align_representation=align_representation,
#                 compute_OT=compute_OT,
#                 delete_results=delete_results,
#                 delete_database=delete_database,
#                 delete_directory=delete_directory,
#                 sim_mat_format=sim_mat_format
#             )

#             # evaluate
#             mc, _ = calc_ot_matching_rate(align_representation.OT_list[0], ks=[1])
#             np.save(cond_results_dir / "OT_matching_rate.npy", mc[0])

#             # save
#             save_data(
#                 align_representation=align_representation,
#                 spike_data_container_dic=spike_data_container_dic,
#                 session_ids=tmp_session_ids,
#                 areas=[area_a, area_b],
#                 cond_results_dir=cond_results_dir,
#             )
# j


# # %%
