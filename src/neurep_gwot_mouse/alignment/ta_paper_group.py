#%% [markdown]
# # pseudo-mouse間の解析に必要な関数群 <br>
# 設定はconfig/ns_mt_cosine_all2_combination.yamlを参照する. <br>
# EntropicGW2Computation
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
from GWTune.src import visualization
from GWTune.src.align_representations import AlignRepresentations, OptimizationConfig, Representation

#%% [markdown]
# ## Function definition

#%%
# Define a variable to store data.
@dataclass
class SpikeDataContainer:
    session_ids: list[str]
    spike_counts: list[xr.DataArray]
    rdms: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.session_ids)

    def get_item(self, session_id: Union[str, int], item: str) -> Union[xr.DataArray, np.ndarray]:
        idx = self.session_ids.index(str(session_id))
        return getattr(self, item)[idx]


@dataclass
class PseudoSpikeDataContainer:
    areas: list[str]
    spike_counts: list[xr.DataArray]
    rdms: list[np.ndarray]
    session_ids: list[str]

    def __len__(self) -> int:
        return len(self.spike_counts)

    def get_item(self, area: str, item: str) -> Union[xr.DataArray, np.ndarray]:
        if item == "session_ids":
            raise ValueError("the index of session_ids is not area")

        idx = self.areas.index(area)
        return getattr(self, item)[idx]


# loading function
def load_spike_data_container(
    data_dir: Path,
    areas: List[str],
) -> Dict[str, SpikeDataContainer]:
    """xr.DataArrayのspike_countsを読み込む関数

    Args:
        data_dir (Path): raw dataのdirectory
        areas (List[str]): 読み込むareaのリスト

    Returns:
        Dict[str, SpikeDataContainer]: areaをkey, SpikeDataContainerをvalueとした辞書
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

            if (num_neurons < 20) or (sum0_stim >= 10):  # neuronの数が10未満の場合は除外, どのニューロンも発火しないstimulusがある場合は除外
                print(f"session {p.parent.stem} is excluded")
                continue

            # データの保存
            spike_data_container.session_ids.append(p.parent.stem)
            spike_data_container.spike_counts.append(da)

        spike_data_container_dic[area] = spike_data_container

    print("\nnumber of sessions")
    for key, container in spike_data_container_dic.items():
        print(f"{key}: number of animas = {len(container)}")

    return spike_data_container_dic


# making pseudo container
def choose_split(session_split_df: pd.DataFrame, condition: int) -> Tuple[List[str], List[str]]:
    """csvからsession_idsを取得する関数

    Args:
        session_split_df (pd.DataFrame): pseudo-mouseの分割が格納されたDataFrame
        condition (int): どの列を選択するか

    Returns:
        Tuple[List[str], List[str]]: session_ids_a, session_ids_b
    """
    # select column
    ds = session_split_df.iloc[:, condition]

    # extract session_ids
    session_ids_a = ds[ds == "a"].index.values
    session_ids_b = ds[ds == "b"].index.values

    return session_ids_a, session_ids_b


def make_pseudo_container(
    areas: List[str], session_ids: Any, pipeline_config: dict, spike_data_container_dic: Dict[str, SpikeDataContainer]
) -> PseudoSpikeDataContainer:
    """pseudo-mouseの神経活動を結合して、RDMを作成する関数

    Args:
        areas (List[str]): 8領野
        session_ids (List[str]): pseudo-mouseのsession_ids
        pipeline_config (dict): RDMを作成するための設定
        spike_data_container_dic (Dict[str, SpikeDataContainer]): areaをkey, SpikeDataContainerをvalueとした辞書

    Returns:
        PseudoSpikeDataContainer: pseudo-mouseの神経活動を結合したもの
    """
    if isinstance(session_ids, np.ndarray):
        session_ids = session_ids.astype(str).tolist()

    pseudo_spike_data_container = PseudoSpikeDataContainer(
        areas=areas, spike_counts=[], rdms=[], session_ids=session_ids
    )

    with tqdm(areas, desc="making pseudo container") as progress_bar:
        for area in progress_bar:
            progress_bar.set_postfix_str(f"area: {area}")

            spike_data_container = spike_data_container_dic[area]

            # indexの抽出
            indices = []
            for session_id in session_ids:
                if str(session_id) in spike_data_container.session_ids:
                    indices.append(spike_data_container.session_ids.index(str(session_id)))

            # spike_countsのconcat
            pseudo_spike_counts = np.concatenate([spike_data_container.spike_counts[i].values for i in indices], axis=2)

            # make rdm
            pipeline = make_pipeline(pipeline_config)
            rdm = pipeline.fit_transform(pseudo_spike_counts)

            # store data
            pseudo_spike_data_container.spike_counts.append(pseudo_spike_counts)
            pseudo_spike_data_container.rdms.append(rdm)

    return pseudo_spike_data_container


# analysis
def make_align_representations(
    areas: List[str],
    rdms: List[np.ndarray],
    optim_config: OptimizationConfig,
    results_dir: Path,
    exp_name: str,
) -> AlignRepresentations:
    # make representations
    representations = []
    names = [f"pseudo_a_{areas[0]}", f"pseudo_b_{areas[1]}"]

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
    """RSAを行う. similarity matrixとdistributionをplotする.

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
        cmap="rocket_r",
    )

    # show distribution
    axes = visualization.show_distribution(
        align_representation,
        fig_dir=None,
        bins=50,
        # keyword arguments
        show_figure=False,
        fig_ext="svg",  # you can also use "png" or "pdf", and so on. Default is "png".
        figsize=(6, 5),
        title_size=15,
        color="C0",
    )

    # RSA
    align_representation.calc_RSA_corr(metric="spearman")

    with open(Path(align_representation.main_results_dir) / "partial_RSA.json", "w") as f:
        json.dump(align_representation.RSA_corr, f)

    pass


def do_gw_alignment(
    align_representation: AlignRepresentations,
    compute_OT: bool = False,
    delete_results: bool = False,
    delete_database: bool = False,
    delete_directory: bool = False,
    sim_mat_format: str = "default",
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
        cmap="rocket_r",
        fig_ext="svg",  # you can also use "png" or "pdf", and so on. Default is "png".
        show_figure=False,
    )

    # Show optimization log
    axes_list = visualization.show_optimization_log(
        align_representation,
        fig_dir=None,
        # keyword arguments
        show_figure=False,
        fig_ext="svg",  # you can also use "png" or "pdf", and so on. Default is "png".
        figsize=(6, 5),
        title_size=15,
        plot_eps_log=True,
        cmap="rocket_r",
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
    """calculating matching rate of OT

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


def save_pseudo_info(
    var_name: str,
    pseudo_spike_data_container_a: PseudoSpikeDataContainer,
    pseudo_spike_data_container_b: PseudoSpikeDataContainer,
    results_dir: Path,
) -> None:
    """save pseudo_spike_data_container

    Args:
        var_name (str): "rdms", "spike_counts", "session_ids"
        pseudo_spike_data_container_a (PseudoSpikeDataContainer): pseudo_spike_data_container_a
        pseudo_spike_data_container_b (PseudoSpikeDataContainer): pseudo_spike_data_container_b
        results_dir (Path): 保存先のdirectory
    """
    if var_name is "session_ids":
        tmp_results_dir = results_dir / f"{var_name}"
        tmp_results_dir.mkdir(exist_ok=True, parents=True)
        np.save(tmp_results_dir / "pseudo_a_session_ids.npy", pseudo_spike_data_container_a.session_ids)
        np.save(tmp_results_dir / "pseudo_b_session_ids.npy", pseudo_spike_data_container_b.session_ids)

    else:
        source_results_dir, target_results_dir = (
            results_dir / f"{var_name}/pseudo_a",
            results_dir / f"{var_name}/pseudo_b",
        )
        source_results_dir.mkdir(exist_ok=True, parents=True)
        target_results_dir.mkdir(exist_ok=True, parents=True)

        for i in range(len(pseudo_spike_data_container_a)):
            area = pseudo_spike_data_container_a.areas[i]
            pseudo_a_var, pseudo_b_var = pseudo_spike_data_container_a.get_item(
                area, var_name
            ), pseudo_spike_data_container_b.get_item(area, var_name)
            np.save(source_results_dir / f"{area}_{var_name}.npy", pseudo_a_var)
            np.save(target_results_dir / f"{area}_{var_name}.npy", pseudo_b_var)
    pass


def reshape_evals(ev: np.ndarray, n: int) -> np.ndarray:
    """[VISp_vs_VISp, VISp_vs_VISrl, ..., VISrl_vs_VISrl, ..., CA1_vs_CA1]のような1次元配列を2次元配列にreshapeする関数

    Args:
        ev (np.ndarray): 1次元配列
        n (int): 領野の数

    Returns:
        np.ndarray: 2次元配列
    """
    new_ev = np.zeros((n, n))
    c = 0

    for i in range(n):
        for j in range(n):
            if i > j:
                continue

            new_ev[i, j] = ev[c]
            c += 1

    return (new_ev + new_ev.T) - np.diag(np.diag(new_ev))


def whole_gw_alignment(
    areas: List[str],
    pseudo_spike_data_container_a: PseudoSpikeDataContainer,
    pseudo_spike_data_container_b: PseudoSpikeDataContainer,
    optim_config: OptimizationConfig,
    exp_name: str,
    results_dir: Path,
    compute_OT: bool = False,
    delete_results: bool = False,
    delete_database: bool = False,
    delete_directory: bool = False,
    sim_mat_format: str = "default",
) -> None:
    """main function. 領野ごとのRSA + GW alignmentを行う.

    Args:
        areas (List[str]): 8領野
        pseudo_spike_data_container_a (PseudoSpikeDataContainer): pseudo_spike_data_container_a
        pseudo_spike_data_container_b (PseudoSpikeDataContainer): pseudo_spike_data_container_b
        optim_config (OptimizationConfig): GW alignmentの設定
        exp_name (str): 実験名
        results_dir (Path): 保存先のdirectory
        compute_OT (bool, optional): _description_. Defaults to False.
        delete_results (bool, optional): _description_. Defaults to False.
        delete_database (bool, optional): _description_. Defaults to False.
        delete_directory (bool, optional): _description_. Defaults to False.
        sim_mat_format (str, optional): _description_. Defaults to "default".
    """
    corrs = []
    OT_list = []
    pairname_list = []
    ot_top1 = []

    for i in range(len(areas)):
        for j in range(len(areas)):

            if i > j:
                continue

            # get data
            tmp_areas = [pseudo_spike_data_container_a.areas[i], pseudo_spike_data_container_b.areas[j]]
            tmp_rdms = [pseudo_spike_data_container_a.rdms[i], pseudo_spike_data_container_b.rdms[j]]

            # make results dir
            tmp_results_dir = results_dir / f"{tmp_areas[0]}_vs_{tmp_areas[1]}"  # pair of areas
            tmp_results_dir.mkdir(exist_ok=True, parents=True)

            # make align representations
            align_representation = make_align_representations(
                areas=tmp_areas,
                rdms=tmp_rdms,
                optim_config=optim_config,
                results_dir=tmp_results_dir,
                exp_name=exp_name,
            )

            # RSA
            do_rsa(align_representation)

            # GW alignment
            do_gw_alignment(
                align_representation,
                compute_OT=compute_OT,
                delete_results=delete_results,
                delete_database=delete_database,
                delete_directory=delete_directory,
                sim_mat_format=sim_mat_format,
            )

            # evaluation
            mc, _ = calc_ot_matching_rate(align_representation.OT_list[0], ks=[1])
            np.save(Path(align_representation.main_results_dir) / "OT_matching_rate.npy", mc[0])

            # store result
            corrs.append(list(align_representation.RSA_corr.values())[0])
            OT_list.append(align_representation.OT_list[0])
            pairname_list.append(f"{tmp_areas[0]}_vs_{tmp_areas[1]}")
            ot_top1.append(mc[0])

    # reshape
    corrs = reshape_evals(np.array(corrs), len(areas))
    ot_top1 = reshape_evals(np.array(ot_top1), len(areas))

    # save info
    save_pseudo_info("rdms", pseudo_spike_data_container_a, pseudo_spike_data_container_b, results_dir)
    save_pseudo_info("spike_counts", pseudo_spike_data_container_a, pseudo_spike_data_container_b, results_dir)
    save_pseudo_info("session_ids", pseudo_spike_data_container_a, pseudo_spike_data_container_b, results_dir)

    # save results
    np.save(results_dir / "spearmanr_matrix.npy", corrs)
    np.save(results_dir / "ot_top1_matching_rate.npy", ot_top1)

    OT_dict = {n: d for n, d in zip(pairname_list, OT_list)}
    with open(results_dir / "OT_dict.pkl", "wb") as f:
        pickle.dump(OT_dict, f)
    pass


# main function
def group_alignment_experiment(
    # main variables
    exp_name: str,  # "ta_paper_nc_group4"
    stimulus: str,  # "natural_scenes"
    storage_name: str,  # "ta_paper_sc"
    align_hyp: Dict[str, Any],
    gw_main_hyp: Dict[str, Any],
    # path settings
    current_dir: Path,
    data_dir: Path,
    session_split_path: Path,
    config_path: Path,
):
    print("=======================================================================")
    print(f"Start experiments ''{exp_name}''!!!")
    print("This script performs alignment b/w areas of different pseudo-mice.\n")

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
    fig_dir = current_dir / f"img/{exp_name}"
    fig_dir.mkdir(exist_ok=True, parents=True)
    results_dir = current_dir / f"results/{exp_name}"
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

    # loading database
    session_split_df = pd.read_csv(session_split_path, index_col=0)

    with open(config_path, "r") as f:
        pipeline_config = yaml.safe_load(f)

    # loading spike-counts
    spike_data_container_dic = load_spike_data_container(data_dir, areas)

    # analysis for each condition (pseudo-mouse)
    for i in range(session_split_df.shape[1]):

        # experiment setting
        cond_results_dir = results_dir / f"condition_{i}"
        cond_exp_name = f"condition_{i}_{exp_name}"

        # making pseudo-mouse and calculating RDM
        session_ids_a, session_ids_b = choose_split(session_split_df, condition=i)
        pseudo_spike_data_container_a = make_pseudo_container(
            areas, session_ids_a, pipeline_config, spike_data_container_dic
        )
        pseudo_spike_data_container_b = make_pseudo_container(
            areas, session_ids_b, pipeline_config, spike_data_container_dic
        )

        # analysis
        whole_gw_alignment(
            areas,
            pseudo_spike_data_container_a,
            pseudo_spike_data_container_b,
            optim_config=optim_config,
            exp_name=cond_exp_name,
            results_dir=cond_results_dir,
            compute_OT=compute_OT,
            delete_results=delete_results,
            delete_database=delete_database,
            delete_directory=delete_directory,
            sim_mat_format=sim_mat_format,
        )
