# %%
# Standard Library
import itertools
import sys
from collections.abc import Iterable
from typing import Optional, Tuple

sys.dont_write_bytecode = True


# Third Party Library
import numpy as np
import pandas as pd
import xarray as xr

# Local Library
from .utils import AllenCache


class SessionToData:
    """sessionからdataを取得するためのclass

    Attributes:
        session_id (int):
            session_id
        session (EcephysSession):
            session
        allen_cache (AllenCache):
            neuropixelのoriginal cache
        unit_metrics (pd.DataFrame):
            unit_metrics
    """

    def __init__(
        self,
        session_id: int,
        allen_cache: AllenCache,
    ) -> None:
        """initializer

        Args:
            session_id (int):
                session_id
            cache (AllenCache):
                neuropixelのoriginal cache
        """

        self.session_id = session_id
        self.session = allen_cache.get_session(session_id)
        self.allen_cache = allen_cache
        self.unit_metrics = allen_cache.get_unit_metrics(session_id)
        pass

    def get_stimulus_presentation_df(self, stimulus_name: str) -> pd.DataFrame:
        """stimulus_presentationのDataFrameを取得する.

        Args:
            stimulus_name (str):
                stimulusの名前. "natural_scenes", "natural_movie_one" or "natural_movie_three".

        Returns:
            pd.DataFrame:
                stimulus_presentationのDataFrame
        """

        stimulus_presentation_df = self.session.get_stimulus_table(stimulus_name)

        if stimulus_name == "natural_scenes":
            # 画像が表示されていないフレームを除外する
            stimulus_presentation_df = stimulus_presentation_df[stimulus_presentation_df["frame"] != -1]

        elif stimulus_name == "natural_movie_one":
            # 動画の開始時点のフレームのみを取得する
            stimulus_presentation_df = stimulus_presentation_df[stimulus_presentation_df["frame"] == 0.0]

        elif stimulus_name == "natural_movie_three":
            # 動画の開始時点のフレームのみを取得する
            stimulus_presentation_df = stimulus_presentation_df[stimulus_presentation_df["frame"] == 0.0]

        # trialを追加する
        trials_dic = dict.fromkeys(sorted(stimulus_presentation_df["frame"].unique()), 0)
        trials = []

        for frame in stimulus_presentation_df["frame"]:
            trials.append(trials_dic[frame])
            trials_dic[frame] += 1

        stimulus_presentation_df["trial"] = trials

        return stimulus_presentation_df

    def get_spike_counts_da(
        self,
        stimulus_name: str,
        unit_ids: Iterable,
        stimulus_presentation_ids: Iterable,
        stimulus_presentation_df: pd.DataFrame,
        time_bin_edges: Optional[np.ndarray] = None,
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """spike_countsのDataArrayとtrial_info_dfを取得する.

        Args:
            stimulus_name (str):
                stimulusの名前. "natural_scenes", "natural_movie_one" or "natural_movie_three".
            unit_ids (Iterable):
                unit_idのリスト
            stimulus_presentation_ids (Iterable):
                stimulus_presentation_idのリスト
            stimulus_presentation_df (pd.DataFrame):
                stimulus_presentationのDataFrame
            time_bin_edges (Optional[np.ndarray], optional):
                time_bin_edges. Defaults to None.

        Raises:
            ValueError: stimulus_nameが不正な場合に発生する

        Returns:
            spike_counts_da (xr.DataArray):
                spike_countsのDataArray. shape=(trial, label, unit_id).
            trial_info_df (pd.DataFrame):
                trial_info_df. index=(trial, label). columns=("order", "start_time", "sum0_stim").
        """

        if time_bin_edges is None:
            if stimulus_name == "natural_scenes":
                time_bin_edges = np.arange(0.0, 0.26, 0.01)

            elif stimulus_name == "natural_movie_one":
                time_bin_edges = np.arange(0.0, 30.25, 0.25)

            elif stimulus_name == "natural_movie_three":
                time_bin_edges = np.arange(0.0, 120.25, 0.25)

            else:
                raise ValueError(f"stimulus_name: {stimulus_name} is invalid.")

        if stimulus_name == "natural_scenes":
            spike_counts_da, trial_info_df = self.get_spike_counts_da_sc(
                unit_ids=unit_ids,
                stimulus_presentation_ids=stimulus_presentation_ids,
                stimulus_presentation_df=stimulus_presentation_df,
                time_bin_edges=time_bin_edges,
            )

        elif stimulus_name in ["natural_movie_one", "natural_movie_three"]:
            spike_counts_da, trial_info_df = self.get_spike_counts_da_mv(
                unit_ids=unit_ids,
                stimulus_presentation_ids=stimulus_presentation_ids,
                stimulus_presentation_df=stimulus_presentation_df,
                time_bin_edges=time_bin_edges,
            )

        return spike_counts_da, trial_info_df

    def get_spike_counts_da_mv(
        self,
        unit_ids: Iterable,
        stimulus_presentation_ids: Iterable,
        stimulus_presentation_df: pd.DataFrame,
        time_bin_edges: Optional[np.ndarray] = None,
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """movie stimulusのspike_countsのDataArrayとtrial_info_dfを取得する.

        Args:
            unit_ids (Iterable):
                unit_idのリスト
            stimulus_presentation_ids (Iterable):
                stimulus_presentation_idのリスト
            stimulus_presentation_df (pd.DataFrame):
                stimulus_presentationのDataFrame
            time_bin_edges (Optional[np.ndarray], optional):
                time_bin_edges. Defaults to None.

        Returns:
            spike_counts_da (xr.DataArray):
                spike_countsのDataArray. shape=(trial, label, unit_id).
            trial_info_df (pd.DataFrame):
                trial_info_df. index=(trial, label). columns=("order", "start_time", "sum0_stim").
        """

        # load spike counts
        spike_counts_da = self.session.presentationwise_spike_counts(
            bin_edges=time_bin_edges,
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=unit_ids,
        )
        spike_counts_da = spike_counts_da.assign_coords(
            stimulus_presentation_id=np.arange(spike_counts_da.shape[0], dtype=int),
            time_relative_to_stimulus_onset=np.arange(spike_counts_da.shape[1], dtype=int),
        ).rename(
            {
                "stimulus_presentation_id": "trial",
                "time_relative_to_stimulus_onset": "label",
            }
        )

        # make time dataframe
        order = np.arange(spike_counts_da.shape[0] * spike_counts_da.shape[1])
        start_time = (stimulus_presentation_df["start_time"].values[:, None] + time_bin_edges[None, :-1]).reshape(-1)
        sum0_stim = (spike_counts_da.sum(dim="unit_id").values.reshape(-1) == 0).astype(int)

        multi_index = pd.MultiIndex.from_tuples(
            itertools.product(spike_counts_da.trial.values, spike_counts_da.label.values),
            names=("trial", "label"),
        )
        trial_info_df = pd.DataFrame(
            {"order": order, "start_time": start_time, "sum0_stim": sum0_stim},
            index=multi_index,
        )

        return spike_counts_da, trial_info_df

    def get_spike_counts_da_sc(
        self,
        unit_ids: Iterable,
        stimulus_presentation_ids: Iterable,
        stimulus_presentation_df: pd.DataFrame,
        time_bin_edges: Optional[np.ndarray] = None,
    ) -> Tuple[xr.DataArray, pd.DataFrame]:
        """scenes stimulusのspike_countsのDataArrayとtrial_info_dfを取得する.

        Args:
            unit_ids (Iterable):
                unit_idのリスト
            stimulus_presentation_ids (Iterable):
                stimulus_presentation_idのリスト
            stimulus_presentation_df (pd.DataFrame):
                stimulus_presentationのDataFrame
            time_bin_edges (Optional[np.ndarray], optional):
                time_bin_edges. Defaults to None.

        Returns:
            spike_counts_da (xr.DataArray):
                spike_countsのDataArray. shape=(trial, label, unit_id).
            trial_info_df (pd.DataFrame):
                trial_info_df. index=(trial, label). columns=("order", "start_time", "sum0_stim").
        """

        # add information to stimulus_presentation_df
        stimulus_presentation_df = stimulus_presentation_df.rename(columns={"frame": "label"}).astype({"label": "int"})
        stimulus_presentation_df["order"] = np.arange(stimulus_presentation_df.shape[0], dtype=int)

        # make time dataframe
        sorted_stimulus_stimulus_presentation_df = stimulus_presentation_df.sort_values(
            ["trial", "label"]
        )  # trialの順番に並び替える
        trial_info_df = sorted_stimulus_stimulus_presentation_df[["trial", "label", "order", "start_time"]].set_index(
            ["trial", "label"]
        )

        # load spike counts
        spike_counts_da = self.session.presentationwise_spike_counts(
            bin_edges=time_bin_edges,
            stimulus_presentation_ids=stimulus_presentation_ids,
            unit_ids=unit_ids,
        )
        sum_spike_counts_da = spike_counts_da.sum(
            axis=1
        )  # (stimulus_presentation_id, unit_id) 250msec内でのspike countの合計

        n_trials = sorted_stimulus_stimulus_presentation_df["trial"].nunique()

        # trialごとにspike_countsを取得する
        new_spike_counts_da_l = []
        for i in range(n_trials):
            idx = sorted_stimulus_stimulus_presentation_df.loc[
                sorted_stimulus_stimulus_presentation_df["trial"] == i
            ].index.values
            new_spike_counts_da_l.append(sum_spike_counts_da.sel(stimulus_presentation_id=idx).values)

        new_spike_counts_da = np.stack(new_spike_counts_da_l)

        # DataArrayに変換する (trial, label, unit_id)
        new_spike_counts_da = xr.DataArray(
            new_spike_counts_da,
            coords=[
                np.arange(n_trials, dtype=int),
                np.arange(new_spike_counts_da.shape[1], dtype=int),
                spike_counts_da.unit_id,
            ],
            dims=["trial", "label", "unit_id"],
        )

        trial_info_df["sum0_stim"] = (new_spike_counts_da.sum(dim="unit_id").values.reshape(-1) == 0).astype(int)

        return new_spike_counts_da, trial_info_df
