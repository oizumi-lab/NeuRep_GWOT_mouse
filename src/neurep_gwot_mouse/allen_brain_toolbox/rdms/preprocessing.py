# %%
# Standard Library
import sys
from pathlib import Path
from typing import Any, Optional

# Third Party Library
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import LocalOutlierFactor

# %%
HOME_DIR = Path(__file__).parent.parent
CURRENT_DIR = Path(__file__).parent
MANIFEST_PATH = HOME_DIR / "data/raw/ecephys_cache_dir/manifest.json"
sys.path.append(str(HOME_DIR))

# input
# shape = (trial, label, unit), np.ndarray


# 0 as nan
class DropZeroTrials(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """全てのunitが0のtrialを削除する

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(trial, label, unit)
        """

        sum_X = X.sum(axis=2)
        bool_idx = np.where((sum_X != 0).any(axis=1))[0]  # 0でないtrialのindex
        return X[bool_idx, :, :]


class FillZero(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """全てのunitが0のtrial, labelをlabelの平均値で埋める

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(trial, label, unit)
        """

        org_shape = X.shape
        X = X.astype(np.float32)
        X = X.reshape(-1, X.shape[2])

        # unitの和が0のtrial,labelをnanに置き換え
        sum_X = X.sum(axis=1)
        bool_idx = np.where(sum_X.reshape(-1) == 0)[0]  # 0である(trial, label)のindex

        if len(bool_idx) / len(sum_X) > 0.5:
            raise ValueError("0であるtrial, labelが半分以上あります")

        X[bool_idx, :] = np.nan
        X = X.reshape(org_shape)

        if len(bool_idx) > 0:
            print("Start FillZero preprocessing.=======================")
            print(f"全てのunitが0をとる施行は{len(bool_idx)}個です")

            # nanを除いてmeanを取る
            mean_X = np.nanmean(X, axis=0)
            for idx in bool_idx:
                tr_idx = idx // X.shape[1]
                la_idx = idx % X.shape[1]

                X[tr_idx, la_idx, :] = mean_X[la_idx, :]

        return X


# %%


# Scaler
class NormalizeScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        mu: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.mu = mu
        self.std = std

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Any:
        """Normalize by mean and std

        Args:
            X (np.ndarray): shape=(trial, label, unit)
            y (Optional[np.ndarray], optional): Defaults to None.
        """

        X = X.reshape(-1, X.shape[2])  # (trial0, label0), (trial0, label1),...のようになる
        if self.mu is None:
            self.mu = np.mean(X, axis=0)

        if self.std is None:
            self.std = np.std(X, axis=0)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """normalizationをかけた後に元のshapeに戻す

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(trial, label, unit)
        """
        original_shape = X.shape
        X = (X.reshape(-1, X.shape[2]) - self.mu) / self.std
        return X.reshape(original_shape)


# Preprocessing Trials
class FilterTrials(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        remained_trials: int = 25,
        n_neighbors: int = 20,
    ) -> None:
        """initialization

        Args:
            remained_trials (int, optional):
                残すtrialの数. Defaults to 25.
            n_neighbors (int, optional):
                LOFに渡すパラメータ. knnの隣接ノード数. Defaults to 20.
        """

        self.remained_trials = remained_trials
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Normalize by mean and std

        Args:
            X (np.ndarray): shape=(trial, label, unit)
            y (Optional[np.ndarray], optional): Defaults to None.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """LOFを用いて外れ値のscoreを検出し、上位remained_trials個のtrialを残す

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(trial, label, unit)
        """

        X = X.transpose(1, 0, 2)  # (label, trial, unit)

        tmp_Xs = []
        for label in range(X.shape[0]):
            tmp_X = X[label]  # (trial, unit)
            clf = LocalOutlierFactor(n_neighbors=self.n_neighbors)
            clf.fit(X[label])
            idx = np.argsort(-clf.negative_outlier_factor_)  # negative valueが大きいほど外れ値
            idx = idx[: self.remained_trials]
            tmp_Xs.append(tmp_X[idx])

        return np.stack(tmp_Xs, axis=1)


class MeanTrials(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """trialの平均をとる.

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(label, unit)
        """
        return X.mean(axis=0)
