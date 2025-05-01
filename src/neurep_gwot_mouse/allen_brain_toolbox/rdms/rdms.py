# %%
# Standard Library
import functools
import sys
from pathlib import Path

# Third Party Library
import numpy as np
import ot
from phate import PHATE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import Isomap

# First Party Library
from ..tphate import TPHATE

# %%
HOME_DIR = Path(__file__).parent.parent
CURRENT_DIR = Path(__file__).parent
MANIFEST_PATH = HOME_DIR / "data/raw/ecephys_cache_dir/manifest.json"
sys.path.append(str(HOME_DIR))

# input
# shape = (trial, label, unit), np.ndarray


def custom_dist(metric: str, **kwargs) -> callable:
    if metric == "phate":
        return functools.partial(phate, **kwargs)

    elif metric == "t-phate":
        return functools.partial(t_phate, **kwargs)

    else:
        return functools.partial(ot.dist, metric=metric, **kwargs)


def phate(
    X: np.ndarray,
    dist_metric: str = "euclidean",
    seed: int = 42,
    n_jobs: int = 64,
    **kwargs,
) -> np.ndarray:
    """PHATEをかける

    Args:
        X (np.ndarray): shape=(label, unit)

    Keyword Args:
        knn (int):
            Number of nearest neighbors. Defaults to 5.
        knn_dist (str):
            Distance metric for nearest neighbor search. Defaults to 'euclidean'.
        mds_dist (str):
            Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
            Any metric from `scipy.spatial.distance` can be used. Custom distance
            functions of form `f(x, y) = d` are also accepted
        mds (str):
            choose from ['classic', 'metric', 'nonmetric'].
            Selects which MDS algorithm is used for dimensionality reduction
        decay (float):
            Alpha decay (default: 15). Decreasing decay increases connectivity on the graph, increasing decay decreases connectivity. This rarely needs to be tuned. Set it to None for a k-nearest neighbors kernel.
        t (int):
            Number of times to power the operator (default: 'auto'). This is equivalent to the amount of smoothing done to the data. It is chosen automatically by default, but you can increase it if your embedding lacks structure, or decrease it if the structure looks too compact.
        gamma (float):
            Informational distance constant (default: 1). gamma=1 gives the PHATE log potential, but other informational distances can be interesting. If most of the points seem concentrated in one section of the plot, you can try gamma=0.

    Returns:
        np.ndarray:
            RDM. shape=(label, label)
    """

    np.random.seed(seed)

    phate_operator = PHATE(n_jobs=n_jobs, **kwargs)
    phate_operator.fit(X)
    features = phate_operator.diff_potential

    rdm = ot.dist(features, metric=dist_metric)
    return rdm


def t_phate(
    X: np.ndarray,
    dist_metric: str = "euclidean",
    seed: int = 42,
    n_jobs: int = 64,
    **kwargs,
) -> np.ndarray:
    """T-PHATEをかける

    Args:
        X (np.ndarray): shape=(label, unit)

    Keyword Args:
        knn (int):
            Number of nearest neighbors. Defaults to 5.
        knn_dist (str):
            Distance metric for nearest neighbor search. Defaults to 'euclidean'.
        mds_dist (str):
            Distance metric for MDS. Recommended values: 'euclidean' and 'cosine'
            Any metric from `scipy.spatial.distance` can be used. Custom distance
            functions of form `f(x, y) = d` are also accepted
        mds (str):
            choose from ['classic', 'metric', 'nonmetric'].
            Selects which MDS algorithm is used for dimensionality reduction
        decay (float):
            Alpha decay (default: 15). Decreasing decay increases connectivity on the graph, increasing decay decreases connectivity. This rarely needs to be tuned. Set it to None for a k-nearest neighbors kernel.
        t (int):
            Number of times to power the operator (default: 'auto'). This is equivalent to the amount of smoothing done to the data. It is chosen automatically by default, but you can increase it if your embedding lacks structure, or decrease it if the structure looks too compact.
        gamma (float):
            Informational distance constant (default: 1). gamma=1 gives the PHATE log potential, but other informational distances can be interesting. If most of the points seem concentrated in one section of the plot, you can try gamma=0.

    Returns:
        np.ndarray:
            RDM. shape=(label, label)
    """

    np.random.seed(seed)

    tphate_operator = TPHATE(n_jobs=n_jobs, **kwargs)
    tphate_operator.fit(X)
    features = tphate_operator.diff_potential

    rdm = ot.dist(features, metric=dist_metric)
    return rdm


# %%
class MakeRDM(BaseEstimator, TransformerMixin):
    def __init__(self, metric: str = "cosine", **kwargs) -> None:
        """RDMを作成する関数

        Args:
            metric (str, optional): 距離の名前. Defaults to "cosine".

        Keyword Args:
            custom_distに渡すパラメータ
        """

        self.metric = metric
        self.dist_func = custom_dist(metric, **kwargs)

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """RDMを作成する. (label, unit)の距離行列を作成する

        Args:
            X (np.ndarray): shape=(label, unit)

        Returns:
            np.ndarray: shape=(label, label)
        """

        return self.dist_func(X)


class MakeTrialwiseRDM(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "mean", metric: str = "cosine", **kwargs) -> None:
        """trialごとにRDMを作成してその統計量を返す関数

        Args:
            method (str, optional): 統計量の名前. "mean" or "median". Defaults to "mean".
            metric (str, optional): 距離の名前. Defaults to "cosine".

        Keyword Args:
            custom_distに渡すパラメータ
        """

        self.make_rdm = MakeRDM(metric, **kwargs)
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """(label, label)の距離行列を作成する

        Args:
            X (np.ndarray): shape=(trial, label, unit)

        Returns:
            np.ndarray: shape=(label, label)
        """

        # trialごとにRDMを作成する
        rdms = []
        for trial in range(X.shape[0]):
            tmp_X = X[trial, :, :]
            rdms.append(self.make_rdm.fit_transform(tmp_X))

        # 統計量を計算する
        if self.method == "mean":
            rdm = np.mean(rdms, axis=0)

        elif self.method == "median":
            rdm = np.median(rdms, axis=0)

        else:
            raise ValueError("metric must be 'mean' or 'median'")

        return rdm


# %%
# manifold-base RDM
class IsomapRDM(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "trialwise", **kwargs) -> None:
        self.method = method
        self.isomap = Isomap(**kwargs)

    def fit(self, X, y=None):
        self.isomap.fit(X.reshape(-1, X.shape[2]))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        dist_matrix = self.isomap.dist_matrix_

        rdm = np.zeros((X.shape[1], X.shape[1]))
        n_trial, n_label, _ = X.shape

        # RDMの平均を取得する
        if self.method == "trialwise":
            for i in range(n_trial):
                rdm += dist_matrix[i * n_label : (i + 1) * n_label, i * n_label : (i + 1) * n_label] / n_trial

        elif self.method == "all":
            for i in range(n_trial):
                for j in range(n_trial):
                    rdm += dist_matrix[i * n_label : (i + 1) * n_label, j * n_label : (j + 1) * n_label] / (n_trial**2)

        np.fill_diagonal(rdm, 0)
        return rdm


# %%
