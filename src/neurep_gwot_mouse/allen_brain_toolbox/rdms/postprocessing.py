# %%
# Standard Library
import sys
from pathlib import Path
from typing import Any

# Third Party Library
import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator, TransformerMixin

# %%
HOME_DIR = Path(__file__).parent.parent
CURRENT_DIR = Path(__file__).parent
MANIFEST_PATH = HOME_DIR / "data/raw/ecephys_cache_dir/manifest.json"
sys.path.append(str(HOME_DIR))

# input
# shape = (trial, label, unit), np.ndarray


class DivideByMean(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        pass

    def fit(self, X, y: Any = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """RDMを平均で正規化する.

        Args:
            X (np.ndarray): shape=(label, label)

        Returns:
            np.ndarray: shape=(label, label)
        """

        mu = sp.spatial.distance.squareform(X, checks=False).mean()
        return X / mu
