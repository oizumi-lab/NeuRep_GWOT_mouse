# %%
# custom distance functionを返す

# Standard Library
from typing import Any, Dict

# Third Party Library
from sklearn.pipeline import Pipeline

from .postprocessing import DivideByMean

# First Party Library
from .preprocessing import (
    DropZeroTrials,
    FillZero,
    FilterTrials,
    MeanTrials,
    NormalizeScaler,
)
from .rdms import MakeRDM


def make_pipeline(step_dict: Dict[str, dict]) -> Pipeline:
    """RDMを作るためのPipelineを作る.
    shape = (trial, label, unit)の配列を受けて, shape = (label, unit)の配列を返す

    Args:
        step_dict (Dict[str, dict]): pipelineを作るためのdict

    Returns:
        Pipeline: pipeline
    """

    pipeline = []
    for step_name, step_params in step_dict.items():
        trf = load_trf(step_name, **step_params)
        pipeline.append((step_name, trf))

    pipeline = Pipeline(pipeline)
    return pipeline


def load_trf(step_name: str, **kwargs) -> Any:
    if step_name == "normalize_scaler":
        return NormalizeScaler(**kwargs)

    elif step_name == "fill_zero":
        return FillZero(**kwargs)

    elif step_name == "drop_zero_trials":
        return DropZeroTrials(**kwargs)

    elif step_name == "filter_trials":
        return FilterTrials(**kwargs)

    elif step_name == "mean_trials":
        return MeanTrials(**kwargs)

    elif step_name == "make_rdm":
        return MakeRDM(**kwargs)

    elif step_name == "divide_by_mean":
        return DivideByMean(**kwargs)


# %%
