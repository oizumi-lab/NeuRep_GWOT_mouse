import contextlib
import functools
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from tqdm.auto import tqdm

from neurep_gwot_mouse.allen_brain_toolbox.data_interface.neuropixel.session_to_data import (
    SessionToData,
)
from neurep_gwot_mouse.allen_brain_toolbox.data_interface.neuropixel.utils import (
    AllenCache,
)

StimulusName = Literal["natural_scenes", "natural_movie_one", "natural_movie_three"]
Area = Literal["VISp", "VISrl", "VISl", "VISal", "VISpm", "VISam", "LGd", "CA1"]


class Stimulus(Enum):
    """Enum class for different types of visual stimuli"""

    NATURAL_SCENES = "natural_scenes"
    NATURAL_MOVIE_ONE = "natural_movie_one"
    NATURAL_MOVIE_THREE = "natural_movie_three"

    @property
    def default_time_window(self) -> float:
        """Default time window for each stimulus type"""
        if self == Stimulus.NATURAL_SCENES:
            return 0.01
        elif self in [Stimulus.NATURAL_MOVIE_ONE, Stimulus.NATURAL_MOVIE_THREE]:
            return 1 / 3

    @property
    def duration(self) -> float:
        """Total duration per trial for each stimulus (seconds)"""
        if self == Stimulus.NATURAL_SCENES:
            return 0.25
        elif self == Stimulus.NATURAL_MOVIE_ONE:
            return 30.0
        elif self == Stimulus.NATURAL_MOVIE_THREE:
            return 120.0

    def get_time_bin_edges(self, time_window: Optional[float] = None) -> np.ndarray:
        """Calculate time bin edges based on the specified time window

        Args:
            time_window: Width of the time window. Uses default value if None

        Returns:
            np.ndarray: Array of time bin edges
        """
        tw = time_window if time_window is not None else self.default_time_window
        duration = self.duration
        return np.arange(0, duration + tw, tw)

    def get_shape(self, time_window: Optional[float] = None) -> tuple[int, int]:
        """Calculate expected array shape based on the specified time window

        Args:
            time_window: Width of the time window. Uses default value if None

        Returns:
            tuple[int, int]: Shape as (trial, label)
        """
        edges = self.get_time_bin_edges(time_window)
        num_frames = len(edges) - 1

        if self == Stimulus.NATURAL_SCENES:
            return (50, 118)
        elif self == Stimulus.NATURAL_MOVIE_ONE:
            return (20, num_frames)
        elif self == Stimulus.NATURAL_MOVIE_THREE:
            return (10, num_frames)
        else:
            raise ValueError(f"Unknown stimulus: {self}")

    def get_save_dir_name(self, base_dir: str, time_window: Optional[float] = None) -> str:
        """Generate directory name for saving data

        Args:
            base_dir: Base directory path
            time_window: Width of the time window. Uses default value if None

        Returns:
            str: Directory path for saving data
        """
        if self == Stimulus.NATURAL_SCENES:
            return str(Path(base_dir) / self.value)
        else:
            edges = self.get_time_bin_edges(time_window)
            num_frames = len(edges) - 1
            return str(Path(base_dir) / self.value / f"{num_frames}frame")


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """Context manager to patch joblib to report into tqdm progress bar given as argument.

    This function applies tqdm to joblib's parallel processing for progress tracking.
    Reference: https://github.com/louisabraham/tqdm_joblib
    """

    tqdm_object = tqdm(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class AllenDataLoader:
    """Class for loading and processing Allen Brain Observatory data"""

    def __init__(self, manifest_json_path: str) -> None:
        """Initialize the data loader

        Args:
            manifest_json_path: Path to the Allen Brain Observatory manifest JSON file.
                                The file will be automatically generated when used for the first time.
        """
        self.manifest_json_path = manifest_json_path
        assert manifest_json_path.endswith(".json"), "manifest_json_path must end with '.json'"

        self.allen_cache = AllenCache(Path(manifest_json_path))

        self.session_ids = self.allen_cache.session_table[
            self.allen_cache.session_table.session_type == "brain_observatory_1.1"
        ].index.values

    def get_stimulus_type(self, stimulus_name: StimulusName) -> Stimulus:
        """Convert string or StimulusName to Stimulus enum

        Args:
            stimulus_name: String stimulus name or Stimulus enum

        Returns:
            Stimulus: Corresponding Stimulus enum

        Raises:
            ValueError: If stimulus name is not recognized
        """
        if isinstance(stimulus_name, str):
            for stim in Stimulus:
                if stim.value == stimulus_name:
                    return stim
        raise ValueError(f"Unknown stimulus name: {stimulus_name}")

    def get_spike_counts(
        self,
        stimulus_name: StimulusName,
        session_id: int,
        area: Area,
        time_window: Optional[float] = None,
        snr_threshold: float = 1.0,
        base_dir: Optional[str] = None,
    ) -> None:
        """Get spike count data for the specified session (Mouse ID)

        The saved data will have the following directory structure:

        ```
        {base_dir} /
        ├── natural_scenes /
        │   ├── {session_id} /
        │      ├── {area}_spike_counts_da.nc
        │      ├── {area}_trial_info_df.pkl
        │
        ├── natural_movie_one /
        │   ├── xxxframe /                          <- Spike Counts aggregated based on time-window。(30 ÷ time_window)
        │   │   ├── {session_id} /                  <- Mouse ID
        │   │   │   ├── {area}_spike_counts_da.nc   <- Spike counts for each brain area in xarray format (3D version of pd.DataFrame).
        │   │   │   │                                  Open like `xr.open_dataarray({path})`
        │   │   │   │                                  trial x label(frame) x unit_id(neuron)
        │           ├── {area}_trial_info_df.pkl    <- Information on when each trial started, etc. Open like `joblib.load({path})`
        │
        └── natural_movie_three /
            ├── xxxframe /                          <- Spike Counts aggregated based on time-window。(30 ÷ time_window)
                ├── {session_id} /
                    ├── {area}_spike_counts_da.nc
                    ├── {area}_trial_info_df.pkl
        ```

        Args:
            stimulus_name: Stimulus name, one of 'natural_scenes', 'natural_movie_one', 'natural_movie_three'
            session_id: Session ID (individual mouse ID)
            area: Brain region
            time_window: Width of time window. Uses default value if None
            snr_threshold: Signal-to-noise ratio threshold
            base_dir: Directory for saving data. `base_dir` is parent directory of `self.manifet_json_path` if None
        """
        # print start
        print(f"Start processing stimulus {stimulus_name}, session {session_id} for area {area}...")

        # Get stimulus enum
        stimulus: Stimulus = self.get_stimulus_type(stimulus_name)

        # Initialize session data handler
        session2data = SessionToData(session_id, self.allen_cache)

        # Get unit IDs (neuron ID) that meet criteria
        unit_ids = session2data.unit_metrics.loc[
            (session2data.unit_metrics["ecephys_structure_acronym"] == area)
            & (session2data.unit_metrics["snr"] >= snr_threshold)
        ].index.values

        # Get stimulus presentation IDs
        stimulus_presentation_df = session2data.get_stimulus_presentation_df(stimulus_name)
        stimulus_presentation_ids = stimulus_presentation_df.index.values

        # get time_bin_edges
        if time_window is None:
            time_window = stimulus.default_time_window
            print(f"Using default time window: {time_window}")

        time_bin_edges = stimulus.get_time_bin_edges(time_window)

        # Get spike count data
        spike_counts_da, trial_info_df = session2data.get_spike_counts_da(
            stimulus_name,
            unit_ids,
            stimulus_presentation_ids,
            stimulus_presentation_df,
            time_bin_edges,
        )

        # save
        if base_dir is None:
            base_dir = str(Path(self.manifest_json_path).parent)
        save_dir = stimulus.get_save_dir_name(base_dir, time_window)
        print(f"Save directory: {save_dir}")

        # Validate data shape
        expected_shape = stimulus.get_shape(time_window)
        actual_shape = spike_counts_da.shape[:2]

        if actual_shape != expected_shape:
            print("Skip saving: Shape mismatch.")
            print(f"Session {session_id}, area {area}: Expected {expected_shape}, got {actual_shape}")
            pass

        save_session_dir = Path(save_dir) / str(session_id)
        save_session_dir.mkdir(exist_ok=True, parents=True)

        spike_counts_da.to_netcdf(save_session_dir / f"{area}_spike_counts_da.nc")
        joblib.dump(trial_info_df, save_session_dir / f"{area}_trial_info_df.pkl")

    def get_spike_counts_parallel(
        self,
        stimulus_name: StimulusName,
        area: Area,
        time_window: Optional[float] = None,
        snr_threshold: float = 1.0,
        base_dir: Optional[str] = None,
        n_jobs: int = 24,
    ) -> None:
        """Get spike count data for the specified stimulus and area in parallel.

        This function automatically retrieves session IDs (Mouse IDs) containing the specified area
        and processes each session ID in parallel.
        The saved data directory structure is reffered to in `get_spike_counts`.

        Args:
            stimulus_name: Stimulus name, one of 'natural_scenes', 'natural_movie_one', 'natural_movie_three'
            area: Brain region
            time_window: Width of time window. Uses default value if None
            snr_threshold: Signal-to-noise ratio threshold
            base_dir: Directory for saving data. `base_dir` is parent directory of `self.manifet_json_path` if None
            n_jobs: Number of parallel jobs. Default is 24.
        """
        # Get session IDs that contain the specified area
        area_session_ids = []
        for session_id in self.session_ids:
            if area in self.allen_cache.session_table.loc[session_id, "ecephys_structure_acronyms"]:
                area_session_ids.append(session_id)
        print(f"Found {len(area_session_ids)} sessions containing area {area}")

        # parallel processing
        func = functools.partial(
            self.get_spike_counts,
            stimulus_name=stimulus_name,
            area=area,
            time_window=time_window,
            snr_threshold=snr_threshold,
            base_dir=base_dir,
        )

        with tqdm_joblib(tqdm(desc=f"save {stimulus_name}", total=len(area_session_ids))) as progress_bar:
            joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                joblib.delayed(func)(session_id=session_id) for session_id in area_session_ids
            )
