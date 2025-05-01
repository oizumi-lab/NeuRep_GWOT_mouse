# %%
# Standard Library
import sys
from pathlib import Path
from typing import Any

# Third Party Library
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

HOME_DIR = Path(__file__).parent.parent
CURRENT_DIR = Path(__file__).parent
sys.path.append(str(HOME_DIR))


class AllenCache:
    """Neuropixelのcacheを取得するためのクラス. 主にsessionを取得するために使用する

    Attributes:
        cache (EcephysProjectCache): neuropixelのcache
        session_table (pd.DataFrame): sessionの情報をまとめたDataFrame
    """

    def __init__(self, manifest_path: Path) -> None:
        """initializer

        Args:
            manifest_path (Path): manifest.jsonのパス
        """

        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
        self.session_table = self.cache.get_session_table()
        pass

    def get_session(self, session_id: int) -> Any:
        """session_idからsessionを取得する

        Args:
            session_id (int): session_id

        Returns:
            pd.DataFrame: session
        """

        return self.cache.get_session_data(session_id)

    def get_unit_metrics(self, session_id: int) -> pd.DataFrame:
        """session_idからunit_metricsを取得する

        Args:
            session_id (int): session_id

        Returns:
            pd.DataFrame: unit_metrics
        """

        return self.cache.get_unit_analysis_metrics_for_session(session_id)
