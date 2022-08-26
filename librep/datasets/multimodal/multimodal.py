from typing import Tuple, Optional, Union, List

import pandas as pd

from librep.base.data import Dataset
from librep.utils.datasets import PandasDataset
from librep.config.type_definitions import ArrayLike


class MultiModalDataset(Dataset):
    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @property
    def window_names(self) -> List[str]:
        raise NotImplementedError

    @property
    def num_windows(self) -> int:
        raise NotImplementedError


class ArrayMultiModalDataset(MultiModalDataset):
    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        window_slices: List[Tuple[int, int]],
        window_names: List[str] = None,
    ):
        self.X = X
        self.y = y
        self._window_slices = window_slices
        self._window_names = window_names or []

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)

    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        return self._window_slices

    @property
    def window_names(self) -> List[str]:
        return self._window_names

    @property
    def num_windows(self) -> int:
        return len(self._window_slices)

    def __str__(self):
        return f"ArrayMultiModalDataset: samples={len(self.X)}, shape={len(self.X)}, no. window={self.num_windows}"


class PandasMultiModalDataset(PandasDataset, MultiModalDataset):
    """Dataset implementation for multi modal PandasDataset.
    It assumes that each sample is composed is a feature vector where
    parts of this vector comes from different natures.
    For instance, a sample with 900 features where features:
    - 0-299: correspond to acelerometer x
    - 300-599: correspond to accelerometer y
    - 600-899: correspond to accelerometer z

    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        feature_prefixes: Optional[Union[str, List[str]]] = None,
        label_columns: Union[str, List[str]] = "activity code",
        as_array: bool = True,
    ):
        """The MultiModalHAR dataset, derived from Dataset.
        The __getitem__ returns 2-element tuple where:
        - The first element is the sample (from the indexed-row of the
        dataframe with the selected features, as features); and
        - The seconds element is the label (from the indexed-row of the
        dataframe with the selected label_columns, as labels) .

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe with KuHar samples.
        feature_prefixes : Optional[Union[str, List[str]]]
            Which features from features must be selected. Features will be
            selected based on these prefixes. If None, select all features.
        label_columns : Union[str, List[str]]
            The columns(s) that represents the label. If the value is an `str`,
            a scalar will be returned, else, a list will be returned.
        as_array : bool
            If true, return a `np.ndarray`, else return a `pd.Series`, for each
            sample.

        Examples
        ----------
        >>> train_csv = pd.read_csv(my_filepath)
        >>> # This will select the accelerometer (x, y, and z) from HAR dataset
        >>> train_dataset = MultiModalHARDataset(feature_prefixes=["accel-x", "accel-y", "accel-z"], label_columns="activity code")
        >>> len(train_dataset)
        10
        >>> train_dataset[0]
        (np.ndarray(0.5, 0.6, 0.7), 0)

        """

        if feature_prefixes is None:
            to_select_features = list(set(dataframe.columns) - set(label_columns))
            self.feature_windows = [
                {
                    "prefix": "all",
                    "start": 0,
                    "end": len(to_select_features),
                    "features": to_select_features,
                }
            ]
        else:
            if isinstance(feature_prefixes, str):
                feature_prefixes = [feature_prefixes]

            start = 0
            self.feature_windows = []

            for prefix in feature_prefixes:
                features = [col for col in dataframe.columns if col.startswith(prefix)]
                end = start + len(features)
                self.feature_windows.append(
                    {"prefix": prefix, "start": start, "end": end, "features": features}
                )
                start = end

            to_select_features = [
                col
                for prefix in feature_prefixes
                for col in dataframe.columns
                if col.startswith(prefix)
            ]

        super().__init__(
            dataframe,
            features_columns=to_select_features,
            label_columns=label_columns,
            as_array=as_array,
        )

    @property
    def window_slices(self) -> List[Tuple[int, int]]:
        return [(window["start"], window["end"]) for window in self.feature_windows]

    @property
    def window_names(self) -> List[str]:
        return [window["prefix"] for window in self.feature_windows]

    @property
    def num_windows(self) -> int:
        return len(self.window_slices)

    def __str__(self):
        return f"PandasMultiModalDataset: samples={len(self.data)}, features={len(self.feature_columns)}, no. window={self.num_windows}"


