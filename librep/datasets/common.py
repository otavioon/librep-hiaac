from typing import Dict, Tuple, Optional, Union, List

import pandas as pd
import numpy as np

from librep.base.data import Dataset
from librep.base.transform import Transform
from librep.utils.dataset import PandasDataset
from librep.config.type_definitions import ArrayLike


class HARDatasetGenerator:
    def create_datasets(
        self,
        train_size: float,
        validation_size: float,
        test_size: float,
        ensure_distinct_users_per_dataset: bool = True,
        balance_samples: bool = True,
        activities_remap: Dict[int, int] = None,
        seed: int = None,
        use_tqdm: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create train/validation/test datasets.

        Parameters
        ----------
        train_size : float
            Fraction of samples to training dataset.
        validation_size : float
            Fraction of samples to validation dataset.
        test_size : float
            Fraction of samples to test dataset.
        ensure_distinct_users_per_dataset : bool
            If True, ensure that samples from an user do not belong to distinct
            datasets (the default is True).
        balance_samples : bool
            If True, the datasets will have the same number of samples per
            class. The number of samples will be reduced to the class with the
            minor number of samples (the default is True).
        activities_remap : Dict[int, int]
            A dictionary used to replace a label from one class to another.
        seed : int
            The random seed (the default is None).
        use_tqdm : bool
            If must use tqdm as iterator (the default is True).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple with the train, validation and test dataframes.

        """
        raise NotImplementedError


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


class TransformMultiModalDataset:
    """Apply a list of transforms into the whole dataset, generating a new
    dataset.

    Parameters
    ----------
    transforms : List[Transform]
        List of transforms to be applyied to each sample, in order.

    Note: It supposes the number of windows will remain the same

    TODO: it not using fit. fit should be called over whole dataset.
    """

    def __init__(
        self, transforms: List[Transform], collate_fn: callable = np.concatenate,
        new_window_name_prefix: str = ""
    ):
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.new_window_name_prefix = new_window_name_prefix

    def __transform_sample(
        self, transform: Transform, x: ArrayLike, slices: List[Tuple[int, int]]
    ):
        return [transform.transform(x[start:end]) for start, end in slices]

    def __call__(self, dataset: MultiModalDataset):
        new_dataset = dataset
        for transform in self.transforms:
            new_X, new_y = [], []
            # Transform each sample window
            for i in range(len(new_dataset)):
                x, y = new_dataset[i]
                new_X.append(
                    self.__transform_sample(transform, x, new_dataset.window_slices)
                )
                new_y.append(y)

            # Calculate new slices
            window_slices = []
            start = 0
            for x in new_X[0]:
                end = start + len(x)
                window_slices.append((start, end))
                start = end

            # Collate the windows into a single array
            new_X = [self.collate_fn(xs) for xs in new_X]

            window_names = [f"{self.new_window_name_prefix}{name}" for name in new_dataset.window_names]
            # Create a new dataset
            new_dataset = ArrayMultiModalDataset(
                new_X, np.array(new_y), window_slices, window_names
            )
        return new_dataset


def __label_selector(a, b):
    return a

def combine_multi_modal_datasets(d1: MultiModalDataset, d2: MultiModalDataset, collate_fn: callable = np.hstack, labels_combine: callable = __label_selector):
    new_X = collate_fn([d1[:][0], d2[:][0]])
    new_y = [labels_combine(y1, y2) for y1, y2 in zip(d1[:][1], d2[:][1])]

    last_slice_index = d1.window_slices[-1][1]
    window_slices = d1.window_slices + [(start+last_slice_index, end+last_slice_index) for start, end in d2.window_slices]
    window_names = d1.window_names + d2.window_names

    return ArrayMultiModalDataset(new_X, new_y, window_slices, window_names)