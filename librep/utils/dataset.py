import random
from pathlib import Path
from typing import Tuple, Any, List, Union

import numpy as np
import pandas as pd

from librep.base.data import Dataset
from librep.config.type_definitions import ArrayLike, PathLike


class PandasDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, features_columns: List[str],
                 label_columns: Union[str, List[str]], as_array: bool = True):
        self.data = dataframe
        self.feature_columns = features_columns
        self.label_columns = label_columns
        self.as_array = as_array

    def __getitem__(self, index: int) -> Tuple[ArrayLike, Any]:
        data = self.data.loc[index, self.feature_columns]
        label = self.data.loc[index, self.label_columns]

        if self.as_array:
            data = data.values
            if isinstance(self.label_columns, list):
                label = label.values

        return (data, label)

    def __len__(self):
        return len(self.data)

    def save(self, filepath: PathLike):
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, compress="infer")

    def __str__(self) -> str:
        return f"PandasDataset: samples={len(self.data)}, features={len(self.feature_columns)}, label_column='{self.label_columns}'"

    def __repr__(self) -> str:
        return str(self)


class MultiFeatureDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, features_columns: List[str],
                 label_columns: Union[str, List[str]], as_array: bool = True,
                 per_sample_transform = None, collate_fn: callable = None):
        self.data = dataframe
        self.feature_columns = features_columns
        self.label_columns = label_columns
        self.as_array = as_array
        self.per_sample_transform = per_sample_transform
        self.collate_fn = collate_fn

    def __getitem__(self, index: int) -> Tuple[ArrayLike, Any]:
        datas = []

        for feature_prefix in self.feature_columns:
            features = [c.startswith(feature_prefix) for c in self.data.columns]
            data = self.data.loc[index, features]
            if self.as_array:
                data = data.values
            if self.per_sample_transform is not None:
                data = self.per_sample_transform.fit_transform(data)
            datas.append(data)

        label = self.data.loc[index, self.label_columns]

        if self.as_array and isinstance(self.label_columns, list):
            label = label.values

        if self.collate_fn is not None:
            data = self.collate_fn(datas)
        else:
            data = datas

        return (data, label)

    def __len__(self):
        return len(self.data)



# Must use DataLoader instead....
def load_full_data(dataset: Dataset,
                   return_X_y: bool = True,
                   shuffle: bool = False):
    indexes = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indexes)

    data = [dataset[index] for index in indexes]

    if not return_X_y:
        return np.array(data)
    else:
        return np.array([d[0] for d in data]), np.array([d[1] for d in data])