import random
from pathlib import Path
from typing import Tuple, Any, List

import numpy as np
import pandas as pd

from librep.base.data import Dataset
from librep.config.type_definitions import ArrayLike, PathLike


class PandasDataset(Dataset):

    def __init__(self, dataframe: pd.DataFrame, features_columns: List[str],
                 label_column: str):
        self.data = dataframe
        self.feature_columns = features_columns
        self.label_column = label_column

    def __getitem__(self, index: int) -> Tuple[ArrayLike, Any]:
        data = self.data.loc[index, self.feature_columns].values
        label = self.data.loc[index, self.label_column]
        return (data, label)

    def __len__(self):
        return len(self.data)

    def save(self, filepath: PathLike):
        filepath = Path(filepath)
        filepath.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(filepath, compress="infer")

    def __str__(self) -> str:
        return f"PandasDataset: samples={len(self.data)}, features={len(self.feature_columns)}, label_column='{self.label_column}'"

    def __repr__(self) -> str:
        return str(self)


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