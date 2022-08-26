from typing import Tuple, Any, List, Union

import pandas as pd

from librep.base.data import Dataset
from librep.config.type_definitions import ArrayLike


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

    def __str__(self) -> str:
        return f"PandasDataset: samples={len(self.data)}, features={len(self.feature_columns)}, label_column='{self.label_columns}'"

    def __repr__(self) -> str:
        return str(self)
