from typing import Any, Tuple, Iterable, List, Sequence

import pandas as pd


class Dataset:
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class SimpleLabeledDataset(Dataset):
    def __init__(self, data: Iterable, labels: Iterable):
        self.data = data
        self.labels = labels
        assert len(data) == len(labels), \
            "Labels and samples must have the same length"

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def X(self):
        return self.data

    @property
    def y(self):
        return self.labels


class PandasDataset(SimpleLabeledDataset):
    def __init__(self, data: pd.DataFrame,
                 data_columns: List[str],
                 label_column: str,
                 is_regex: bool = False):
        if isinstance(data_columns, list):
            raise ValueError("Data columns must be an list of strings")
        if is_regex:
            raise NotImplementedError(
                "Get data by regex is not implemented yet."
            )
        self.data_columns = data_columns
        self.label_column = label_column

        self.data = data[data_columns].values
        self.labels = data[[label_column]].values


class CanonicalDataset:
    def to_canonical(self):
        raise NotImplementedError