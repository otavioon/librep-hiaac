import bisect

from typing import Any, Sequence, List, Iterable
from librep.config.type_definitions import ArrayLike, PathLike


# Simple wrap around torch.utils.data.Dataset.
# We implement the same interface
# Borrowed form Pytorch

class Dataset:
    """An abstract class representing a generic map-style dataset which
    implement the getitem and len protocols. All datasets that represent a map
    from keys to data samples should subclass it.

    Datasets subclassed from this class can be acessed using the subscription
    syntax, such as `dataset[index]`. It usually return a tuple where the first
    element represent the sample and the second, the label.

    All map-style datasets must be implement the len protocol which returns the
    number of samples in the dataset.
    """


    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __add__(self, other: 'Dataset') -> 'ConcatDataset':
        return ConcatDataset([self, other])


class Subset(Dataset):
    """Subset of a dataset at specified indices..

    Parameters
    ----------
    dataset : Dataset
        The dataset.
    indices : Sequence[int]
        Indices selected for subset.
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class IterableDataset(Dataset):
    def __iter__(self):
        raise NotImplementedError

    def __add__(self, other: Dataset):
        return ChainDataset([self, other])


class ChainDataset(IterableDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            for i in range(len(d)):
                print(d)
                yield d[i]

    def __len__(self):
        total = 0
        for d in self.datasets:
            total += len(d)  # type: ignore[arg-type]
        return total


class ConcatDataset(Dataset):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class SimpleDataset(Dataset):
    """A simple dataset implementation that return samples from an array.

    Parameters
    ----------
    X : ArrayLike
        An array-like data.
    y : ArrayLike
        An array-like set of labels.
    """
    def __init__(self, X: ArrayLike, y: ArrayLike):
        self.X = X
        self.y = y
        self._it = 0

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self) -> int:
        return len(self.X)
