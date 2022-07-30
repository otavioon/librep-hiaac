from typing import Any, Sequence, List
from librep.config.type_definitions import PathLike


# Simple wrap around torch.utils.data.Dataset.
# We implement the same interface
class Dataset:
    description: str = ""
    alias: str = ""

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Subset(Dataset):

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)
