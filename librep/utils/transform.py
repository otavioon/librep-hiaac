from typing import List

import numpy as np

from librep.base.transform import Transform
from librep.base.data import SimpleDataset
from librep.config.type_definitions import ArrayLike


class WindowedTransform:
    def __init__(
        self,
        transform: Transform,
        window_size: int,
        overlap: int = 0,
        start: int = 0,
        end: int = None,
        collate_fn: callable = np.concatenate,
    ):
        self.transform = transform
        self.window_size = window_size
        self.overlap = overlap
        self.start = start
        self.end = end
        self.collate_fn = collate_fn
        assert overlap < window_size, "Overlap must be less than window size"

    def fit_transform(
        self, X: ArrayLike, y: ArrayLike = None, **fit_params
    ) -> ArrayLike:
        end = self.end or len(X)
        datas = []
        for i in range(self.start, end, self.window_size - self.overlap):
            data = X[i:i + self.window_size]
            data = self.transform.fit_transform(data, y, **fit_params)
            datas.append(data)

        return self.collate_fn(datas)


class TransformDataset:
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def __call__(self, dataset):
        new_X, new_y = [], []
        for i in range(len(dataset)):
            x, y = dataset[i]
            for transform in self.transforms:
                x = transform.fit_transform(x, y)
            new_X.append(x)
            new_y.append(y)
        return SimpleDataset(new_X, new_y)