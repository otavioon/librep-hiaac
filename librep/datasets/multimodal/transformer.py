from typing import Tuple, List

import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

from .multimodal import MultiModalDataset, ArrayMultiModalDataset


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
        self,
        transforms: List[Transform],
        collate_fn: callable = np.hstack,
        new_window_name_prefix: str = "",
    ):
        self.transforms = transforms
        self.collate_fn = collate_fn
        self.new_window_name_prefix = new_window_name_prefix

    def __transform_sample(
        self,
        transform: Transform,
        X: ArrayLike,
        y: ArrayLike,
        slices: List[Tuple[int, int]],
    ):
        return [
            transform.fit_transform(X[..., start:end], y)
            for start, end in slices
        ]

    def __call__(self, dataset: MultiModalDataset):
        new_dataset = dataset
        for transform in self.transforms:
            X = new_dataset[:][0]
            y = new_dataset[:][1]
            new_X = self.__transform_sample(
                transform=transform,
                X=X,
                y=y,
                slices=new_dataset.window_slices,
            )
            new_y = y

            # Calculate new slices
            window_slices = []
            start = 0
            for x in new_X:
                end = start + len(x[0])
                window_slices.append((start, end))
                start = end

            # Collate the windows into a single array
            new_X = self.collate_fn(new_X)

            # Create a new dataset
            new_dataset = ArrayMultiModalDataset(
                new_X, np.array(new_y), window_slices, new_dataset.window_names
            )
        window_names = [
            f"{self.new_window_name_prefix}{name}" for name in new_dataset.window_names
        ]
        new_dataset._window_names = window_names
        return new_dataset


def __label_selector(a, b):
    return a


def combine_multi_modal_datasets(
    d1: MultiModalDataset,
    d2: MultiModalDataset,
    collate_fn: callable = np.hstack,
    labels_combine: callable = __label_selector,
):
    new_X = collate_fn([d1[:][0], d2[:][0]])
    new_y = [labels_combine(y1, y2) for y1, y2 in zip(d1[:][1], d2[:][1])]

    last_slice_index = d1.window_slices[-1][1]
    window_slices = d1.window_slices + [
        (start + last_slice_index, end + last_slice_index)
        for start, end in d2.window_slices
    ]
    window_names = d1.window_names + d2.window_names

    return ArrayMultiModalDataset(new_X, new_y, window_slices, window_names)