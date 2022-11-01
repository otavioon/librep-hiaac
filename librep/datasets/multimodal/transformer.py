from typing import Tuple, List

import numpy as np

from librep.base.transform import Transform
from librep.config.type_definitions import ArrayLike

from .multimodal import MultiModalDataset, ArrayMultiModalDataset

class WindowedTransform:

    def __init__(
        self,
        transform: Transform,
        fit_on: str = "all",  # can be window or None
        transform_on: str = "window",
        select_windows: List[str] = None,
        keep_remaining_windows: bool = True
    ):
        self.the_transform = transform
        self.fit_on = fit_on
        self.transform_on = transform_on
        self.select_windows = select_windows
        self.keep_remaining_windows = keep_remaining_windows

        assert self.fit_on in ["all", "window", None]
        assert self.transform_on in ["all", "window"]

        if self.fit_on == "window":
            assert self.transform_on == "window"


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
        if not isinstance(self.transforms, list):
            self.transforms = [self.transforms]
        self.collate_fn = collate_fn
        self.new_window_name_prefix = new_window_name_prefix

    def __transform_sample(
        self,
        transform: Transform,
        X: ArrayLike,
        y: ArrayLike,
        slices: List[Tuple[int, int]],
        do_fit: bool,
    ):
        if do_fit:
            return [
                transform.fit_transform(X[..., start:end], y)
                for start, end in slices
            ]
        else:
            return [
                transform.transform(X[..., start:end]) for start, end in slices
            ]

    def split(self, dataset: MultiModalDataset, window_names: List[str]):
        new_X, new_slices, new_names  = [], [], []
        i = 0
        for w in window_names:
            index = dataset.window_names.index(w)
            start, stop = dataset.window_slices[index]
            x = dataset[:][0][..., start:stop]
            new_X.append(x)
            new_slices.append( (i, i+(stop-start)) )
            new_names.append(w)
            i += (stop-start)
        return ArrayMultiModalDataset(self.collate_fn(new_X), y=dataset[:][1], window_slices=new_slices, window_names=new_names)

    def __call__(self, dataset: MultiModalDataset):
        new_dataset = dataset
        for window_transform in self.transforms:
            if not isinstance(window_transform, WindowedTransform):
                window_transform = WindowedTransform(window_transform)

            select_windows = window_transform.select_windows or new_dataset.window_names
            selected_dataset = self.split(new_dataset, select_windows)
            X = selected_dataset[:][0]
            y = selected_dataset[:][1]

            # Combinations:
            # fit_on=None, transform_on=window *
            # fit_on=None, transform_on=all *
            # fit_on=window, transform_on=window *
            # fit_on=window, transform_on=all    (does not make sense)
            # fit_on=all, transform_on=window *
            # fit_on=all, transform_on=all *

            if window_transform.fit_on == "all":
                X, y = selected_dataset[:][0], selected_dataset[:][1]
                window_transform.the_transform.fit(X, y)
            elif window_transform.fit_on is None:
                pass

            if window_transform.transform_on == "window":
                # fit_on=None, transform_on=window *
                # fit_on=all, transform_on=window *
                # fit_on=window, transform_on=window *
                new_X = self.__transform_sample(
                    transform=window_transform.the_transform,
                    X=X,
                    y=y,
                    slices=selected_dataset.window_slices,
                    do_fit=window_transform.fit_on == "window",
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
                new_y = np.array(new_y)
                new_slices = window_slices
                new_names = selected_dataset.window_names

                # Create a new dataset
                # new_dataset = ArrayMultiModalDataset(new_X, np.array(new_y),
                #                                      window_slices,
                #                                      new_dataset.window_names)

            else:
                # fit_on=all, transform_on=all *
                # fit_on=None, transform_on=all *
                new_X = window_transform.the_transform.transform(X=X)
                new_y = y
                new_slices = selected_dataset.window_slices
                new_names = selected_dataset.window_names

            selected_dataset = ArrayMultiModalDataset(
                    new_X, new_y, new_slices, new_names
            )

            not_selected_windows = [w for w in new_dataset.window_names if w not in select_windows]
            if not_selected_windows:
                if window_transform.keep_remaining_windows:
                    not_selected_dataset = self.split(new_dataset, not_selected_windows)
                    new_dataset = combine_multi_modal_datasets(selected_dataset, not_selected_dataset)
                else:
                    new_dataset = selected_dataset
            else: 
                new_dataset = selected_dataset

        window_names = [
            f"{self.new_window_name_prefix}{name}"
            for name in new_dataset.window_names
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
