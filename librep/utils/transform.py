# from typing import List

# import numpy as np

# from librep.base.transform import Transform
# from librep.base.data import SimpleDataset
# from librep.config.type_definitions import ArrayLike


# class WindowedTransform:
#     """Apply a transform to a sliding window of features. It is useful when a
#     single sample contains features from different natures. Each window of
#     features from the sample is divided, the transform is applyied individually,
#     and then merged using the collate_fn.
#     You can see this as an column selector from the input.

#     Parameters
#     ----------
#     transform : Transform
#         The transform to be applyied.
#     window_size : int
#         The sliding window size (in relation the the number of features).
#         The `window_size` amount of features from sample will be selected and
#         then the transform will be applyied. This is done until reaching the
#         total number of features.
#     overlap : int
#         If one sliding window must overlap with another (the default is 0).
#     start : int
#         Which feature the sliding window must start (the default is 0).
#     end : int
#         Which feature the sliding window must start end (the default is None).
#     collate_fn : callable
#         The function used to merge the features transformed from the sliding
#         windows (the default is np.concatenate).


#     """

#     def __init__(
#         self,
#         transform: Transform,
#         window_size: int,
#         overlap: int = 0,
#         start: int = 0,
#         end: int = None,
#         collate_fn: callable = np.concatenate,
#     ):
#         self.transform = transform
#         self.window_size = window_size
#         self.overlap = overlap
#         self.start = start
#         self.end = end
#         self.collate_fn = collate_fn
#         assert overlap < window_size, "Overlap must be less than window size"

#     def fit_transform(
#         self, X: ArrayLike, y: ArrayLike = None, **fit_params
#     ) -> ArrayLike:
#         end = self.end or len(X)
#         datas = []
#         for i in range(self.start, end, self.window_size - self.overlap):
#             data = X[i:i + self.window_size]
#             data = self.transform.fit_transform(data, y, **fit_params)
#             datas.append(data)

#         return self.collate_fn(datas)


# class TransformMultiModalDataset:
#     """Apply a list of transforms into the whole dataset, generating a new
#     dataset.

#     Parameters
#     ----------
#     transforms : List[Transform]
#         List of transforms to be applyied to each sample, in order.

#     """

#     def __init__(self, transforms: List[Transform]):
#         self.transforms = transforms

#     def __call__(self, dataset):
#         new_X, new_y = [], []
#         for i in range(len(dataset)):
#             x, y = dataset[i]
#             for transform in self.transforms:
#                 x = transform.fit_transform(x, y)
#             new_X.append(x)
#             new_y.append(y)
#         return SimpleDataset(new_X, new_y)
