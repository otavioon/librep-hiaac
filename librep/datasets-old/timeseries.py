from typing import Tuple, Iterable, Optional, Number, List

import numpy as np

from librep.datasets.dataset import SimpleLabeledDataset

import matplotlib.pyplot as plt


# TODO rate or timestamp?
class TimeSeriesSample:
    def __init__(self, points: Iterable[Number],
                 timestamps: Optional[Iterable[Number]] = None):
        self._points = points
        self._timestamps = timestamps
        if timestamps is not None:
            assert len(points) == len(timestamps), \
                "Points and timestamps must have the same length"
        # else:
        #     warnings.warn("Creating a time series without timestamps")

    def rate(self) -> Tuple[float, float]:
        if self._timestamps is None:
            raise ValueError("No timestamps were provided")

        times_diff = np.diff(self._timestamps)
        return np.average(times_diff), np.std(times_diff)

    @property
    def data(self):
        return self._points

    @property
    def points(self):
        return self._points

    @property
    def timestamps(self):
        return self._timestamps

    def __getitem__(self, index):
        return self.points[index]

    def __len__(self) -> int:
        return len(self.points)


class TimeSeriesSample3D:
    def __init__(self, x: TimeSeriesSample,
                 y: TimeSeriesSample,
                 z: TimeSeriesSample):
        self.x = x
        self.y = y
        self.z = z

    def ravel(self, axis: int = 0):
        return np.stack(
            [self.x.points, self.y.points, self.z.points],
            axis=axis).ravel()

    def ravel_timestamps(self, axis: int = 0):
        return np.stack(
            [self.x.timestamps, self.y.timestamps, self.z.timestamps],
            axis=axis).ravel()

    def norm(self):
        # TODO must check rate
        return np.sqrt(
            self.x.data * self.x.data +
            self.y.data * self.y.data +
            self.z.data * self.z.data)


class MotionSensorSample:
    def __init__(self, accelerometer: TimeSeriesSample3D,
                 gyroscope: TimeSeriesSample3D):
        self.accelerometer = accelerometer
        self.gyroscope = gyroscope

    @property
    def acc(self):
        return self.accelerometer

    @property
    def gyr(self):
        return self.gyroscope


class MotionSensorDataset(SimpleLabeledDataset):
    def __init__(self, data: Iterable[MotionSensorSample], labels: Iterable):
        self.data = data
        self.labels = labels


# ----------- Serialization Functions -----------
class MotionSensorDatasetSerializer:
    @staticmethod
    def load(file):
        raise NotImplementedError

    @staticmethod
    def save(obj: MotionSensorDataset, file):
        raise NotImplementedError


# ----------- Visualization Functions -----------
def display_sample(samples: List[MotionSensorSample], label_map: dict = None,
                   sharey: bool = True, figsize: tuple = (8, 8)):
    if isinstance(samples, list):
        samples = [samples]

    nsamples = len(samples)

    fig, axs = plt.subplots(
        nrows=2, ncols=nsamples, figsize=(figsize[0]*nsamples, figsize[1])
    )

    def plot_XYZ_chart(ax, sample: TimeSeriesSample3D,
                       label_prefix: str = "", title: str = ""):
        ax.set_title(title)
        ax.plot(sample.x.data, color="r", label=f"{label_prefix}.x")
        ax.plot(sample.y.data, color="b", label=f"{label_prefix}.y")
        ax.plot(sample.z.data, color="g", label=f"{label_prefix}.z")
        ax.legend()

    for i, sample in enumerate(nsamples):
        raise
