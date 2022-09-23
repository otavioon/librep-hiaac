from typing import List
from pathlib import Path

import pandas as pd

from librep.config.type_definitions import ArrayLike
from librep.utils.dataset import PandasDatasetsIO
from librep.datasets.multimodal import PandasMultiModalDataset


class PandasMultiModalLoader:
    url = ""
    description = ""
    feature_columns = []
    label = ""

    def __init__(
        self, root_dir: ArrayLike, download: bool = False, version: str = "v1"
    ):
        self.root_dir = Path(root_dir)
        self.version = version
        if download:
            self.download()

    def download(self):
        # download from self.url and extract
        raise NotImplementedError

    def load(
        self,
        load_train: bool = True,
        load_validation: bool = True,
        load_test: bool = True,
        concat_train_validation: bool = False,
        concat_all: bool = False,
        features: List[str] = None,
        label: str = None,
        as_array: bool = True,
    ):
        """Load the dataset and returns a 3-element tuple:

        - The first element is the train PandasMultiModalDataset
        - The second element is the validation PandasMultiModalDataset
        - The third element is the test PandasMultiModalDataset

        .. note: The number of elements return may vary depending on the arguments.
        .. note: Assumed that all views have the train file.

        Parameters
        ----------
        load_train : bool
            If must load the train dataset (the default is True).
        load_validation : bool
            If must load the validation dataset (the default is True).
        load_test : bool
            If must load the test dataset (the default is True).
        concat_train_validation : bool
             If must concatenate the train and validation datasts, returning a single
             train PandasMultiModalDataset. This will return a 2-element tuple
             with train and test PandasMultiModalDataset (the default is False).
        concat_all : bool
             If must concatenate the train, validation and test datasts, returning a
             single PandasMultiModalDataset. This will return a 1-element tuple
             with a single PandasMultiModalDataset. Note that this option is
             exclusive with `concat_train_validation` (the default is False).
        features : List[str].
            List of strings to use as feature_prefixes (the default is None).
        label : str
            Name of the label column (the default is None).
        as_array : bool
            If the PandasMultiModalDataset must return an array when elements are
            accessed (the default is True).

        Returns
        -------
        Tuple
            A tuple with datasets.

        """
        if concat_train_validation and concat_all:
            raise ValueError(
                "concat_all and concat_train_validation options are mutually exclusive"
            )

        train, validation, test = PandasDatasetsIO(self.root_dir).load(
            load_train=load_train, load_validation=load_validation, load_test=load_test
        )

        if train is None and validation is None and test is None:
            raise ValueError("Train, validation and test not loaded")

        default_columns = []
        for df in [train, validation, test]:
            if df is not None:
                default_columns = df.columns.values.tolist()
                break

        if train is None:
            train = pd.DataFrame([], columns=default_columns)
        if validation is None:
            validation = pd.DataFrame([], columns=default_columns)
        if test is None:
            validation = pd.DataFrame([], columns=default_columns)

        if concat_train_validation:
            train = pd.concat([train, validation], ignore_index=True)
        elif concat_all:
            train = pd.concat([train, validation, test], ignore_index=True)

        features = features or self.feature_columns
        label = label or self.label

        train_dataset = PandasMultiModalDataset(
            train, feature_prefixes=features, label_columns=label, as_array=as_array
        )

        validation_dataset = PandasMultiModalDataset(
            validation,
            feature_prefixes=features,
            label_columns=label,
            as_array=as_array,
        )

        test_dataset = PandasMultiModalDataset(
            test, feature_prefixes=features, label_columns=label, as_array=as_array
        )

        if concat_all:
            return (train_dataset,)
        elif concat_train_validation:
            return (train_dataset, test_dataset)
        return (train_dataset, validation_dataset, test_dataset)

    def readme(self, filename: str = "README.md"):
        path = self.root_dir / filename
        with path.open("r") as f:
            return f.read()

    def print_readme(self, filename: str = "README.md"):
        from IPython.display import display, Markdown

        display(Markdown(self.readme(filename)))


class KuHarResampledView20HZ(PandasMultiModalLoader):
    url = "https://drive.google.com/file/d/1-fAzWQQ8jV8oSfxLSfzKA0eiJE66qgD9/view?usp=sharing"
    description = "KuHar Balanced View Resampled to 20HZ"
    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"


class MotionSenseResampledView20HZ(PandasMultiModalLoader):
    url = "https://drive.google.com/file/d/16d-16fyhKc8D-_ocATFdYsNhSwoUY4ft/view?usp=sharing"
    description = "MotionSense Balanced View Resampled to 20HZ"
    feature_columns = [
        "userAcceleration.x",
        "userAcceleration.y",
        "userAcceleration.z",
        "rotationRate.x",
        "rotationRate.y",
        "rotationRate.z",
    ]
    label = "activity code"


class CHARMUnbalancedView(PandasMultiModalLoader):
    url = "https://drive.google.com/file/d/1e_HgeGYmfWmv4_1ZdZRJuda9QuoNak2E/view?usp=sharing"
    description = "CHARM Unbalanced View (default is 20HZ)"
    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"

    def load(
        self,
        load_train: bool = True,
        load_validation: bool = False,
        load_test: bool = True,
        concat_train_validation: bool = False,
        concat_all: bool = False,
        features: List[str] = None,
        label: str = None,
        as_array: bool = True,
    ):
        if load_validation:
            raise ValueError("View does not have validation files")
        return super().load(
            load_train,
            load_validation,
            load_test,
            concat_train_validation,
            concat_all,
            features or self.feature_columns,
            label or self.label,
            as_array
        )


class WISDMInterpolatedUnbalancedView(PandasMultiModalLoader):
    url = "https://drive.google.com/file/d/1Mjn5hrY_Fke76aKJujhRj4z8ak63dDIT/view?usp=sharing"
    description = "WISDM Interpolated Unbalanced View (default is 20HZ)"
    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"

    def load(
        self,
        load_train: bool = True,
        load_validation: bool = False,
        load_test: bool = True,
        concat_train_validation: bool = False,
        concat_all: bool = False,
        features: List[str] = None,
        label: str = None,
        as_array: bool = True,
    ):
        if load_validation:
            raise ValueError("View does not have validation files")
        return super().load(
            load_train,
            load_validation,
            load_test,
            concat_train_validation,
            concat_all,
            features or self.feature_columns,
            label or self.label,
            as_array
        )


class UCIHARUnbalancedView(PandasMultiModalLoader):
    url = "https://drive.google.com/file/d/11rflAgjcFvuH3f0fQCRW3yWhao6wLNY2/view?usp=sharing"
    description = "UCI-HAR Unbalanced View (default is 20HZ)"
    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"

    def load(
        self,
        load_train: bool = True,
        load_validation: bool = False,
        load_test: bool = True,
        concat_train_validation: bool = False,
        concat_all: bool = False,
        features: List[str] = None,
        label: str = None,
        as_array: bool = True,
    ):
        if load_validation:
            raise ValueError("View does not have validation files")
        return super().load(
            load_train,
            load_validation,
            load_test,
            concat_train_validation,
            concat_all,
            features or self.feature_columns,
            label or self.label,
            as_array
        )
