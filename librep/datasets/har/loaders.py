from typing import List, Union
from pathlib import Path
import shutil

import pandas as pd

from librep.config.type_definitions import ArrayLike
from librep.utils.dataset import PandasDatasetsIO
from librep.datasets.multimodal import PandasMultiModalDataset
from librep.utils.file_ops import (
    Downloader,
    GoogleDriveDownloader,
    Checksum,
    MD5Checksum,
    Extractor,
    ZipExtractor,
    DownloaderExtractor
)


class PandasMultiModalLoader:
    url: str = ""
    description: str = "Loader"

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename: str = "file"
    zip_checksum: str = None
    extracted_zip_root_dir: str = ""

    train_file = "train.csv"
    validation_file = "validation.csv"
    test_file = "test.csv"

    
    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    default_label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {}
    standard_activity_codes = {}

    def __init__(self, root_dir: ArrayLike, download: bool = False, version: int = 1):
        self.root_dir = Path(root_dir)
        self.version = version
        if download:
            self.download()

    def download(self):
        # download from self.url and extract
        downloader = DownloaderExtractor(
            downloader_cls=self.downloader_cls,
            extractor_cls=self.extractor_cls,
            checker_cls=self.checksum_cls
        )
        self.root_dir.mkdir(exist_ok=False, parents=True)
        downloader.download_extract_check(
            url=self.url,
            destination_download_file=self.zip_filename,
            checksum=self.zip_checksum,
            extract_folder=self.root_dir,
            remove_on_check_error=True,
            remove_downloads=True
        )

        if self.extracted_zip_root_dir:
            source_path = self.root_dir / self.extracted_zip_root_dir
            for src_file in source_path.glob('*.*'):
                shutil.move(str(src_file), str(self.root_dir))
            source_path.rmdir()

        print(f"Data downloaded and extracted to {self.root_dir}")

    def load(
        self,
        load_train: bool = None,
        load_validation: bool = None,
        load_test: bool = None,
        concat_train_validation: bool = False,
        concat_all: bool = False,
        as_multimodal: bool = True,
        # Multimodal kwargs
        features: List[str] = None,
        label: Union[str, List[str]] = None,
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
            If None, use pre-defined list of feature prefixes.
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
        # Check concatenation values
        if concat_train_validation and concat_all:
            raise ValueError(
                "concat_all and concat_train_validation options are "
                + "mutually exclusive options."
            )

        # Check load parameters
        if load_train is None:
            load_train = self.train_file or False
        if load_validation is None:
            load_validation = self.validation_file or False
        if load_test is None:
            load_test = self.test_file or False
        # If all elements are False, raise Error
        if not load_train and not load_validation and not load_test:
            raise ValueError("Nothing to load. All load parameters are False")

        # Load files
        loader = PandasDatasetsIO(
            self.root_dir,
            train_filename=self.train_file,
            validation_filename=self.validation_file,
            test_filename=self.test_file,
        )
        train, validation, test = loader.load(
            load_train=load_train, load_validation=load_validation, load_test=load_test
        )

        # Check if anything is loaded
        if train is None and validation is None and test is None:
            raise ValueError("Nothing loaded (train, validation or test)")

        # Take the column names from one of the sets (train or val or test)
        # It is supposed that all dataframes have the same columns
        default_columns = []
        for df in [train, validation, test]:
            if df is not None:
                default_columns = df.columns.values.tolist()
                break

        # Create empty dataframes from some ones that were None
        if train is None:
            train = pd.DataFrame([], columns=default_columns)
        if validation is None:
            validation = pd.DataFrame([], columns=default_columns)
        if test is None:
            test = pd.DataFrame([], columns=default_columns)

        # Concat train + validation
        if concat_train_validation:
            train = pd.concat([train, validation], ignore_index=True)
        # Concat train + validation + test
        elif concat_all:
            train = pd.concat([train, validation, test], ignore_index=True)

        # If not multimodal, datasets are the csvs only
        if not as_multimodal:
            train_dataset = train
            validation_dataset = validation
            test_dataset = test
        else:
            features = features or self.feature_columns
            label = label or self.default_label

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
            return train_dataset
        elif concat_train_validation:
            return (train_dataset, test_dataset)
        else:
            return (train_dataset, validation_dataset, test_dataset)

    def readme(self, filename: str = "README.md"):
        path = self.root_dir / filename
        with path.open("r") as f:
            return f.read()

    def print_readme(self, filename: str = "README.md"):
        from IPython.display import display, Markdown

        display(Markdown(self.readme(filename)))

    def __str__(self) -> str:
        return f"Loader: {self.description}"

    def __repr__(self) -> str:
        return str(self)


class KuHar_BalancedView20HzMotionSenseEquivalent(PandasMultiModalLoader):
    url: str = "19fyh0RyH5Q1N5uimHwGr80or31EkiQaj"
    description = (
        "KuHar Balanced View Resampled to 20HZ with classes " +
        "equivalent to MotionSense"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "KuHar-balanced_20Hz_motionsense_equivalent-v1.zip"
    zip_checksum: str = "e35af827fc1153753ec6d11a4b3a4654"
    extracted_zip_root_dir: str = "balanced_20Hz_motionsense_equivalent-v1"

    train_file = "train.csv"
    validation_file = "validation.csv"
    test_file = "test.csv"

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {
        0: "stair down",
        1: "stair up",
        2: "sit",
        3: "stand",
        4: "walk",
        5: "run"
    }

    standard_activity_codes = {
        0: "sit",
        1: "stand",
        2: "walk",
        3: "stair up",
        4: "stair down",
        5: "run"
    }


class MotionSense_BalancedView20HZ(PandasMultiModalLoader):
    url: str = "19fyh0RyH5Q1N5uimHwGr80or31EkiQaj"
    description = (
        "MoationSense Balanced View Resampled to 20HZ"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "MotionSense-balanced_20Hz-v1.zip"
    zip_checksum: str = "de946e65a0452421e3c716ddf3dcf57d"
    extracted_zip_root_dir: str = "balanced_20Hz-v1"

    train_file = "train.csv"
    validation_file = "validation.csv"
    test_file = "test.csv"

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {
        0: "downstairs",
        1: "upstairs",
        2: "sitting",
        3: "stand",
        4: "walking",
        5: "jogging"
    }

    standard_activity_codes = {
        0: "sit",
        1: "stand",
        2: "walk",
        3: "stair up",
        4: "stair down",
        5: "run"
    }


class ExtraSensorySense_UnbalancedView20HZ(PandasMultiModalLoader):
    url: str = "1xNkKw40kAy3M3zP9FA4kNZwJbA-iHoJH"
    description = (
        "ExtraSensory UnBalanced View Resampled to 20HZ (train only)"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "ExtraSensory-unbalanced_20Hz_train-v1.zip"
    zip_checksum: str = "8cd977b11fee03a60344b9f8a2fb5357"
    extracted_zip_root_dir: str = "unbalanced_20Hz_train-v1"

    train_file = "train.csv"
    validation_file = None
    test_file = None

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {
        0: "sitting",
        1: "or_standing",
        2: "fix_walking",
        3: "fix_running",
    }

    standard_activity_codes = {
        1: "stand",
        2: "walk",
        5: "run"
    }


class CHARM_BalancedView20Hz(PandasMultiModalLoader):
    url: str = "143X30U6rJL0Ffuy-CvPDOjC6NvKQiu4y"
    description = (
        "CHARM Balanced View"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "CHARM-balanced_20Hz_train_test-v2.zip"
    zip_checksum: str = "812e5585ee6de86205c0844e07948e65"
    extracted_zip_root_dir: str = "balanced_20Hz_train_test-v2"

    train_file = "train.csv"
    validation_file = None
    test_file = "test.csv"

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {
        0: "sitting on a chair",
        1: "sitting on a couch",
        2: "standing",
        6: "walking",
        7: "running",
        8: "walking upstairs",
        9: "walking downstairs"
    }

    standard_activity_codes = {
        0: "sit",
        1: "stand",
        2: "walk",
        3: "stair up",
        4: "stair down",
        5: "run"
    }


class WISDM_UnbalancedView20Hz(PandasMultiModalLoader):
    url: str = "1qs6xUCBEbTl_oEIKq_KUbPZaK9NLK9FS"
    description = (
        "WISDM Balanced View Resampled to 20Hz"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "WISDM-unbalanced_20Hz_train_test-v1.zip"
    zip_checksum: str = "3b689ef119546ac77722b213bf44a4e2"
    extracted_zip_root_dir: str = "unbalanced_20Hz_train_test-v1"

    train_file = "train.csv"
    validation_file = None
    test_file = "test.csv"

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = "standard activity code"
    activity_codes = {
        0: "walking",
        1: "jogging",
        2: "stairs",
        3: "sitting",
        4: "standing"
    }

    standard_activity_codes = {
        0: "sit",
        1: "stand",
        2: "walk",
        6: "stair up and down"
    }


class UCIHAR_UnbalancedView20Hz(PandasMultiModalLoader):
    url: str = "1KIdwMdTA7QXnbP721ynus7eOxjwNiMaH"
    description = (
        "UCI-HAR Unbalanced View Resampled to 20Hz"
    )

    downloader_cls: Downloader = GoogleDriveDownloader
    extractor_cls: Extractor = ZipExtractor
    checksum_cls: Checksum = MD5Checksum
    zip_filename = "UCI-HAR-unbalanced_20Hz_train_test-v1.zip"
    zip_checksum: str = "c8122d953a3116c3b92b76a2d25d7f04"
    extracted_zip_root_dir: str = "unbalanced_20Hz_train_test-v1"

    train_file = "train.csv"
    validation_file = None
    test_file = "test.csv"

    feature_columns = ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
    label = "activity code"
    standard_label = {
        1: "walking",
        2: "walking upstairs",
        3: "walking downstairs",
        4: "sitting",
        5: "standing"
    }

    standard_activity_codes = {
        0: "sit",
        1: "stand",
        2: "walk",
        3: "stair up",
        4: "stair down"
    }