import os
from typing import Dict, List, Tuple, Union, Optional, Generator
from pathlib import Path
import random

import numpy as np
import pandas as pd
import tqdm

from librep.utils.file_ops import download_unzip_check
from librep.config.type_definitions import PathLike
from librep.base.data import Dataset
from librep.utils.dataset import PandasDataset


class DatasetSplitError(Exception):
    pass


class RawKuHar:
    # Version 5 KuHar Raw
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/d3126562-b795-4eba-8559-310a25859cc7/file_downloaded"

    # Activity names and codes
    activity_names = {
        0: "Stand",
        1: "Sit",
        2: "Talk-sit",
        3: "Talk-stand",
        4: "Stand-sit",
        5: "Lay",
        6: "Lay-stand",
        7: "Pick",
        8: "Jump",
        9: "Push-up",
        10: "Sit-up",
        11: "Walk",
        12: "Walk-backwards",
        13: "Walk-circle",
        14: "Run",
        15: "Stair-up",
        16: "Stair-down",
        17: "Table-tennis",
    }

    activity_codes = {v: k for k, v in activity_names.items()}

    def __init__(self, dataset_dir: PathLike, download: bool = False):
        self.dataset_dir = Path(dataset_dir)
        if download:
            self.__download_and_extract()
        self.metadata_df = self.__read_metadata()

    def __download_and_extract(self):
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        file_path = self.dataset_dir / "kuhar.zip"
        download_unzip_check(
            url=RawKuHar.dataset_url,
            download_destination=file_path,
            unzip_dir=self.dataset_dir,
        )

    def __read_metadata(self) -> pd.DataFrame:
        # Let's list all CSV files in the directory
        files = self.dataset_dir.rglob("*.csv")
        # files = glob.glob(os.path.join(self.dataset_dir, "*", "*.csv"))

        # And create a relation of each user, activity and CSV file
        users_relation = []
        for f in files:
            # Pick activity name (folder name, e.g.: 5.Lay)
            activity_name = f.parents[0].name
            # Pick CSV file name (e.g.: 1052_F_1.csv)
            # Split activity number and name (e.g.: [5, 'Lay'])
            act_no, act_name = activity_name.split(".")
            act_no = int(act_no)
            # Split user code, act type and serial number (e.g.: [1055, 'G', 1])
            csv_splitted = f.stem.split("_")
            user = int(csv_splitted[0])
            serial = "_".join(csv_splitted[2:])
            # Generate a tuple with the information and append to the relation's list
            users_relation.append(
                {
                    "activity code": act_no,
                    "user": user,
                    "serial": serial,
                    "file": str(f),
                }
            )

        # Create a dataframe with all meta information
        column_dtypes = [
            ("activity code", np.int),
            ("user", np.int),
            ("serial", np.int),
            ("file", str),
        ]
        metadata_df = pd.DataFrame(
            users_relation, columns=[d[0] for d in column_dtypes]
        )
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df

    def read_information(
        self,
        file: PathLike,
        activity_code: int,
        user: int,
        serial: Optional[int] = None,
    ) -> pd.DataFrame:
        file = Path(file)
        # Default feature names from this dataset
        feature_dtypes = {
            "accel-start-time": np.float,
            "accel-x": np.float,
            "accel-y": np.float,
            "accel-z": np.float,
            "gyro-start-time": np.float,
            "gyro-x": np.float,
            "gyro-y": np.float,
            "gyro-z": np.float,
        }

        with file.open("r") as f:
            csv_matrix = pd.read_csv(
                f, names=list(feature_dtypes.keys()), dtype=feature_dtypes
            )
            # Reordering to same format as all Ku-Har datasets
            csv_matrix = csv_matrix[
                [
                    "accel-x",
                    "accel-y",
                    "accel-z",
                    "gyro-x",
                    "gyro-y",
                    "gyro-z",
                    "accel-start-time",
                    "gyro-start-time",
                ]
            ]
            csv_matrix["accel-end-time"] = csv_matrix["accel-start-time"]
            csv_matrix["gyro-end-time"] = csv_matrix["gyro-start-time"]
            csv_matrix["activity code"] = activity_code
            csv_matrix["length"] = 1
            csv_matrix["serial"] = serial
            csv_matrix["index"] = range(len(csv_matrix))
            csv_matrix["user"] = user
            return csv_matrix

    @property
    def users(self):
        return np.sort(self.metadata_df["user"].unique()).tolist()

    @property
    def activities(self):
        return np.sort(self.metadata_df["activity code"].unique()).tolist()

    def __str__(self):
        return f"KuHar Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files)"

    def __repr__(self):
        return f"KuHar Dataset at: '{self.dataset_dir}'"


class TrimmedRawKuHar(RawKuHar):
    # 2. Trimmed Raw Dataset v5
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded"


class RawKuHarIterator:
    def __init__(
        self,
        kuhar: RawKuHar,
        users: List[str] = None,
        activities: List[str] = None,
        shuffle: bool = False,
    ):
        self.kuhar = kuhar
        self.users_to_select = users if users is not None else kuhar.users
        self.activities_to_select = (
            activities if activities is not None else kuhar.activities
        )
        self.shuffle = shuffle
        self.it = None

    def __get_data_iterator(self) -> Generator[pd.DataFrame, None, None]:
        selecteds = self.kuhar.metadata_df[
            (self.kuhar.metadata_df["user"].isin(self.users_to_select))
            & (self.kuhar.metadata_df["activity code"].isin(self.activities_to_select))
        ]

        # Shuffle data
        if self.shuffle:
            selecteds = selecteds.sample(frac=1)

        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            act_code = row["activity code"]
            user = row["user"]
            serial = row["serial"]
            file = row["file"]
            data = self.kuhar.read_information(
                file=file, activity_code=act_code, user=user, serial=serial
            )
            yield data

    def __str__(self) -> str:
        return f"Kuhar Iterator: users={len(self.users_to_select)}, activities={len(self.activities_to_select)}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        self.it = self.__get_data_iterator()
        return self.it

    def __next__(self):
        return next(self.it)


class KuHarDatasetGenerator:
    def __init__(
        self,
        kuhar_iterator: RawKuHarIterator,
        time_window: int = None,
        window_overlap: int = None
    ):
        self.kuhar_iterator = kuhar_iterator
        self.time_window = time_window
        self.window_overlap = window_overlap

        if window_overlap is not None:
            assert (
                time_window is not None
            ), "Time window must be set when overlap is set"

    def __create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        values = []
        column_names = []
        selected_features = [
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ]

        for i in range(0, data.shape[0], self.time_window - self.window_overlap):
            window_df = data[i : i + self.time_window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window
            window_values = window_df[selected_features].unstack().to_numpy()
            acc_time = (
                window_df["accel-start-time"].iloc[0],
                window_df["accel-end-time"].iloc[-1],
            )
            gyro_time = (
                window_df["gyro-start-time"].iloc[0],
                window_df["gyro-end-time"].iloc[-1],
            )
            act_class = window_df["activity code"].iloc[0]
            length = self.time_window
            serial = window_df["serial"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate(
                (
                    window_values,
                    [
                        acc_time[0],
                        gyro_time[0],
                        acc_time[1],
                        gyro_time[1],
                        act_class,
                        length,
                        serial,
                        start_idx,
                        act_user,
                    ],
                )
            )
            values.append(temp)

        # Name the cows
        column_names = [
            f"{feat}-{i}" for feat in selected_features for i in range(self.time_window)
        ]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()

        # Hack to maintain types
        for c in ["activity code", "length", "serial", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def get_full_df(self, use_tqdm: bool = True) -> pd.DataFrame:
        it = iter(self.kuhar_iterator)
        if use_tqdm:
            it = tqdm.tqdm(
                it, desc="Generating full df over KuHar View",
                position=0, 
                leave=True
            )

        if self.time_window is None:
            return pd.concat(it)
        else:
            return pd.concat(self.__create_time_series(d) for d in it)

    def check_if_unique_per_df(
        self,
        dataset_to_check: pd.DataFrame,
        datasets_list: List[pd.DataFrame],
        column: str = "user",
    ) -> bool:
        def get_uniques(df):
            return list(df[column].unique())

        column_in_ds_to_check = get_uniques(dataset_to_check)
        for ds in datasets_list:
            user_in_ds = get_uniques(ds)
            for i in column_in_ds_to_check:
                if i in user_in_ds:
                    return False
        return True

    def train_test_split(
        self,
        df: pd.DataFrame,
        users: List[int],
        activities: List[int],
        train_size: float,
        validation_size: float,
        test_size: float,
        retries: int = 10,
        ensure_distinct_users_per_dataset: bool = True,
        seed: int = None,
    ):
        n_users = len(users)

        for i in range(retries):
            # [start ---> train_size)
            random.shuffle(users)
            train_users = users[0:int(n_users * train_size)]
            # [train_size --> train_size+validation_size)
            validation_users = users[
                int(n_users * train_size):
                int(n_users * (train_size + validation_size))
            ]
            # [train_size+validation_size --> end]
            test_users = users[int(n_users * (train_size + validation_size)):]
            # iterate over user's lists, filter df for users in the respective list
            all_sets = [
                df[df["user"].isin(u)]
                for u in [train_users, validation_users, test_users]
            ]

            if not ensure_distinct_users_per_dataset:
                return all_sets

            # We must guarantee that all sets contains at least 1 sample from each activities listed
            oks = [set(s["activity code"]) == set(activities) for s in all_sets]
            if all(oks):
                # If all sets contains at least 1 sample for each activity, return train, val, test sets!
                return all_sets

        raise DatasetSplitError(
            "Does not found a 3 sets that contain the respective activities!"
        )

    def balance_dataset_to_minimum(
        self, dataframe: pd.DataFrame, column: str = "activity code"
    ) -> pd.DataFrame:
        df_list = []
        histogram = dataframe.groupby(dataframe[column], as_index=False).size()
        for c in histogram[column]:
            temp = dataframe.loc[dataframe[column] == c]
            temp = temp.sample(n=histogram["size"].min())
            df_list.append(temp)
        return pd.concat(df_list)

    def create_datasets(
        self,
        train_size: float,
        validation_size: float,
        test_size: float,
        ensure_distinct_users_per_dataset: bool = True,
        balance_samples: bool = True,
        activities_remap: Dict[int, int] = None,
        seed: int = None,
        use_tqdm: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert np.isclose(
            sum([train_size, validation_size, test_size]), 1.0
        ), "The sizes must sum up to 1"
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)

        df = self.get_full_df(use_tqdm=use_tqdm)
        users = df["user"].unique()
        activities = df["activity code"].unique()

        train, validation, test = self.train_test_split(
            df=df,
            users=users,
            activities=activities,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            ensure_distinct_users_per_dataset=ensure_distinct_users_per_dataset,
            seed=seed,
        )

        if ensure_distinct_users_per_dataset:
            if (
                not self.check_if_unique_per_df(train, [validation, test])
                or not self.check_if_unique_per_df(validation, [train, test])
                or not self.check_if_unique_per_df(test, [validation, train])
            ):
                raise DatasetSplitError(
                    "Samples from the same user belongs to different dataset splits."
                )

        if activities_remap is not None:
            train.replace({"activity code": activities_remap}, inplace=True)
            validation.replace({"activity code": activities_remap}, inplace=True)
            test.replace({"activity code": activities_remap}, inplace=True)

        # balance datasets!
        if balance_samples:
            train = self.balance_dataset_to_minimum(train)
            validation = self.balance_dataset_to_minimum(validation)
            test = self.balance_dataset_to_minimum(test)

        # reset indexes
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

        # features = [
        #     column
        #     for col_prefix in [
        #         "accel-x",
        #         "accel-y",
        #         "accel-z",
        #         "gyro-x",
        #         "gyro-y",
        #         "gyro-z",
        #     ]
        #     for column in df.columns
        #     if column.startswith(col_prefix)
        # ]

        # train = PandasDataset(train,
        #                       features_columns=features,
        #                       label_columns=self.labels)
        # validation = PandasDataset(validation,
        #                            features_columns=features,
        #                            label_columns=self.labels)
        # test = PandasDataset(test,
        #                      features_columns=features,
        #                      label_columns=self.labels)

        return train, validation, test

    def __str__(self) -> str:
        return f"Dataset generator: time_window={self.time_window}, overlap={self.window_overlap}"

    def __repr__(self) -> str:
        return str(self)


class KuHarDataset(PandasDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sensors: Optional[Union[str, List[str]]] = None,
        label_columns: Union[str, List[str]] = "activity code",
        as_array: bool = True,
    ):
        if sensors is None:
            features = set(dataframe.columns) - set(label_columns)
        else:
            if isinstance(sensors, str):
                sensors = [sensors]
            features = [
                col
                for sensor in sensors
                for col in dataframe.columns
                if col.startswith(sensor)
            ]

        super().__init__(
            dataframe,
            features_columns=features,
            label_columns=label_columns,
            as_array=as_array,
        )