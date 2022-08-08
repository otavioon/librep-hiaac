import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import glob
import tqdm
import random

from librep.utils.file_ops import download_unzip_check
from librep.config.type_definitions import PathLike
from librep.utils.dataset import PandasDataset


class DatasetSplitError(Exception):
    pass


class RawMotionSense:
    # Version 1 MotionSense (Device Motion Data)
    dataset_url = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/A_DeviceMotion_data.zip"

    # Activity names and codes
    activity_names = {
        0: "dws",
        1: "ups",
        2: "sit",
        3: "std",
        4: "wlk",
        5: "jog"
    }

    activity_codes = {v: k for k, v in activity_names.items()}

    def __init__(self, dataset_dir: PathLike, download: bool = False):
        self.dataset_dir = Path(dataset_dir)
        if download:
            self._download_and_extract()
        self.metadata_df = self._read_metadata()

    def _download_and_extract(self):
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        file_path = self.dataset_dir / "motionsense.zip"
        download_unzip_check(url=RawMotionSense.dataset_url,
                             download_destination=file_path,
                             unzip_dir=self.dataset_dir)

    def _read_metadata(self):
        # Let's list all CSV files in the directory
        files = glob.glob(os.path.join(self.dataset_dir, "*", "*.csv"))

        # And create a relation of each user, activity and CSV file
        users_relation = []
        for f in files:
            # Split the path into a list
            dirs = f.split(os.sep)
            # Pick activity name (folder name, e.g.: dws_1)
            activity_name = dirs[-2]
            # Pick CSV file name (e.g.: sub_1.csv)
            csv_file = dirs[-1]
            # Split activity name and trial code(e.g.: ['dws', 1])
            act_name, trial_code = activity_name.split("_")
            trial_code = int(trial_code)
            # Get the activity number from the activity's code
            act_no = self.activity_codes[act_name]
            # Get user code
            csv_splitted = csv_file.split("_")
            user = csv_splitted[1]
            # Remove .csv
            user = user[:-4]
            user = int(user)
            #sequence = '_'.join(csv_splitted[2:])
            # Remove the .csv from sequence
            #sequence = sequence[:-4]
            # Generate a tuple with the information and append to the relation's list
            users_relation.append((act_no, act_name, user, trial_code, f))

        # Create a dataframe with all meta information
        column_dtypes = [("class", np.int), ("cname", str), ("user", np.int),
                         ("trial_code", np.int), ("file", str)]
        metadata_df = pd.DataFrame(users_relation,
                                   columns=[d[0] for d in column_dtypes])
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df

    def read_csv_from_metadata(self, info) -> pd.DataFrame:
        # Default feature names from this dataset
        feature_dtypes = {
            "attitude.roll": np.float,
            "attitude.pitch": np.float,
            "attitude.yaw": np.float,
            "gravity.x": np.float,
            "gravity.y": np.float,
            "gravity.z": np.float,
            "rotationRate.x": np.float,
            "rotationRate.y": np.float,
            "rotationRate.z": np.float,
            "userAcceleration.x": np.float,
            "userAcceleration.y": np.float,
            "userAcceleration.z": np.float
        }

        with open(info['file'], 'r') as f:
            csv_matrix = pd.read_csv(f,
                                     names=list(feature_dtypes.keys()),
                                     dtype=feature_dtypes,
                                     skiprows=1)
            #csv_matrix["User"] = info["User"]
            #csv_matrix["Action Code"] = info["Action Code"]
            #csv_matrix["Sequence"] = info["Sequence"]

            # Reordering to same format as all MotionSense datasets
            csv_matrix = csv_matrix[[
                "attitude.roll", "attitude.pitch", "attitude.yaw", "gravity.x",
                "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y",
                "rotationRate.z", "userAcceleration.x", "userAcceleration.y",
                "userAcceleration.z"
            ]]
            csv_matrix["class"] = info["class"]
            csv_matrix["length"] = 1
            csv_matrix["trial_code"] = info["trial_code"]
            csv_matrix["index"] = range(len(csv_matrix))
            csv_matrix["user"] = info["user"]
            return csv_matrix

    def get_all_user_ids(self) -> List[int]:
        return np.sort(self.metadata_df["user"].unique()).tolist()

    def get_all_activity_ids(self) -> List[int]:
        return np.sort(self.metadata_df["class"].unique()).tolist()

    def get_all_activity_names(self) -> List[str]:
        return [self.activity_names[i] for i in self.get_all_activity_ids()]

    def __str__(self):
        return f"MotionSense Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files, {len(self.get_all_user_ids())} users and {len(self.get_all_activity_ids())} activities)"

    def __repr__(self):
        return f"MotionSense Dataset at: '{self.dataset_dir}'"


class RawMotionSenseIterator:

    def __init__(self,
                 motionsense: RawMotionSense,
                 users: List[str] = None,
                 activities: List[str] = None,
                 shuffle: bool = False):
        self.motionsense = motionsense
        self.users = users if users is not None else motionsense.get_all_user_ids(
        )
        self.activities = (activities if activities is not None else
                           motionsense.get_all_activity_ids())
        self.shuffle = shuffle
        self.it = None

    def __get_data_iterator(self):
        selecteds = self.motionsense.metadata_df[
            (self.motionsense.metadata_df["user"].isin(self.users)) &
            (self.motionsense.metadata_df["class"].isin(self.activities))]

        # Shuffle data
        if self.shuffle:
            selecteds = selecteds.sample(frac=1)

        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self.motionsense.read_csv_from_metadata(row)
            yield data

    def __str__(self) -> str:
        return f"MotionSense Iterator: users={len(self.users)}, activities={len(self.activities)}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        return self.__get_data_iterator()

    def __next__(self):
        return next(self.it)


class MotionSenseDatasetGenerator:
    # Activity names and codes
    activity_names = {
        0: "dws",
        1: "ups",
        2: "sit",
        3: "std",
        4: "wlk",
        5: "jog"
    }

    def __init__(self,
                 motionsense_iterator: RawMotionSenseIterator,
                 time_window: int = None,
                 window_overlap: int = None,
                 labels: Union[str, List[str]] = "class"):
        self.motionsense_iterator = motionsense_iterator
        self.time_window = time_window
        self.window_overlap = window_overlap
        self.labels = labels

        if window_overlap is not None:
            assert time_window is not None, "Time window must be set when overlap is set"

    def __create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        values = []
        column_names = []
        
        selected_features = [
            "attitude.roll", "attitude.pitch", "attitude.yaw", "gravity.x",
            "gravity.y", "gravity.z", "rotationRate.x", "rotationRate.y",
            "rotationRate.z", "userAcceleration.x", "userAcceleration.y",
            "userAcceleration.z"
        ]

        for i in range(0, data.shape[0], self.time_window - self.window_overlap):
            window_df = data[i:i + self.time_window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window
            window_values = window_df[selected_features].unstack().to_numpy()
            #acc_time = window_df["accel-start-time"].iloc[0], window_df["accel-end-time"].iloc[-1]
            #gyro_time = window_df["gyro-start-time"].iloc[0], window_df["gyro-end-time"].iloc[-1]
            act_class = window_df["class"].iloc[0]
            length = self.time_window
            trial_code = window_df["trial_code"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate((
                window_values,
                [
                    #acc_time[0], gyro_time[0], acc_time[1], gyro_time[1],
                    act_class,
                    length,
                    trial_code,
                    start_idx,
                    act_user
                ]))
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
        for c in ["class", "length", "trial_code", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def get_full_df(self, use_tqdm: bool = True) -> pd.DataFrame:
        it = iter(self.motionsense_iterator)
        if use_tqdm:
            it = tqdm.tqdm(it,
                           desc="Generating full df over MotionSense View",
                           position=0,
                           leave=True)

        if self.time_window is None:
            return pd.concat(it)
        else:
            return pd.concat(self.__create_time_series(d) for d in it)

    def check_if_unique_per_df(self,
                               dataset_to_check: pd.DataFrame,
                               datasets_list: List[pd.DataFrame],
                               column: str = "user") -> bool:

        def get_uniques(df):
            return list(df[column].unique())

        column_in_ds_to_check = get_uniques(dataset_to_check)
        for ds in datasets_list:
            user_in_ds = get_uniques(ds)
            for i in column_in_ds_to_check:
                if i in user_in_ds:
                    return False
        return True

    def train_test_split(self,
                         df: pd.DataFrame,
                         users: List[int],
                         activities: List[int],
                         train_size: float,
                         validation_size: float,
                         test_size: float,
                         retries: int = 10,
                         ensure_distinct_users_per_dataset: bool = True,
                         seed: int = None):
        n_users = len(users)

        for i in range(retries):
            # [start ---> train_size)
            random.shuffle(users)
            train_users = users[0:int(n_users * train_size)]
            # [train_size --> train_size+validation_size)
            validation_users = users[int(n_users *
                                         train_size):int(n_users *
                                                         (train_size +
                                                          validation_size))]
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
            oks = [set(s["class"]) == set(activities) for s in all_sets]
            if all(oks):
                # If all sets contains at least 1 sample for each activity, return train, val, test sets!
                return all_sets

        raise DatasetSplitError(
            "Does not found a 3 sets that contain the respective activities!")

    def balance_dataset_to_minimum(self,
                                   dataframe: pd.DataFrame,
                                   column: str = "class") -> pd.DataFrame:
        df_list = []
        histogram = dataframe.groupby(dataframe["class"], as_index=False).size()
        for c in histogram["class"]:
            temp = dataframe.loc[dataframe["class"] == c]
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
        use_tqdm: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert np.isclose(sum([train_size, validation_size, test_size]),
                          1.0), "The sizes must sum up to 1"
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)

        df = self.get_full_df(use_tqdm=use_tqdm)
        users = df["user"].unique()
        activities = df["class"].unique()

        train, validation, test = self.train_test_split(
            df=df,
            users=users,
            activities=activities,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            ensure_distinct_users_per_dataset=ensure_distinct_users_per_dataset,
            seed=seed)

        if ensure_distinct_users_per_dataset:
            if (not self.check_if_unique_per_df(train, [validation, test]) or
                    not self.check_if_unique_per_df(validation, [train, test])
                    or
                    not self.check_if_unique_per_df(test, [validation, train])):
                raise DatasetSplitError(
                    "Samples from the same user belongs to different dataset splits."
                )

        if activities_remap is not None:
            train = train["class"].replace(activities_remap)
            validation = validation["class"].replace(activities_remap)
            test = test["class"].replace(activities_remap)

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
        #     column for col_prefix in [
        #         "accel-x",
        #         "accel-y",
        #         "accel-z",
        #         "gyro-x",
        #         "gyro-y",
        #         "gyro-z",
        #     ] for column in df.columns if column.startswith(col_prefix)
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
        return f"Dataset generator: time_window={self.time_window}, overlap={self.window_overlap}, labels={self.labels}"

    def __repr__(self) -> str:
        return str(self)


class MotionSenseDataset(PandasDataset):

    def __init__(self,
                 dataframe: pd.DataFrame,
                 sensors: Optional[Union[str, List[str]]] = None,
                 label_columns: Union[str, List[str]] = "class",
                 as_array: bool = True):
        if sensors is None:
            features = set(dataframe.columns) - set(label_columns)
        else:
            if isinstance(sensors, str):
                sensors = [sensors]
            features = [
                col for sensor in sensors for col in dataframe.columns
                if col.startswith(sensor)
            ]

        super().__init__(dataframe,
                         features_columns=features,
                         label_columns=label_columns,
                         as_array=as_array)