import glob
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tqdm

from librep.base.data import Dataset
from librep.config.type_definitions import PathLike
from librep.utils.dataset import PandasDataset

from .raw import RawKuHar


class DatasetSplitError(Exception):
    pass


# class KuHarGenerator:

#     def __init__(self, kuhar: RawKuHar):
#         assert isinstance(kuhar, RawKuHar), "Invalid Raw kuhar handler"
#         self.kuhar = kuhar

#     def generate(self) -> List[pd.DataFrame]:
#         raise NotImplementedError

# class RawGenerator(KuHarGenerator):

#     def __init__(self,
#                  kuhar: RawKuHar,
#                  users: List[int] = None,
#                  activities: List[int] = None):
#         super.__init__(kuhar)
#         self.users = users if users is not None else kuhar.get_all_user_ids()
#         self.activities = (activities if activities is not None else
#                            kuhar.get_all_activity_ids())

#     def generate(self) -> List[pd.DataFrame]:
#         it = self.kuhar.get_data_iterator(users=self.users,
#                                           activities=self.activities)
#         return list(it)

#     def __str__(self) -> str:
#         return f"RawGenerator (no. users={len(self.users)}, no. activities={len(self.activities)})"

#     def __repr__(self) -> str:
#         return str(self)

# class CustomSeriesGenerator(KuHarGenerator):

#     def __init__(
#         self,
#         kuhar: RawKuHar,
#         users: List[int] = None,
#         activities: List[int] = None,
#         window: int = None,
#         overlap: int = 0,
#         use_tqdm: bool = True,
#     ):
#         self.kuhar = kuhar
#         self.users = users if users is not None else kuhar.get_all_user_ids()
#         self.activities = (activities if activities is not None else
#                            kuhar.get_all_activity_ids())
#         self.window = window
#         self.overlap = overlap
#         self.use_tqdm = use_tqdm

#     def _create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
#         window = self.window
#         overlap = self.overlap

#         if self.window is None:
#             window = data.shape[0]

#         if overlap < 0 or overlap >= window:
#             raise ValueError("Overlap value must be in range [0, window)")

#         values = []
#         column_names = []
#         selected_features = [
#             "accel-x",
#             "accel-y",
#             "accel-z",
#             "gyro-x",
#             "gyro-y",
#             "gyro-z",
#         ]

#         for i in range(0, data.shape[0], window - overlap):
#             window_df = data[i:i + window]
#             # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window
#             window_values = window_df[selected_features].unstack().to_numpy()
#             acc_time = (
#                 window_df["accel-start-time"].iloc[0],
#                 window_df["accel-end-time"].iloc[-1],
#             )
#             gyro_time = (
#                 window_df["gyro-start-time"].iloc[0],
#                 window_df["gyro-end-time"].iloc[-1],
#             )
#             act_class = window_df["class"].iloc[0]
#             length = window
#             serial = window_df["serial"].iloc[0]
#             start_idx = window_df["index"].iloc[0]
#             act_user = window_df["user"].iloc[0]

#             temp = np.concatenate((
#                 window_values,
#                 [
#                     acc_time[0],
#                     gyro_time[0],
#                     acc_time[1],
#                     gyro_time[1],
#                     act_class,
#                     length,
#                     serial,
#                     start_idx,
#                     act_user,
#                 ],
#             ))
#             values.append(temp)

#         # Name the cows
#         column_names = [
#             f"{feat}-{i}" for feat in selected_features for i in range(window)
#         ]
#         column_names += [c for c in data.columns if c not in selected_features]
#         df = pd.DataFrame(values, columns=column_names)
#         # Drop non values (remove last rows that no. samples does not fit window size)
#         df = df.dropna()

#         # Hack to maintain types
#         for c in ["class", "length", "serial", "index", "user"]:
#             df[c] = df[c].astype(np.int)

#         return df

#     def generate(self) -> List[pd.DataFrame]:
#         it = self.kuhar.get_data_iterator(users=self.users,
#                                           activities=self.activities)
#         if self.use_tqdm:
#             it = tqdm.tqdm(it, desc="Generating time windows", position=0)
#         dfs = [self._create_time_series(i) for i in it]
#         return dfs

#     def __str__(self):
#         window = "whole data" if self.window is None else self.window
#         overlap = self.overlap
#         return f"KuHarTimeSeriesView (window={window}, overlap={overlap}, no. users={len(self.users)}, no. activities={len(self.activities)})"

#     def __repr__(self) -> str:
#         return str(self)


class KuHarRawIterator:

    def __init__(self,
                 kuhar: RawKuHar,
                 users: List[str] = None,
                 activities: List[str] = None,
                 shuffle: bool = False):
        self.kuhar = kuhar
        self.users = users if users is not None else kuhar.get_all_user_ids()
        self.activities = (activities if activities is not None else
                           kuhar.get_all_activity_ids())
        self.shuffle = shuffle
        self.it = None

    def __str__(self) -> str:
        return f"KuharView: users={len(self.users)}, activities={len(self.activities)}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        self.it = self.kuhar.get_data_iterator(users=self.users,
                                               activities=self.activities,
                                               shuffle=self.shuffle)
        return self.it

    def __next__(self):
        return next(self.it)


class KuHarDatasetGenerator:
    description = ""

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

    def __init__(self,
                 kuhar_view: KuHarRawIterator,
                 time_window: int = None,
                 window_overlap: int = None,
                 labels: Union[str, List[str]] = "class"):
        self.kuhar_view = kuhar_view
        self.time_window = time_window
        self.window_overlap = window_overlap
        self.labels = labels

        if window_overlap is not None:
            assert time_window is not None, "Time window must be set when overlap is set"

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

        for i in range(0, data.shape[0],
                       self.time_window - self.window_overlap):
            window_df = data[i:i + self.time_window]
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
            act_class = window_df["class"].iloc[0]
            length = self.time_window
            serial = window_df["serial"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate((
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
            ))
            values.append(temp)

        # Name the cows
        column_names = [
            f"{feat}-{i}" for feat in selected_features
            for i in range(self.time_window)
        ]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()

        # Hack to maintain types
        for c in ["class", "length", "serial", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def get_full_df(self, use_tqdm: bool = True) -> pd.DataFrame:
        it = iter(self.kuhar_view)
        if use_tqdm:
            it = tqdm.tqdm(it, desc="Generating full df over KuHar View", position=0, leave=True)

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
                         ensure_distinct_users_per_dataset: bool = True):
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


#     @staticmethod
#     def load(filepath: PathLike) -> PandasDataset:
#         data = pd.read_csv(filepath)

#         features = [
#             column for col_prefix in [
#                 "accel-x",
#                 "accel-y",
#                 "accel-z",
#                 "gyro-x",
#                 "gyro-y",
#                 "gyro-z",
#             ] for column in data.columns if column.startswith(col_prefix)
#         ]

#         return PandasDataset(data,
#                              features_columns=features,
#                              label_column="class")

    def create_datasets(
            self,
            train_size: float,
            validation_size: float,
            test_size: float,
            ensure_distinct_users_per_dataset: bool = True,
            balance_samples: bool = True,
            activities_remap: Dict[int, int] = None,
            seed: int = None,
            use_tqdm: bool = True) -> Tuple[Dataset, Dataset, Dataset]:
        assert np.isclose(sum([train_size, validation_size, test_size]),
                          1.0), "The sizes must sum up to 1"
        random.seed(seed)
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
            ensure_distinct_users_per_dataset=ensure_distinct_users_per_dataset)

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

        features = [
            column for col_prefix in [
                "accel-x",
                "accel-y",
                "accel-z",
                "gyro-x",
                "gyro-y",
                "gyro-z",
            ] for column in df.columns if column.startswith(col_prefix)
        ]

        train = PandasDataset(train,
                              features_columns=features,
                              label_columns=self.labels)
        validation = PandasDataset(validation,
                                   features_columns=features,
                                   label_columns=self.labels)
        test = PandasDataset(test,
                             features_columns=features,
                             label_columns=self.labels)

        return train, validation, test

    def __str__(self) -> str:
        return f"Dataset generator: time_window={self.time_window}, overlap={self.window_overlap}, labels={self.labels}"
    
    def __repr__(self) -> str:
        return str(self)