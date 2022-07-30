import os
import numpy as np
import pandas as pd
import glob
from typing import List, Tuple, Dict
from pathlib import Path
import random
import warnings

import tqdm

from .raw import RawKuHar, TrimmedRawKuHar
from librep.base.data import Dataset
from librep.config.type_definitions import PathLike
from librep.utils.dataset import PandasDataset


class DatasetSplitError(Exception):
    pass


class KuHarView:

    def __init__(self,
                 kuhar: RawKuHar,
                 users: List[int] = None,
                 activities: List[int] = None):
        self.kuhar = kuhar
        self.users = users
        self.activities = activities

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.kuhar.get_data_iterator(users=self.users,
                                          activities=self.activities)
        return list(it)

    @property
    def cannonical_df(self) -> pd.DataFrame:
        return pd.concat(self.to_canonical())

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        return f"KuHarView (no. users={users}, no. activities={activities})"


class TimeSeriesKuHarView:

    def __init__(self,
                 kuhar: RawKuHar,
                 users: List[int] = None,
                 activities: List[int] = None,
                 window: int = None,
                 overlap: int = 0,
                 use_tqdm: bool = True):
        self.kuhar = kuhar
        self.users = users
        self.activities = activities
        self.window = window
        self.overlap = overlap
        self.use_tqdm = use_tqdm

    def _create_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        window = self.window
        overlap = self.overlap

        if self.window is None:
            window = data.shape[0]

        if overlap < 0 or overlap >= window:
            raise ValueError("Overlap value must be in range [0, window)")

        values = []
        column_names = []
        selected_features = [
            "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"
        ]

        for i in range(0, data.shape[0], window - overlap):
            window_df = data[i:i + window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window
            window_values = window_df[selected_features].unstack().to_numpy()
            acc_time = window_df["accel-start-time"].iloc[0], window_df[
                "accel-end-time"].iloc[-1]
            gyro_time = window_df["gyro-start-time"].iloc[0], window_df[
                "gyro-end-time"].iloc[-1]
            act_class = window_df["class"].iloc[0]
            length = window
            serial = window_df["serial"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate((window_values, [
                acc_time[0], gyro_time[0], acc_time[1], gyro_time[1], act_class,
                length, serial, start_idx, act_user
            ]))
            values.append(temp)

        # Name the cows
        column_names = [
            f"{feat}-{i}" for feat in selected_features for i in range(window)
        ]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()

        # Hack to maintain types
        for c in ["class", "length", "serial", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.kuhar.get_data_iterator(users=self.users,
                                          activities=self.activities)
        if self.use_tqdm:
            it = tqdm.tqdm(it, desc="Generating time windows", position=0)
        dfs = [self._create_time_series(i) for i in it]
        return dfs

    @property
    def cannonical_df(self) -> pd.DataFrame:
        return pd.concat(self.to_canonical())

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        window = 'whole data' if self.window is None else self.window
        overlap = self.overlap
        return f"KuHarTimeSeriesView (window={window}, overlap={overlap}, no. users={users}, no. activities={activities})"


class KuHarV1:
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
        17: "Table-tennis"
    }

    def __init__(self, kuhar_view: TimeSeriesKuHarView):
        self.kuhar_view = kuhar_view

    def __valid_split(self, ds_to_check, dsl) -> bool:

        def get_users_list(df):
            return list(df["user"].unique())

        users_in_ds_to_check = get_users_list(ds_to_check)
        for ds in dsl:
            user_in_ds = get_users_list(ds)
            for i in users_in_ds_to_check:
                if i in user_in_ds:
                    return False
        return True

    def __split(self,
                df: pd.DataFrame,
                users: List[int],
                activities: List[int],
                train_size: float,
                validation_size: float,
                test_size: float,
                seed=42,
                retries: int = 10):
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
            # We must guarantee that all sets contains at least 1 sample from each activities listed
            oks = [set(s["class"]) == set(activities) for s in all_sets]
            if all(oks):
                # If all sets contains at least 1 sample for each activity, return train, val, test sets!
                return all_sets

        raise DatasetSplitError(
            "Does not found a 3 sets that contain the respective activities!")

    def __balance_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df_list = []
        histogram = dataframe.groupby(dataframe["class"], as_index=False).size()
        for c in histogram["class"]:
            temp = dataframe.loc[dataframe["class"] == c]
            temp = temp.sample(n=histogram["size"].min())
            df_list.append(temp)
        return pd.concat(df_list)

    @staticmethod
    def load(filepath: PathLike) -> PandasDataset:
        data = pd.read_csv(filepath)

        features = [
            column for col_prefix in
            ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
            for column in data.columns if column.startswith(col_prefix)
        ]

        return PandasDataset(data,
                             features_columns=features,
                             label_column="class")

    def generate(
            self,
            train_size: float,
            validation_size: float,
            test_size: float,
            seed: int = None,
            balance_samples: bool = True,
            class_select: List[int] = None,
            class_remap: Dict[int,
                              int] = None) -> Tuple[Dataset, Dataset, Dataset]:
        assert np.isclose(sum([train_size, validation_size, test_size]),
                          1.0), "The sizes must sum up to 1"
        random.seed(seed)
        df = self.kuhar_view.cannonical_df
        train, validation, test = self.__split(
            df=df,
            users=self.kuhar_view.kuhar.get_all_user_ids(),
            activities=self.kuhar_view.kuhar.get_all_activity_ids(),
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            seed=seed)

        if not self.__valid_split(train, [validation, test]) or \
           not self.__valid_split(validation, [train, test]) or \
           not self.__valid_split(test, [validation, train]):
            raise DatasetSplitError("Samples from the same user "
                                    "belongs to different dataset splits.")

        if class_select is not None:
            train = train.loc[train["class"].isin(class_select)]
            validation = validation.loc[validation["class"].isin(class_select)]
            test = test.loc[test["class"].isin(class_select)]

        if class_remap is not None:
            train = train["class"].replace(class_remap)
            validation = validation["class"].replace(class_remap)
            test = test["class"].replace(class_remap)

        # balance datasets!
        if balance_samples:
            train = self.__balance_dataset(train)
            validation = self.__balance_dataset(validation).reset_index(
                drop=True)
            test = self.__balance_dataset(test).reset_index(drop=True)

        # reset indexes
        train = train.reset_index(drop=True)
        validation = validation.reset_index(drop=True)
        test = test.reset_index(drop=True)

        features = [
            column for col_prefix in
            ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"]
            for column in df.columns if column.startswith(col_prefix)
        ]

        train = PandasDataset(train,
                              features_columns=features,
                              label_column="class")
        validation = PandasDataset(validation,
                                   features_columns=features,
                                   label_column="class")
        test = PandasDataset(test,
                             features_columns=features,
                             label_column="class")
        return train, validation, test
