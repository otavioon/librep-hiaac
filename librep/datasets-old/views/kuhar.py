from typing import List
import pandas as pd
import numpy as np
from librep.datasets.raw.kuhar import RawKuHarDataset


class KuHarView:
    def __init__(self, kuhar: RawKuHarDataset,
                 users: List[int] = None,
                 activities: List[int] = None):
        self.kuhar = kuhar
        self.users = users
        self.activities = activities

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.kuhar.get_data_iterator(
            users=self.users, activities=self.activities
        )
        return list(it)

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        return f"KuHarView (no. users={users}, no. activities={activities})"


class TimeSeriesKuHarView:
    def __init__(self, kuhar: RawKuHarDataset, users: List[int] = None, activities: List[int] = None, window: int = None, overlap: int = 0):
        self.kuhar = kuhar
        self.users = users
        self.activities = activities
        self.window = window
        self.overlap = overlap

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

        for i in range(0, data.shape[0], window-overlap):
            window_df = data[i:i+window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window 
            window_values = window_df[selected_features].unstack().to_numpy()
            acc_time = window_df["accel-start-time"].iloc[0], window_df["accel-end-time"].iloc[-1]
            gyro_time = window_df["gyro-start-time"].iloc[0], window_df["gyro-end-time"].iloc[-1]
            act_class = window_df["class"].iloc[0]
            length = window
            serial = window_df["serial"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate(
                (window_values, [
                    acc_time[0], gyro_time[0], acc_time[1], gyro_time[1],
                    act_class, length, serial, start_idx, act_user
                ])
            )
            values.append(temp)

        # Name the cows    
        column_names = [f"{feat}-{i}" for feat in selected_features for i in range(window)]
        column_names += [c for c in data.columns if c not in selected_features]
        df = pd.DataFrame(values, columns=column_names)
        # Drop non values (remove last rows that no. samples does not fit window size)
        df = df.dropna()

        # Hack to maintain types
        for c in ["class", "length", "serial", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.kuhar.get_data_iterator(users=self.users, activities=self.activities)
        dfs = [self._create_time_series(i) for i in it]
        return dfs

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        window = 'whole data' if self.window is None else self.window
        overlap = self.overlap
        return f"KuHarTimeSeriesView (window={window}, overlap={overlap}, no. users={users}, no. activities={activities})"
