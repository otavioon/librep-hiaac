from typing import List
import pandas as pd
import numpy as np
from librep.datasets.raw.motionsense import DeviceMotionMotionSenseDataset


class MotionSenseView:
    def __init__(self, motionsense: DeviceMotionMotionSenseDataset,
                 users: List[int] = None, activities: List[int] = None):
        self.motionsense = motionsense
        self.users = users
        self.activities = activities

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.motionsense.get_data_iterator(
            users=self.users, activities=self.activities)
        return list(it)

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        return f"MotionSenseView (users={users}, activities={activities})"


class TimeSeriesMotionSenseView:
    def __init__(self, motionsense: DeviceMotionMotionSenseDataset,
                 users: List[int] = None, activities: List[int] = None,
                 window: int = None, overlap: int = 0):
        self.motionsense = motionsense
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
            "attitude.roll", "attitude.pitch", "attitude.yaw",
            "gravity.x", "gravity.y", "gravity.z",
            "rotationRate.x", "rotationRate.y", "rotationRate.z",
            "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
        ]

        for i in range(0, data.shape[0], window-overlap):
            window_df = data[i:i+window]
            # print(i, i+window, len(window_df)) # --> dropna will remove i:i+window ranges < window 
            window_values = window_df[selected_features].unstack().to_numpy()
            #acc_time = window_df["accel-start-time"].iloc[0], window_df["accel-end-time"].iloc[-1]
            #gyro_time = window_df["gyro-start-time"].iloc[0], window_df["gyro-end-time"].iloc[-1]
            act_class = window_df["class"].iloc[0]
            length = window
            trial_code = window_df["trial_code"].iloc[0]
            start_idx = window_df["index"].iloc[0]
            act_user = window_df["user"].iloc[0]

            temp = np.concatenate(
                (window_values, [
                    #acc_time[0], gyro_time[0], acc_time[1], gyro_time[1],
                    act_class, length, trial_code, start_idx, act_user
                ])
            )
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
        for c in ["class", "length", "trial_code", "index", "user"]:
            df[c] = df[c].astype(np.int)

        return df

    def to_canonical(self) -> List[pd.DataFrame]:
        it = self.motionsense.get_data_iterator(users=self.users, activities=self.activities)
        dfs = [self._create_time_series(i) for i in it]
        return dfs

    def __str__(self):
        users = 'all' if self.users is None else len(self.users)
        activities = 'all' if self.activities is None else len(self.activities)
        window = 'whole data' if self.window is None else self.window
        overlap = self.overlap
        return f"MotionSenseTimeSeriesView (window={window}, overlap={overlap}, no. users={users}, no. activities={activities})"
