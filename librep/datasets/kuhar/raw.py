import os
import numpy as np
import pandas as pd
import glob
from typing import List
from pathlib import Path

from librep.utils.file_ops import download_unzip_check
from librep.config.type_definitions import PathLike


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
        17: "Table-tennis"
    }

    activity_codes = {
        "Stand": 0,
        "Sit": 1,
        "Talk-sit": 2,
        "Talk-stand": 3,
        "Stand-sit": 4,
        "Lay": 5,
        "Lay-stand": 6,
        "Pick": 7,
        "Jump": 8,
        "Push-up": 9,
        "Sit-up": 10,
        "Walk": 11,
        "Walk-backwards": 12,
        "Walk-circle": 13,
        "Run": 14,
        "Stair-up": 15,
        "Stair-down": 16,
        "Table-tennis": 17
    }

    def __init__(self, dataset_dir: PathLike, download: bool = False):
        self.dataset_dir = Path(dataset_dir)
        if download:
            self._download_and_extract()
        self.metadata_df = self._read_metadata()

    def _download_and_extract(self):
        # Create directories
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        file_path = self.dataset_dir / "kuhar.zip"
        download_unzip_check(url=RawKuHar.dataset_url,
                             download_destination=file_path,
                             unzip_dir=self.dataset_dir)

    @property
    def users(self):
        return self.get_all_user_ids()

    @property
    def activities(self):
        return self.get_all_activity_ids()

    def _read_metadata(self):
        # Let's list all CSV files in the directory
        files = glob.glob(os.path.join(self.dataset_dir, "*", "*.csv"))

        # And create a relation of each user, activity and CSV file
        users_relation = []
        for f in files:
            # Split the path into a list
            dirs = f.split(os.sep)
            # Pick activity name (folder name, e.g.: 5.Lay)
            activity_name = dirs[-2]
            # Pick CSV file name (e.g.: 1052_F_1.csv)
            csv_file = dirs[-1]
            # Split activity number and name (e.g.: [5, 'Lay'])
            act_no, act_name = activity_name.split(".")
            act_no = int(act_no)
            # Split user code, act type and sequence number (e.g.: [1055, 'G', 1])
            csv_splitted = csv_file.split("_")
            user = int(csv_splitted[0])
            sequence = '_'.join(csv_splitted[2:])
            # Remove the .csv from sequence
            sequence = sequence[:-4]
            # Generate a tuple with the information and append to the relation's list
            users_relation.append((act_no, act_name, user, sequence, f))

        # Create a dataframe with all meta information
        column_dtypes = [("class", np.int), ("cname", str), ("user", np.int),
                         ("serial", np.int), ("file", str)]
        metadata_df = pd.DataFrame(users_relation,
                                   columns=[d[0] for d in column_dtypes])
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df

    def _read_csv_data(self, info) -> pd.DataFrame:
        # Default feature names from this dataset
        feature_dtypes = {
            "accel-start-time": np.float,
            "accel-x": np.float,
            "accel-y": np.float,
            "accel-z": np.float,
            "gyro-start-time": np.float,
            "gyro-x": np.float,
            "gyro-y": np.float,
            "gyro-z": np.float
        }

        with open(info['file'], 'r') as f:
            csv_matrix = pd.read_csv(f,
                                     names=list(feature_dtypes.keys()),
                                     dtype=feature_dtypes)
            # Reordering to same format as all Ku-Har datasets
            csv_matrix = csv_matrix[[
                "accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z",
                "accel-start-time", "gyro-start-time"
            ]]
            csv_matrix["accel-end-time"] = csv_matrix["accel-start-time"]
            csv_matrix["gyro-end-time"] = csv_matrix["gyro-start-time"]
            csv_matrix["class"] = info["class"]
            csv_matrix["length"] = 1
            csv_matrix["serial"] = info["serial"]
            csv_matrix["index"] = range(len(csv_matrix))
            csv_matrix["user"] = info["user"]
            return csv_matrix

    def get_all_user_ids(self) -> List[int]:
        return np.sort(self.metadata_df["user"].unique()).tolist()

    def get_all_activity_ids(self) -> List[int]:
        return np.sort(self.metadata_df["class"].unique()).tolist()

    def get_all_activity_names(self) -> List[str]:
        return [self.activity_names[i] for i in self.get_all_activity_ids()]

    def get_data_iterator(self,
                          users: List[int] = None,
                          activities: List[int] = None,
                          shuffle: bool = False) -> List[pd.DataFrame]:
        # Must select first
        if users is None:
            users = self.get_all_user_ids()
        if activities is None:
            activities = self.get_all_activity_ids()

        selecteds = self.metadata_df[(self.metadata_df["user"].isin(users)) & (
            self.metadata_df["class"].isin(activities))]

        # Shuffle data
        if shuffle:
            selecteds = selecteds.sample(frac=1)

        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self._read_csv_data(row)
            yield data

    def __str__(self):
        return f"KuHar Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files, {len(self.get_all_user_ids())} users and {len(self.get_all_activity_ids())} activities)"

    def __repr__(self):
        return f"KuHar Dataset at: '{self.dataset_dir}'"


class TrimmedRawKuHar(RawKuHar):
    # 2. Trimmed Raw Dataset v5
    dataset_url = "https://data.mendeley.com/public-files/datasets/45f952y38r/files/49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded"