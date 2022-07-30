import os
import numpy as np
import pandas as pd
import glob
from typing import List, Dict, Tuple
from ..utils import download_url, unzip_file

class CHARMDataset:
    # Version 1.1 CHARM
    acc_url = "https://zenodo.org/record/4642560/files/CHARM_v1.1_accelerometer.csv?download=1"
    gyr_url = "https://zenodo.org/record/4642560/files/CHARM_v1.1_gyroscope.csv?download=1"
    about_url = "https://zenodo.org/record/4642560/files/CHARM_dataset_v1.1_raw_about.txt?download=1"

    # Activity names and codes
    activity_names = {
        0: "CHAIR",
        1: "COUCH",
        2: "STANDING",
        3: "LYING_UP",
        4: "LYING_SIDE",
        5: "SURFACE",
        6: "WALKING",
        7: "RUNNING",
        8: "UPSTAIRS",
        9: "DOWNSTAIRS"
    }

    activity_codes = {
        "CHAIR": 0,
        "COUCH": 1,
        "STANDING": 2,
        "LYING_UP": 3,
        "LYING_SIDE": 4,
        "SURFACE": 5,
        "WALKING": 6,
        "RUNNING": 7,
        "UPSTAIRS": 8,
        "DOWNSTAIRS": 9
    }    

    def __init__(self, dataset_dir: str, download: bool = False):
        self.dataset_dir = dataset_dir
        if download:
            self._download_and_extract()
        self.metadata_df = self._read_metadata()

    def _download_and_extract(self):
        # Create directories
        os.makedirs(self.dataset_dir, exist_ok=True)
        fname = self.dataset_dir
        fname_about = os.path.join(self.dataset_dir, "about.txt") 
        fname_acc = os.path.join(self.dataset_dir, "acc.csv")
        fname_gyr = os.path.join(self.dataset_dir, "gyr.csv")
        if not os.path.exists(fname):
            print(f"Downloading dataset to '{fname}'")
            download_url(self.acc_url, fname=fname_acc)
            download_url(self.gyr_url, fname=fname_gyr)
            download_url(self.about_url, fname=fname_about)
        else:
            print(f"'{fname}' already exists and will not be downloaded again")
        os.unlink(fname)
        print("Done!")

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
        column_dtypes = [
            ("class", np.int),
            ("cname", str),
            ("user", np.int),
            ("trial_code", np.int), 
            ("file", str)
        ]
        metadata_df = pd.DataFrame(users_relation, columns=[d[0] for d in column_dtypes])
        for name, t in column_dtypes:
            metadata_df[name] = metadata_df[name].astype(t)
        return metadata_df

    def _read_csv_data(self, info) -> pd.DataFrame:
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
            csv_matrix = pd.read_csv(f, names=list(feature_dtypes.keys()), dtype=feature_dtypes, skiprows=1)
            
            # Reordering to same format as all MotionSense datasets
            csv_matrix = csv_matrix[[
                "attitude.roll", "attitude.pitch", "attitude.yaw",
                "gravity.x", "gravity.y", "gravity.z",
                "rotationRate.x", "rotationRate.y", "rotationRate.z",
                "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
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
    
    def get_data_iterator(self, users: List[int] = None, activities: List[int] = None, shuffle: bool = False) -> List[pd.DataFrame]:
        # Must select first
        if users is None:
            users = self.get_all_user_ids()
        if activities is None:
            activities = self.get_all_activity_ids()
            
        selecteds = self.metadata_df[
            (self.metadata_df["user"].isin(users)) & 
            (self.metadata_df["class"].isin(activities))
        ]
        
        # Shuffle data
        if shuffle:
            selecteds = selecteds.sample(frac=1)
        
        for i, (row_index, row) in enumerate(selecteds.iterrows()):
            data = self._read_csv_data(row)
            yield data
            
    def __str__(self):
        return f"CHARM Dataset at: '{self.dataset_dir}' ({len(self.metadata_df)} files, {len(self.get_all_user_ids())} users and {len(self.get_all_activity_ids())} activities)"
