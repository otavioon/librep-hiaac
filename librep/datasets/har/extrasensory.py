from pathlib import Path
import warnings

import tqdm
from tqdm.contrib.concurrent import thread_map
import pandas as pd
from typing import List
from librep.config.type_definitions import PathLike


class User:
    """Class representing an User in the Extrasensory dataset"""

    def __init__(
        self,
        uuid: str,
        feature_file: Path,
        user_accelerometer_dir: Path,
        user_gyroscope_dir: Path,
        user_gravity_dir: Path = None,
        store_raw_df: bool = False,
        default_dtype: str = "float64",
    ):
        """

        Parameters
        ----------
        uuid : str
            Unique user identifier.
        feature_file : Path
            Path from the user's feature file.
        user_accelerometer_dir : Path
            Path from the user's accelerometer directory.
        user_gyroscope_dir : Path
            Path from the user's gyroscope directory.
        user_gravity_dir : Path, optional
            Path from the user's gravity directory.
        store_raw_df : bool
            Boolean indicating if the user's dataframe must be cached to be
            accessed quickly, once read. Thus, when calling `raw_measurements`
            method, the cached version will be returned, instead of
            constructing the dataframe again, which may be slow.
            Note that dataframe may be quite large.
        """
        self.uuid = uuid
        self.user_accelerometer_dir = user_accelerometer_dir
        self.user_gyroscope_dir = user_gyroscope_dir
        self.user_gravity_dir = user_gravity_dir
        self.feature_path = feature_file
        self.default_dtype = default_dtype
        self.processed_features_df = None
        self.labels_df = None
        self._raw_df = None
        self.store_raw_df = store_raw_df
        self._parse_features()
        # self.metadata = self._read_user_labels()

    def _parse_features(self):
        """Parse user's feature file. Store at `feature_df`.
        - `feature_names` member will store all features that is not label,
        that is, which does not start with 'label' string.
        - `label_names` member will store the name of all labels.
        """
        # Parse user's CSV file
        feature_df = pd.read_csv(self.feature_path, compression="gzip").astype(
            {"timestamp": int}
        )

        # Use 'timestamp' column as index
        feature_df.rename(columns={"timestamp": "timestamp source"}, inplace=True)
        feature_df.set_index("timestamp source", inplace=True)

        # ----------- Features -----------
        # Extract feature names, that is, columns not starting with "label"
        # nor "timestamp"
        feature_names = sorted(
            [
                col
                for col in feature_df.columns
                if not col.startswith("label") and not col.startswith("timestamp")
            ]
        )
        self.processed_features_df = feature_df[feature_names].astype(
            self.default_dtype
        )

        # ----------- Labels -----------
        # Dictionary mapping the label columns, that is, columns with name starting
        # with "label:" string, to the same label, but without "label:" prefix
        labels_rename = {
            col: col.replace("label:", "")
            for col in feature_df.columns
            if col.startswith("label:")
        }
        # Rename the label dataframe's columns to remove the "label:" string
        feature_df.rename(columns=labels_rename, inplace=True)

        # Select label columns
        label_names = sorted(list(labels_rename.values()))
        # Transform label columns values to booleans instead int or NaN
        # Note: NaN and 0 values will be considered False.
        self.labels_df = feature_df[label_names].fillna(0).astype(bool)

        # Merged labels column
        merged_labels = []
        for i, row in self.labels_df.iterrows():
            true_labels = [k for k in sorted(row.keys()) if row[k]]
            true_labels = ", ".join(true_labels)
            merged_labels.append(true_labels)
        self.labels_df["merged label"] = merged_labels

    @property
    def num_samples(self) -> int:
        """Number of samples attached to this user

        Returns
        -------
        int
            Numbe of sampes from this user.
        """
        return len(self.feature_df)

    @property
    def raw_measurements(self) -> pd.DataFrame:
        """Returns a dataframe with raw mesaurements of an User from all
        timestamps files.

        Note
        ----
        If `store_raw_df` member is set to true, this dataframe will be cached.

        Returns
        -------
        pd.DataFrame
            A dataframe with the user's measurements from all timestamps files.
        """

        # If a cached value exists, returns it.
        if self._raw_df is not None:
            return self._raw_df

        def read_raw_timestamp(timestamp: int) -> pd.DataFrame:
            """Read measurements from a specific timestamp
            Parameters
            ----------
            timestamp : str
                Timestamp to read
            Returns
            -------
            pd.DataFrame
                Dataframe with the user's measurements from a given timestamp.
            """
            try:
                # ---- Accelerometer ----
                acc_file = self.user_accelerometer_dir / f"{timestamp}.m_raw_acc.dat"
                # Read accelerometer's CSV
                acc_data = pd.read_csv(
                    acc_file,
                    header=None,
                    sep=" ",
                    dtype=self.default_dtype,
                    names=[
                        "accelerometer timestamp",
                        "accelerometer-x",
                        "accelerometer-y",
                        "accelerometer-z",
                    ],
                )

                # ---- Gyroscope ----
                gyro_file = self.user_gyroscope_dir / f"{timestamp}.m_proc_gyro.dat"
                # Read gyroscope's CSV
                gyro_data = pd.read_csv(
                    gyro_file,
                    header=None,
                    sep=" ",
                    dtype=self.default_dtype,
                    names=[
                        "gyroscope timestamp",
                        "gyroscope-x",
                        "gyroscope-y",
                        "gyroscope-z",
                    ],
                )

                # ---- Gravity ----
                if self.user_gravity_dir is not None:
                    gravity_file = (
                        self.user_gyroscope_dir / f"{timestamp}.m_proc_gravity.dat"
                    )
                    # Read gravity's CSV
                    gravity_data = pd.read_csv(
                        gravity_file,
                        header=None,
                        sep=" ",
                        dtype=self.default_dtype,
                        names=[
                            "gravity timestamp",
                            "gravity-x",
                            "gravity-y",
                            "gravity-z",
                        ],
                    )
                else:
                    gravity_data = pd.DataFrame()

                # labels = self.feature_df.loc[[timestamp], self.label_names]

                # Concatenate all dataframes (column order)
                data = pd.concat([acc_data, gyro_data, gravity_data], axis=1)

                #                 # Read labels from this timestamp and add to dataframe
                #                 labels = self.feature_df.loc[
                #                     [timestamp], self.label_names + ["label_source"]]
                #                 for i in labels.columns:
                #                     data[i] = labels[i].to_list()*len(data)
                #                 data = data.dropna()

                #                 # Read features from this timestamp
                #                 catch_features = [
                #                     f for f in self.feature_names
                #                     if  f.startswith("raw_acc:") or \
                #                         f.startswith("proc_gyro:") or \
                #                         f.startswith("proc_gravity:")
                #                 ]

                #                 features = self.feature_df.loc[[timestamp], catch_features]
                #                 for i in features.columns:
                #                     data[i] = features[i].to_list()*len(data)

                # Assign a new column (imestamp source) telling from where
                # these values comes from
                data["timestamp source"] = timestamp

                # Just reordering columns.
                # Put "timestamp source" at the beggining
                columns = data.columns.to_list()
                columns = columns[-1:] + columns[:-1]
                data = data[columns]

                # Return data
                return data
            except FileNotFoundError:
                return None

        # Process all user's timstamps
        res = thread_map(
            read_raw_timestamp,
            list(self.labels_df.index),
            desc=f"Reading files from user {self.uuid}...",
        )
        # Filter erronous dataframes
        res = [r for r in res if r is not None]
        # Concatenate all dataframes from all timestamps.
        # Then sort rows at 3 levels:
        #     1. timestamp source (where file comes from)
        #     2. accelerometer timestamp (accelerometer time)
        #     3. gyroscope timestamp (gyroscope time)
        df = pd.concat(res).sort_values(
            by=["timestamp source", "accelerometer timestamp", "gyroscope timestamp"]
        )

        # Cache dataframe, if `store_raw_df` is set
        if self.store_raw_df:
            self._raw_df = df

        # Return dataframe
        return df

    @property
    def processed_features(self) -> pd.DataFrame:
        return self.processed_features_df

    @property
    def labels(self) -> pd.DataFrame:
        return self.labels_df


class RawExtraSensory:
    def __init__(
        self,
        labels_dir: Path,
        accelerometer_dir: Path,
        gyroscope_dir: Path,
        gravity_dir: Path = None,
        store_raw_df: bool = False,
        default_dtype: str = "float32",
    ):
        self.labels_dir = labels_dir
        self.accelerometer_dir = accelerometer_dir
        self.gyroscope_dir = gyroscope_dir
        self.gravity_dir = gravity_dir
        self._users = None
        self.store_raw_df = store_raw_df
        self.default_dtype = default_dtype
        self._read_users()

    def _read_users(self):
        # get all files
        label_files = list(self.labels_dir.rglob("*.csv.gz"))

        def _read_user(label_file: Path):
            userid = label_file.stem.split(".")[0]

            acc_path = self.accelerometer_dir / userid
            if not acc_path.exists():
                print(f"User {userid}: has no accelerometer files (skipping)")
                return None
            gyro_path = self.gyroscope_dir / userid
            if not gyro_path.exists():
                print(f"User {userid}: has no gyroscope files (skipping)")
                return None

            gravity_path = None
            if self.gravity_dir is not None:
                gravity_path = self.gravity_dir / userid
                if not gravity_path.exists():
                    print(f"User {userid}: has no gravity files (skipping)")
                    return None

            return User(
                uuid=userid,
                feature_file=label_file,
                user_accelerometer_dir=acc_path,
                user_gyroscope_dir=gyro_path,
                user_gravity_dir=gravity_path,
                store_raw_df=self.store_raw_df,
                default_dtype=self.default_dtype,
            )

        users = thread_map(_read_user, label_files, desc="Reading users.")
        self._users = {u.uuid: u for u in users if u is not None}

    @property
    def users(self) -> List[User]:
        return [self._users[i] for i in sorted(self.user_ids)]

    @property
    def user_ids(self) -> List[str]:
        return sorted(list(self._users.keys()))

    def user_raw_measurements(self, userid: str) -> pd.DataFrame:
        return self._users[userid].raw_measurements

    def user_processed_features(self, userid: str) -> pd.DataFrame:
        return self._users[userid].processed_features

    def user_labels(self, userid: str) -> pd.DataFrame:
        return self._users[userid].labels

    def __str__(self):
        return (
            f"Extrasensory dataset at {self.labels_dir} with "
            + f"{len(self.all_users)} users"
        )

    def __repr__(self):
        return str(self)


class RawExtraSensoryPreprocessor:
    """Split dataset into raw, labels and feateurs"""

    def __init__(self, raw_es: RawExtraSensory, output_dir: PathLike):
        self.raw_es = raw_es
        self.output_dir = Path(output_dir)

    def convert(
        self,
        user_ids: List[str] = None,
        save_raw: bool = True,
        save_labels: bool = True,
        save_features: bool = True,
    ):
        if user_ids is None:
            users = [
                u for uuid in user_ids for u in self.raw_es.users if u.uuid == uuid
            ]
        else:
            users = self.raw_es.users

        self.output_dir.mkdir(parents=True, exist_ok=True)

        for user in tqdm.tqdm(users, desc="Reading users"):
            if save_raw:
                df = user.raw_measurements
                output_file = self.output_dir / f"{user.uuid}_raw.csv"
                df.to_csv(output_file, index=False)
            if save_labels:
                df = user.labels
                output_file = self.output_dir / f"{user.uuid}_labels.csv"
                df.to_csv(output_file, index=False)
            if save_features:
                df = user.processed_features
                output_file = self.output_dir / f"{user.uuid}_features.csv"
                df.to_csv(output_file, index=False)


# class PreprocessedExtrasensoryDatasetGenerator:
#     raw_url = ""
#     labels_url = ""
#     features_url = ""

#     def __init__(self, dataset_dir: PathLike, download: bool = False):
#         self.dataset_dir = Path(dataset_dir)
#         if download:
#             self.__download_and_extract()
#         self.metadata_df: pd.DataFrame = self.__read_metadata()

#     def __download_and_extract(self):
#         """Download and extract the MotionSense dataset (A only)."""
#         raise NotImplementedError

#     def __read_metadata(self) -> pd.DataFrame:
#         raw_files = {
#             f.stem.split("_raw")[0]: f
#             for f in self.dataset_dir.rglob("*_raw.csv")
#         }
#         label_files = {
#             f.stem.split("_labels")[0]: f
#             for f in self.dataset_dir.rglob("*_labels.csv")
#         }
#         feature_files = {
#             f.stem.split("_features")[0]: f
#             for f in self.dataset_dir.rglob("*_features.csv")
#         }

#         users_relation = []
#         for uuid, path in label_files.items():
#             if uuid not in raw_files:
#                 warnings.warn(f"Skipping user '{uuid}': it does not have correspoding raw file")
#                 continue
#             if uuid not in feature_files:
#                 warnings.warn(f"Skipping user '{uuid}': it does not have correspoding features file")
#                 continue

#             labels_df = pd.read_csv(path)
#             for row_index, row in labels_df.iterrows():
#                 if not row["merged label"]:
#                     continue
#                 users_relation.append({
#                     "activities": row["merged label"].split(","),
#                     "user": uuid,
#                     "timestamp source": row["timestamp source"],
#                     "file": raw_files[uuid]
#                 })

#         return None