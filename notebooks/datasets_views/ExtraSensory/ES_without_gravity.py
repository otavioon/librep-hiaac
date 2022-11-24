#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../../../")


# In[2]:


from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import thread_map

from librep.datasets.har.generator import HARDatasetGenerator, DatasetSplitError

from typing import List, Tuple, Dict, Union, Optional
import random

from scipy import signal
import scipy
import itertools
import gc

# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


standard_activity_codes = {
    0: "sit",
    1: "stand",
    2: "walk",
    3: "stair up",
    4: "stair down",
    5: "run",
    6: "stair up and down"
}

standard_activity_names =  {v: k for k, v in standard_activity_codes.items()}


# ## Loading files

# In[4]:


es_dir = Path("../../../../data/datasets/ExtraSensory/raw")
label_dir = Path("../../../../data/datasets/ExtraSensory/labels")
es_files = sorted(list(es_dir.rglob("*.csv")))
es_files = {
    f.stem.split("_")[0]: {
        "data": f,
        "label": next(label_dir.glob(f"{f.stem.split('_')[0]}*"))
    }
    for f in es_files
}

print(list(es_files.items())[0])


# In[5]:


def label_map(merged_label):
    if not isinstance(merged_label, str):
        return float('nan')
    if "SITTING" in merged_label:
        return standard_activity_names["sit"]
    elif "OR_standing" in merged_label:
        return standard_activity_names["stand"]
    elif "FIX_walking" in merged_label:
        return standard_activity_names["walk"]
    elif "FIX_running" in merged_label:
        return standard_activity_names["run"]
    else:
        return float('nan')


# In[6]:


maintain = [
    'timestamp source', 'accelerometer timestamp', 'accelerometer-x',
    'accelerometer-y', 'accelerometer-z', 'gyroscope timestamp',
    'gyroscope-x', 'gyroscope-y', 'gyroscope-z', 'gravity timestamp',
    'gravity-x', 'gravity-y', 'gravity-z', 'merged label',
    'standard activity code'
]

output_dir = Path("../../../../data/datasets/ExtraSensory/raw_processed")

# for user, values in tqdm.tqdm(es_files.items(), total=len(es_files), desc="Procesing files..."):
#     data, label = pd.read_csv(values["data"]), pd.read_csv(values["label"])

#     # Merge the dataframes
#     merged_df = pd.merge(data, label, on="timestamp source")

#     # Drop samples that does not match these conditions:
#     # Is not a SITTING, nor stand, nor walk, not run
#     # merged_df = merged_df.drop(merged_df[
#     #     (merged_df["SITTING"] == False) &
#     #     (merged_df["OR_standing"] == False) &
#     #     (merged_df["FIX_walking"] == False) &
#     #     (merged_df["FIX_running"] == False)].index)

#     # Make standard label
#     merged_df["standard activity code"] = merged_df["merged label"].apply(label_map)
#     merged_df = merged_df[merged_df['standard activity code'].notna()]
#     merged_df = merged_df[maintain]

#     merged_df["standard activity code"] = merged_df["standard activity code"].astype(int)
#     merged_df["user"] = user
#     merged_df = merged_df.reset_index(drop=True)
#     merged_df.to_csv(output_dir / f"{user}.csv", index=False)


# ## Join merged dfs in a single dataframe
# full_es_file = output_dir / "the_extra_sensory.csv"
# full_es_file.parent.mkdir(parents=True, exist_ok=True)
# with full_es_file.open("w") as es_file:
#     es_file.write(",".join(maintain + ["user"]))
#     es_file.write("\n")

#     for user_file in tqdm.tqdm(list(output_dir.glob("*.csv")), desc="Merging files into a single file..."):
#         if "the_extra_sensory" in str(user_file):
#             continue
#         with user_file.open("r") as f:

#             for line in f:
#                 es_file.write(line)


h = signal.butter(3, 0.3, 'hp', fs=50, output='sos')

time = 800 / 40 # Number of samples (es default) / frequency (es default) = seconds (20)


def process_df(df, user: str = ""):
    # Check if it is in G or m/s**2
    for name, subdf in df.groupby("timestamp source"):
        subdf["norm"] = np.sqrt((subdf["accelerometer-x"])**2 + (subdf["accelerometer-y"])**2 + (subdf["accelerometer-z"])**2)
        if np.mean(subdf["norm"]) < 6:
            G_val = scipy.constants.g
        else:
            G_val = 1  # neutral multiplication element
        break

    final_df = pd.DataFrame()
    groups = df.groupby("timestamp source")
    for name, subdf in tqdm.tqdm(groups, total=groups.ngroups, desc=f"Computing user: {user} (G={G_val})", leave=False):
        new_df = pd.DataFrame()
        for sensor, axis in itertools.product(["accelerometer", "gyroscope"], ["x","y","z"]):
            column = f"{sensor}-{axis}"
            if sensor == "accelerometer":
                values = (df[column] * G_val).values
                values = signal.sosfiltfilt(h, values)
            values = signal.resample(values, int(time*20))
            new_df[column] = values
        new_df["timestamp source"] = name
        new_df["merged label"] = subdf["merged label"].values[0]
        new_df["standard activity code"] = subdf["standard activity code"].values[0]
        new_df["user"] = subdf["user"].values[0]
        final_df = pd.concat([final_df, new_df]).reset_index(drop=True)
        # dfs.append(new_df)
    return final_df


files = [x for x in output_dir.glob("*.csv") if "_processed.csv" not in str(x)]

for f in tqdm.tqdm(files, desc="Processing dataset (removing gravity)"):
    user = str(f.stem)

    try:
        df = pd.read_csv(f)
        df = process_df(df, user=user)
        output_path = f.parent / f"{user}_processed.csv"
        df.to_csv(output_path, index=False)
        f.unlink()
    except Exception as ex:
        print(f"Failed to process {user}! {ex.__class__.__name__}: {ex}")
    finally:
        gc.collect()

print("Done")
