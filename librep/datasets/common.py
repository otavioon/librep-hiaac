from typing import Dict, Tuple
import pandas as pd


class HARDatasetGenerator:
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
        """Create train/validation/test datasets.

        Parameters
        ----------
        train_size : float
            Fraction of samples to training dataset.
        validation_size : float
            Fraction of samples to validation dataset.
        test_size : float
            Fraction of samples to test dataset.
        ensure_distinct_users_per_dataset : bool
            If True, ensure that samples from an user do not belong to distinct
            datasets (the default is True).
        balance_samples : bool
            If True, the datasets will have the same number of samples per
            class. The number of samples will be reduced to the class with the
            minor number of samples (the default is True).
        activities_remap : Dict[int, int]
            A dictionary used to replace a label from one class to another.
        seed : int
            The random seed (the default is None).
        use_tqdm : bool
            If must use tqdm as iterator (the default is True).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple with the train, validation and test dataframes.

        """
        raise NotImplementedError