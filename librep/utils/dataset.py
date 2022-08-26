from pathlib import Path
from typing import Tuple

import pandas as pd

from librep.config.type_definitions import PathLike


class PandasDatasetsIO:
    def __init__(
        self,
        path: PathLike,
        train_filename: str = "train.csv",
        validation_filename: str = "validation.csv",
        test_filename: str = "test.csv",
    ):
        self.path = Path(path)
        self.train_filename = train_filename
        self.validation_filename = validation_filename
        self.test_filename = test_filename

    def save(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        description_file: str = "README.md",
        description: str = None,
    ):
        self.path.mkdir(parents=True, exist_ok=True)
        # Save train
        if train is not None:
            train_filename = self.path / self.train_filename
            train.to_csv(train_filename)
        # Save validation
        if validation is not None:
            validation_filename = self.path / self.validation_filename
            validation.to_csv(validation_filename)
        # Save test
        if test is not None:
            test_filename = self.path / self.test_filename
            test.to_csv(test_filename)

        if description is not None:
            description_filename = self.path / description_file
            with description_filename.open("w") as f:
                f.write(description)

    def __str__(self) -> str:
        return f"PandasDatasetIO at '{self.path}'"

    def __repr__(self) -> str:
        return f"PandasDatasetIO at '{self.path}'"

    def load(
        self,
        load_train: bool = True,
        load_validation: bool = True,
        load_test: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Load train file
        train_filename = self.path / self.train_filename
        train = pd.read_csv(train_filename) if load_train else None
        # Load validation file
        validation_filename = self.path / self.validation_filename
        validation = pd.read_csv(validation_filename) if load_validation else None
        # Load test file
        test_filename = self.path / self.test_filename
        test = pd.read_csv(test_filename) if load_test else None
        return (train, validation, test)