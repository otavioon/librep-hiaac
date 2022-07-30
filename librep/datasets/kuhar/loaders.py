from librep.config.type_definitions import PathLike
from librep.base.data import Dataset


class KuHarV1:
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

    def __init__(self, dataset_path: PathLike):
        self.dataset_path = dataset_path

    def load(self) -> Dataset:
        pass


class KuHarV2:
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

    def __init__(self, dataset_path: PathLike):
        self.dataset_path = dataset_path

    def load(self) -> Dataset:
        pass


class KuHarV3:
    activity_names = {
        0: "Stand",
        1: "Sit",
        2: "Walk",
        3: "Run",
        4: "Stair-up",
        5: "Stair-down"
    }

    def __init__(self, dataset_path: PathLike):
        self.dataset_path = dataset_path

    def load(self) -> Dataset:
        pass


class KuHarV4:
    activity_names = {
        0: "Stand-sit",
        1: "Lay-stand",
        2: "Pick",
        3: "Jump",
        4: "Push-up",
        5: "Sit-up"
    }

    def __init__(self, dataset_path: PathLike):
        self.dataset_path = dataset_path

    def load(self) -> Dataset:
        pass