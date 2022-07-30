from librep.config.type_definitions import PathLike
from librep.datasets.kuhar.raw import RawKuHar, Trimm
from librep.base.data import *


class KuHarV1Generator(IDatasetWritter, IDatasetGenerator):
    description = ""

    def __init__(self, kuhar_handler: RawKuHar):
        self.kuhar_handler = kuhar_handler

    def generate(self, random_seed: int = 42, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError