import os
import numpy
from typing import Union, Hashable

# PathLike: The PathLike type is used for defining a file path.
PathLike = Union[str, os.PathLike]
ArrayLike = Union[numpy.ndarray]
KeyType = Hashable