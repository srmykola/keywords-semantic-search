import json
import pandas as pd
from os.path import join
from scipy.sparse import load_npz
from collections import OrderedDict
from os.path import (
    exists,
    join,
    isfile
)
from os import makedirs
from tqdm import tqdm

# Locals:
from .annoy_index import (
    annoy_index,
    get_nns_by_vector
)
import pysparnn.cluster_index as ci
from src import Config

config = Config()


class Index:
    """ Contains the url index for fast querying. """

    def __init__(self, vector_len: int, index_file_path: str):
        pass
