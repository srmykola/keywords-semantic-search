from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
from os import listdir
from os.path import (
    join,
    isfile
)
import os
import numpy as np
from tqdm import tqdm
import json
from time import time
from vectorizers import vectorise_list


def build_annoy_index(input_file_path: str, index_file_path: str, n_trees: int):
    """ Building the annoy index file """

    vectors = np.load(input_file_path)
    vectors = normalize(vectors)

    t = AnnoyIndex(vectors.shape[1], 'dot')
    t.on_disk_build(index_file_path)

    for i, v in tqdm(enumerate(vectors)):
        t.add_item(i, v)

    t.build(n_trees)


def build_annoy_index_from_strings(strings: list, vectors_file_path: str, index_file_path: str, n_trees: int):
    """
    Building vector representations file, and annoy index file
    """
    vectors_folder = "/".join(vectors_file_path.split("/")[:-1])
    if not os.path.isdir(vectors_folder):
        os.makedirs(vectors_folder)
    index_folder = "/".join(index_file_path.split("/")[:-1])
    if not os.path.isdir(index_folder):
        os.makedirs(index_folder)
    vectors = vectorise_list(texts=strings)
    np.save(vectors_file_path, vectors)
    build_annoy_index(index_file_path=index_file_path, input_file_path=vectors_file_path, n_trees=n_trees)
