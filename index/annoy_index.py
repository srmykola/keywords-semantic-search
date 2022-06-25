from annoy import AnnoyIndex
import numpy as np


def annoy_index(vector_len: int, index_file_path: str) -> AnnoyIndex:
    """ Loading the Annoy index to the RAM """

    t = AnnoyIndex(vector_len, 'dot')
    t.load(index_file_path, prefault=True)
    return t


def get_nns_by_vector(a: AnnoyIndex, vector: np.array, threshold: float = .8) -> tuple:
    """ Returns nearest neighbours of the vector from Annoy index """

    min_distance = 1
    index_size = a.get_n_items()
    if index_size > 10:
        n_nns = 10
    else:
        n_nns = index_size

    while min_distance >= threshold:

        ids, scores = a.get_nns_by_vector(vector, n_nns, include_distances=True)
        min_distance = scores[-1]
        n_nns = n_nns * 10

        if n_nns >= index_size:
            break

    ids_thresholded = []
    scores_thresholded = []

    for id, score in zip(ids, scores):
        if score < threshold:
            break
        ids_thresholded.append(id)
        scores_thresholded.append(score)

    return ids_thresholded, scores_thresholded
