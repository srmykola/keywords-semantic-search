import numpy as np
from numpy import dot
from numpy.linalg import norm


def cosine_similarity(a: list, b: list) -> float:
    """
    Return cosine similarity between two lists
    """

    return dot(a, b)/(norm(a)*norm(b))
