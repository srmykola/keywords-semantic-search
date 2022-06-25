from index.annoy_index import get_nns_by_vector
from vectorizers import vectorize
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize


def get_text_nns(a: AnnoyIndex, query: str, threshold: float = .8) -> list:
    """
    Gets the nearest neighbours for the text query
    :param texts:
    :param a:
    :param query:
    :param threshold:
    :return:
    """
    vector = normalize([vectorize(text=query)])[0]
    return get_nns_by_vector(a=a, vector=vector, threshold=threshold)
