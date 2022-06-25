import numpy as np
import os
from os.path import exists
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import logging


def load_encoder(path: str) -> SentenceTransformer:
    """ Load and return the encoder """

    return SentenceTransformer(path)


encoders = {
    "default": load_encoder("paraphrase-multilingual-mpnet-base-v2")
}


def vectorize(text: str,
              encoder_version: str = "default",
              show_progress_bar: bool = False,) -> np.array:
    """ Returns the vector representation of the text """

    return normalize(encoders[encoder_version].encode(text, show_progress_bar=show_progress_bar).reshape(1, -1))[0]


def vectorise_list(texts: str,
                   batch_size: int = 128,
                   show_progress_bar: bool = False,
                   encoder_version: str = "default") -> list:
    """ Returns the vector representation of the text """

    return encoders[encoder_version].encode(texts, batch_size, show_progress_bar=show_progress_bar)
