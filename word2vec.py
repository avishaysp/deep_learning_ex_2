import os
from gensim.models import Word2Vec
from pathlib import Path

import numpy as np

from consts import NUM_HIDDEN_UNITS
from data_handler import load_all_data


def init_n_train_word2vec_model(train_data):
    model =  Word2Vec(
        sentences=train_data,
        vector_size=NUM_HIDDEN_UNITS,
        window=5,
        min_count=2,
        workers=os.cpu_count()
    )
    pad_idx = len(model.wv)
    model.wv.add_vector("<pad>", np.zeros(model.vector_size))
    return model, pad_idx


if __name__ == '__main__':
    # test for correct syntax
    train_data, _ = load_all_data(Path(Path.cwd()) / "PTB")
    word2vec_model, _ = init_n_train_word2vec_model(train_data)
    # example word
    print(len(word2vec_model.wv['word']))
