# data_handler.py
from pathlib import Path
import nltk


def load_data(file_path):
    with open(file_path, "r") as f:
        sentences = f.readlines()
    return [nltk.tokenize.word_tokenize(sentence.lower()) for sentence in sentences]


def load_all_data(ptb_path: Path):
    train_data = load_data(ptb_path / "ptb.train.txt")
    valid_data = load_data(ptb_path / "ptb.valid.txt")
    return train_data, valid_data
