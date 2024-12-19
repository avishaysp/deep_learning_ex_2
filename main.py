from consts import DROP_EPOCHS, LEARNING_RATE, EPOCHS, BATCH_SIZE
import warnings
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pathlib import Path
from data_handler import load_all_data
from dataset import PTBDataset
from word2vec import init_n_train_word2vec_model

warnings.filterwarnings(
    "ignore",
    message="Adding single vectors to a KeyedVectors which grows by one each time can be costly. Consider adding in batches or preallocating to the required size.",
)

from train import train_model, select_device
from lstm_base import LSTMModel
from lstm_dropout import LSTMDropoutModel
from gru import GRUModel
from gru_dropout import GRUDropoutModel
from test import test_model

if __name__ == "__main__":
    device = select_device()
    print(f"Using device: {device}")

    train_data, valid_data, test_data = load_all_data(Path(Path.cwd()) / "PTB")
    print("Training WORD2VEC model")
    word2vec_model, pad_idx = init_n_train_word2vec_model(train_data)
    print("Done")

    train_dataset = PTBDataset(train_data, word2vec_model)
    val_dataset = PTBDataset(valid_data, word2vec_model)
    test_dataset = PTBDataset(test_data, word2vec_model)

    def collate_fn(batch):
        embeddings, indices = zip(*batch)
        padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
        padded_indices = pad_sequence(indices, batch_first=True, padding_value=pad_idx)
        return padded_embeddings.to(device), padded_indices.to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

    models = [
        (LSTMModel, {"epochs": EPOCHS}),
        (LSTMDropoutModel, {"dropout_rate": 0.5, "epochs": DROP_EPOCHS}),
        (GRUModel, {"epochs": EPOCHS}),
        (GRUDropoutModel, {"dropout_rate": 0.5, "epochs": DROP_EPOCHS}),
    ]

    for model_cls, params in models:
        print(f"\nTraining and testing {model_cls.__name__}")
        trained_model = train_model(
            model_cls,
            device,
            word2vec_model,
            pad_idx,
            train_loader,
            val_loader,
            criterion=criterion,
            **params,
        )
        test_model(trained_model, test_loader, criterion, device)

    print("All done.")
