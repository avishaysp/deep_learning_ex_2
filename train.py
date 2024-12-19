import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from consts import BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_HIDDEN_UNITS
from data_handler import load_all_data
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import PTBDataset
from lstm_base import LSTMModel
from word2vec import init_n_train_word2vec_model


def compute_perplexity(loss):
    return np.exp(loss)


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(
        model_cls=LSTMModel,
        device=None,
        word2vec_model=None,
        pad_idx=None,
        train_loader=None,
        val_loader=None,
        criterion=None,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        dropout_rate=None
        ):
    
    if dropout_rate is not None:
        model = model_cls(
            input_size=NUM_HIDDEN_UNITS,
            hidden_size=NUM_HIDDEN_UNITS,
            num_layers=1,
            output_size=len(word2vec_model.wv),
            dropout_rate=dropout_rate,
        ).to(device)
    else:
        model = model_cls(
            input_size=NUM_HIDDEN_UNITS,
            hidden_size=NUM_HIDDEN_UNITS,
            num_layers=1,
            output_size=len(word2vec_model.wv),
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []
    
    best_val_perplexity = float('inf')
    best_model = None

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_train_loss = 0

        for batch_embeddings, batch_indices in train_loader:
            optimizer.zero_grad()

            inputs = batch_embeddings[:, :-1, :]  # Current words embeddings
            targets = batch_indices[:, 1:]  # Next words indices

            outputs = model(inputs)
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        train_perplexities.append(compute_perplexity(epoch_train_loss))

        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for batch_embeddings, batch_indices in val_loader:
                inputs = batch_embeddings[:, :-1, :]
                targets = batch_indices[:, 1:]

                outputs = model(inputs)
                batch_size, seq_len, vocab_size = outputs.shape
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        epoch_val_loss /= len(val_loader)
        val_losses.append(epoch_val_loss)
        current_val_perplexity = compute_perplexity(epoch_val_loss)
        val_perplexities.append(current_val_perplexity)

        tqdm.write(
            f"\tTrain Perplexity: {train_perplexities[-1]:.2f} | Val Perplexity: {val_perplexities[-1]:.2f}"
        )

        # Early stopping check - stop on first perplexity increase
        if current_val_perplexity < best_val_perplexity:
            best_val_perplexity = current_val_perplexity
            best_model = model.state_dict().copy()
        else:
            tqdm.write(f"\tStopping at epoch {epoch + 1} due to increasing perplexity")
            model.load_state_dict(best_model)
            break

    # Plot Perplexity (only for actual epochs ran)
    plt.plot(range(1, len(train_perplexities) + 1), train_perplexities, label="Train Perplexity")
    plt.plot(range(1, len(val_perplexities) + 1), val_perplexities, label="Validation Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title(model_cls.__name__)
    plt.legend()
    plt.grid()
    plt.show()

    return model
