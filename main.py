import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt

from consts import BATCH_SIZE
from data_handler import load_all_data
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import PTBDataset
from lstm_base import LSTMModel
from word2vec import init_n_train_word2vec_model


def compute_perplexity(loss):
    return np.exp(loss)


def main():
    train_data, valid_data = load_all_data(Path(Path.cwd()) / "PTB")
    word2vec_model, pad_idx = init_n_train_word2vec_model(train_data)
    train_dataset = PTBDataset(train_data, word2vec_model)
    val_dataset = PTBDataset(valid_data, word2vec_model)

    def collate_fn(batch):
        embeddings, indices = zip(*batch)
        padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)
        padded_indices = pad_sequence(indices, batch_first=True, padding_value=pad_idx)
        return padded_embeddings, padded_indices

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = LSTMModel(
        input_size=200,
        hidden_size=200,
        num_layers=1,
        output_size=len(word2vec_model.wv),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []
    train_perplexities, val_perplexities = [], []

    num_epochs = 13

    for epoch in range(num_epochs):
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
        val_perplexities.append(compute_perplexity(epoch_val_loss))

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Perplexity: {train_perplexities[-1]:.2f} | Val Perplexity: {val_perplexities[-1]:.2f}")

    # Plot Perplexity
    plt.plot(range(1, num_epochs + 1), train_perplexities, label='Train Perplexity')
    plt.plot(range(1, num_epochs + 1), val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Train and Validation Perplexity over Epochs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
