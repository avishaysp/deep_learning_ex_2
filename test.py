import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_embeddings, batch_indices in tqdm(test_loader, desc="Testing"):
            inputs = batch_embeddings[:, :-1, :]
            targets = batch_indices[:, 1:]

            outputs = model(inputs)
            batch_size, seq_len, vocab_size = outputs.shape
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    test_perplexity = np.exp(avg_loss)
    print(f"Test Perplexity: {test_perplexity:.2f}")
    return test_perplexity
