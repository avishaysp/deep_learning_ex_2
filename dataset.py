from torch.utils.data import Dataset
import torch


class PTBDataset(Dataset):
    def __init__(self, data, w2v_model):
        self.data = data
        self.w2v_model = w2v_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        embeddings = [
            torch.tensor(self.w2v_model.wv[word])
            for word in sentence
            if word in self.w2v_model.wv
        ]
        if not embeddings:
            # Return a placeholder tensor if no valid embeddings are found
            return torch.zeros((1, self.w2v_model.vector_size))
        return torch.stack(embeddings)



