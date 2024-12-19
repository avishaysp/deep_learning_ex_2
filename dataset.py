from torch.utils.data import Dataset
import torch


class PTBDataset(Dataset):
    def __init__(self, data, w2v_model):
        self.data = data
        self.w2v_model = w2v_model
        self.word_to_index = {word: idx for idx, word in enumerate(w2v_model.wv.index_to_key)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        embeddings = []
        indices = []

        for word in sentence:
            if word in self.w2v_model.wv:
                embeddings.append(torch.tensor(self.w2v_model.wv[word]))
                indices.append(self.word_to_index[word])

        if not embeddings:
            return torch.zeros((1, self.w2v_model.vector_size)), torch.tensor([0])

        return torch.stack(embeddings), torch.tensor(indices)



