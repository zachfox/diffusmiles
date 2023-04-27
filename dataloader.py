import torch
from torch.utils.data import Dataset, DataLoader

class SMILESDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Here, you can preprocess your data if needed and return a tuple (or dictionary) containing the sample and its label
        sample = self.data[index]
        label = 0  # Example label
        return sample


