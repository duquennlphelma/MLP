import numpy as np
from sklearn import datasets
from torch.utils.data import Dataset


class MoonDataset(Dataset):

    def __init__(self, n_sample: int, shuffle=None, noise=None, random_state=None):
        self.samples, _ = datasets.make_moons(n_sample, shuffle=shuffle, noise=noise, random_state=random_state)

    def __len__(self):
        return np.size(self.samples, 0)

    def __getitem__(self, idx):
        return self.samples[idx]
