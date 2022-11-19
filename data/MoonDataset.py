import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from torch.utils.data import Dataset


class MoonDataset(Dataset):

    """Characterize the 2 moons dataset for pytorch """

    def __init__(self, n_sample: int, shuffle=None, noise=None, random_state=None, transform=None):
        """Initialisation
        :param n_sample: Number of samples created.
        :param shuffle: Whether to shuffle the points.
        :param noise: Standard deviation of Gaussian noise added to the data.
        :param random_state: Determines random number generation for dataset shuffling and noise.
        """
        self.samples, _ = datasets.make_moons(n_sample, shuffle=shuffle, noise=noise, random_state=random_state)
        self.noise = noise
        self.transform = transform

    def __len__(self):
        """Returns the number of samples"""
        return np.size(self.samples, 0)

    def __getitem__(self, idx):
        """Get the idx sample of data"""
        return self.samples[idx]

    def show(self):
        plt.figure()
        plt.plot(self.samples[:, 0], self.samples[:, 1], '.')
        plt.title('MoonDataset for ' + str(len(self)) + ' samples with Gaussian noise std = ' + str(self.noise))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

test=MoonDataset(100)
test.show()