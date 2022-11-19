import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

class FunDataset(Dataset):

    """Characterize the fun dataset for pytorch """

    def __init__(self, n_sample: int, noise=None, transform=None):
        """Initialisation
        :param n_sample: Number of samples created.
        :param noise: Standard deviation of Gaussian noise added to the data.
        """

        im = cv2.imread('../sandbox/TUB_Logo.jpg')
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../sandbox/TUB_Logo_gray.jpg', im_gray)
        th, im_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)
        cv2.imwrite('../sandbox/TUB_Logo_bin.jpg', im_th)

        sequence = []
        for i in range(len(im_th) - 1):
            for j in range(len(im_th[0])):
                if im_th[i, j]==0:
                    sequence.append([i, j])

        self.samples=np.array(random.choices(sequence, weights=None, cum_weights=None, k=n_sample))
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
        plt.title('FunDataset for ' + str(len(self)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

test=FunDataset(2000)
test.show()


