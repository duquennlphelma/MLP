import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random

DIRECTORY = '/home/space/datasets/RNVP_FunDataset'


class FunDataset(Dataset):

    """Characterize the fun dataset for pytorch """

    def __init__(self, n_sample: int, noise=None, transform=None, download=False):
        """Initialisation
        :param n_sample: Number of samples created.
        :param noise: Standard deviation of Gaussian noise added to the data.
        """
        directory = DIRECTORY
        file_name = 'fun_' + str(n_sample) + '_' + str(noise) + '_' + str(transform) + '.csv'
        path = os.path.join(directory, file_name)

        if download:
            if os.path.exists(path):
                self.samples = np.loadtxt(path, dtype=float, delimiter=',')
            else:
                im = cv2.imread('shrek.jpg')
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                th, im_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)

                sequence = []
                for i in range((len(im_th) - 1)):
                    for j in range((len(im_th[0])-1)):
                        if im_th[i, j] == 0:
                            sequence.append([i, j])
                clean_samples = np.array(random.choices(sequence, weights=None, cum_weights=None, k=n_sample))
                if noise is None:
                    self.samples = clean_samples
                else:
                    random_noise = np.random.normal(np.mean(clean_samples), noise,
                                                    [len(clean_samples), len(clean_samples[0])])
                    self.samples = clean_samples + random_noise
                np.savetxt(path, self.samples, delimiter=',', fmt='%f')

        else:
            im = cv2.imread('shrek.jpg')
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            th, im_th = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY)

            sequence = []
            for i in range(len(im_th) - 1):
                for j in range(len(im_th[0])):
                    if im_th[i, j] == 0:
                        sequence.append([i, j])
            clean_samples = np.array(random.choices(sequence, weights=None, cum_weights=None, k=n_sample),dtype=np.float32)
            if noise is None:
                self.samples = clean_samples
            else:
                random_noise = np.random.normal(0, noise,
                                                [len(clean_samples), len(clean_samples[0])])
                self.samples = clean_samples + random_noise

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
        plt.plot( self.samples[:, 0],self.samples[:, 1], '.')
        plt.title('FunDataset for ' + str(len(self)) + ' samples with Gaussian noise std = ' + str(self.noise))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()



