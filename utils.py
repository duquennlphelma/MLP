"""Creation of the 2 moons Dataset

@duquennel
@battoubattou

13/11/2022
"""

import os
import torch
from torch.utils.data import Dataset
from sklearn import datasets
import numpy as np


def create_moon_data(name_folder: str, path_folder: str, n_sample: int, n_point: int, shuffle=True, noise=None, random_state=None):
    """
    Create a dataset of two moons.

    :param name_folder: Name of the folder to load the data in (can be created).
    :param path_folder: Path to the folder.
    :param n_sample: Number of samples created.
    :param n_point: Number of points in each data sample
    :param shuffle: Whether to shuffle the points.
    :param noise: Standard deviation of Gaussian noise added to the data.
    :param random_state: Determines random number generation for dataset shuffling and noise.
    """
    path_join = os.path.join(path_folder, name_folder)
    os.mkdir(path_join)
    for i in range(n_sample):
        arr, _ = datasets.make_moons(n_point, shuffle=shuffle, noise=noise, random_state=random_state)

        np.savetxt(path_join + '/moon_' + str(i) + '.csv', arr, delimiter=',', fmt='%f')


create_data('test', '/Users/alicebatte/Documents/TUB/PML/MLP_Project/', 3, 10)
