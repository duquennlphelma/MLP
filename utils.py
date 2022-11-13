import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import os


def create_data(name, path, n_sample, n_point, shuffle=True, noise=None, random_state=None):
    path_join = os.path.join(path, name)
    os.mkdir(path_join)
    for i in range(n_sample):
        arr, _ = datasets.make_moons(n_point, shuffle=shuffle, noise=noise, random_state=random_state)

        np.savetxt(path_join + '/moon_' + str(i) + '.csv', arr, delimiter=',', fmt='%f')


create_data('test', '/Users/alicebatte/Documents/TUB/PML/MLP_Project/', 3, 10)
