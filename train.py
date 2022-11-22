import os
import torch
import torch.utils.data as data
import torchvision
from model.rnvp.RNVP2 import RNVP

from data import MoonDataset, FunDataset

path_data_cluster = '/home/space/datasets'


def load_data(dataset: str, transformation=None, n_train=None, n_test=None):
    """Loading of the dataset"""

    # Default settings
    if not n_train:
        n_train = 100
    if not n_test:
        n_test = 100

    batch_size = 64

    match dataset:
        case 'FunDataset':
            train_dataset = FunDataset.FunDataset(n_train, transform=transformation)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = FunDataset.FunDataset(n_test, transform=transformation)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        case 'MoonDataset':

            train_dataset = MoonDataset.MoonDataset(n_train, transform=transformation)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = MoonDataset.MoonDataset(n_test, transform=transformation)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        case 'MNIST':

            path_dataset = '/home/space/datasets/MNIST'

            train_dataset = torchvision.datasets.MNIST(path_dataset, train=True, transform=transformation, download=True)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = torchvision.datasets.MNIST(path_dataset, train=False, transform=transformation, download=True)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        case _:
            print('Dataset not found')
            train_loader = None
            test_loader = None

    return train_loader, test_loader


# train, test = load_data('MoonDataset', transformation=None, n_train=100, n_test=100)
# model_rnvp = RNVP(100, 40)
# sortie = model_rnvp.forward(torch.tensor(train))



