import os
import torch
import torch.utils.data as data
import torchvision

from data import MoonDataset

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
        case 'MoonDatset':

            train_dataset = MoonDataset.MoonDataset(n_train, transform=transformation)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = MoonDataset.MoonDataset(n_test, transform=transformation)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        case 'MNIST':

            patch_dataset = os.path.join(path_data_cluster, dataset)

            train_dataset = torchvision.datasets.MNIST(patch_dataset, train=True, transform=transformation, download=True)
            train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = torchvision.datasets.MNIST(patch_dataset, train=False, transform=transformation, download=True)
            test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
