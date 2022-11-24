#!/usr/bin/env python3
#$ -N MS1_RNVP
#$ -l cuda=1
#$ -q all.q
#$ -cwd
#$ -V
#$ -t 1-100
#$ -M louisedqne@gmail.com

import os
import torch
import torch.utils.data as data
import torchvision
from model.rnvp.RNVP2 import RNVP
import matplotlib.pyplot as plt
import numpy as np

from data import MoonDataset, FunDataset
from utils import show

path_data_cluster = '/home/space/datasets'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset: str, transformation=None, n_train=None, n_test=None, noise=None, download= False):
    """Loading of the dataset"""

    # Default settings
    if not n_train:
        n_train = 100
    if not n_test:
        n_test = 100

    batch_size = 1

    if dataset == 'FunDataset':
        directory = FunDataset.DIRECTORY
        train_dataset = FunDataset.FunDataset(n_train, noise= noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = FunDataset.FunDataset(n_test, noise= noise, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset == 'MoonDataset':
        directory = MoonDataset.DIRECTORY
        train_dataset = MoonDataset.MoonDataset(n_train, noise= noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MoonDataset.MoonDataset(n_test, noise= noise, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset == 'MNIST':
        directory = '/home/space/datasets/MNIST'
        train_dataset = torchvision.datasets.MNIST(directory, train=True, transform=transformation,  download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.MNIST(directory, train=False, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    else:
        print('Dataset not found')
        train_dataset = None
        test_dataset = None
        train_loader = None
        test_loader = None

    return train_dataset, train_loader, test_dataset, test_loader


if __name__ == "__main__":
    #Dowload a MoonDataset example
    data_Moon, train_Moon, _, _ = load_data('MoonDataset', transformation=None, n_train=100, n_test=100, noise=0.1, download=True)

    # Dowload a FunDataset example
    data_Fun, train_Fun, _, _ = load_data('FunDataset', transformation=None, n_train=100, n_test=100, noise=0.1, download=True)

    #Download MNIST
    data_MNIST, train_MNIST, _, _ = load_data('MNIST', transformation=None, n_train=100, n_test=100, download=True)

    #Creating the model
    model_rnvp = RNVP(2, 1)

    #Passing MoonData into the model
    exit_array = []
    for element in train_Moon:
        exit_data = model_rnvp(element)
        print(exit_data)
        exit_data = exit_data.detach().numpy()
        print(exit_data)
        exit_array.append(exit_data[0])

    # Plot the data
    exit_array = np.array(exit_array)
    show(exit_array, outfile=None)










