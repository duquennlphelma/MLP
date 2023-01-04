import os
import torch
import torch.utils.data as data
import torchvision
from model.rnvp.RNVP2 import RNVP
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from data import MoonDataset, FunDataset
from utils import show, train_one_epoch
from model.rnvp.loss_function import loss_log

path_data_cluster = '/home/space/datasets/MNIST'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset: str, transformation=None, n_train=None, n_test=None, noise=None, batch_size=32, shuffle=True, download= False):
    """Loading of the dataset"""

    # Default settings
    if not n_train:
        n_train = 100
    if not n_test:
        n_test = 100

    if dataset == 'FunDataset':
        directory = FunDataset.DIRECTORY
        train_dataset = FunDataset.FunDataset(n_train, noise= noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = FunDataset.FunDataset(n_test, noise= noise, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MoonDataset':
        directory = MoonDataset.DIRECTORY
        train_dataset = MoonDataset.MoonDataset(n_train, noise= noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = MoonDataset.MoonDataset(n_test, noise= noise, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MNIST':
        directory = path_data_cluster
        train_dataset = torchvision.datasets.MNIST(directory, train=True, transform=transformation,  download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = torchvision.datasets.MNIST(directory, train=False, transform=transformation,  download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        print('Dataset not found')
        train_dataset = None
        test_dataset = None
        train_loader = None
        test_loader = None

    return train_dataset, train_loader, test_dataset, test_loader


def train_apply(model, dataset: str, epochs=10, batch_size=32, lr=0.01, momentum=0.0):
    _, train_loader, _, test_loader = load_data(dataset, transformation=None, n_train=100, n_test=100, noise=0.1,
                                                batch_size=batch_size, shuffle=True, download=False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    epoch_loss = []

    # Train the model epochs * times
    # Collect training metrics progress over the training
    for i in range(epochs):
        epoch_loss_i = train_one_epoch(model, train_loader, optimizer)
        epoch_loss.append(epoch_loss_i)

    arr_epoch_loss = np.array(epoch_loss)

    return arr_epoch_loss


if __name__ == "__main__":
    #Dowload a MoonDataset example
    data_Moon, train_Moon, _, _ = load_data('MoonDataset', transformation=None, n_train=100, n_test=100, noise=0.1, download=False)

    # Dowload a FunDataset example
    # data_Fun, train_Fun, _, _ = load_data('FunDataset', transformation=None, n_train=100, n_test=100, noise=0.1, download=True)

    #Download MNIST
    # data_MNIST, train_MNIST, _, _ = load_data('MNIST', transformation=None, n_train=100, n_test=100, download=True)

    #Creating the model
    model_rnvp = RNVP(2, 1)

    #Passing MoonData into the model
    exit_array = []
    for element in train_Moon:
        print('element', element)
        exit_data = model_rnvp(element)
        print('exit data before detach',exit_data)
        exit_data = exit_data[0].detach().numpy()
        print(exit_data)
        exit_array.append(exit_data[0])

    # Plot the data
    exit_array = np.array(exit_array)
    show(exit_array, 'plot_before_training')
    directory_fig = '/home/pml_07/MLP'

    out = train_apply(model_rnvp, 'MoonDataset', 10, batch_size=25)
    print('Final output')
    print(out)

    #Passing MoonData into the model
    exit_array = []
    for element in train_Moon:
        exit_data = model_rnvp(element)
        print('exit data before detach',exit_data)
        exit_data = exit_data[0].detach().numpy()
        print('exit data after detach',exit_data)
        exit_array.append(exit_data[0])
        print('EXIT ARRAY', exit_array)

    # Plot the data

    exit_array = np.array(exit_array)
    print('EXIT ARRAY', exit_array)
    show(exit_array, 'plot_after_training')
    directory_fig = '/home/pml_07/MLP'










