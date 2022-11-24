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


def load_data(dataset: str, transformation=None, n_train=None, n_test=None):
    """Loading of the dataset"""

    # Default settings
    if not n_train:
        n_train = 100
    if not n_test:
        n_test = 100

    batch_size = 1

    if dataset == 'FunDataset':
        directory = FunDataset.DIRECTORY
        train_dataset = FunDataset.FunDataset(n_train, transform=transformation)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = FunDataset.FunDataset(n_test, transform=transformation)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset == 'MoonDataset':
        directory = FunDataset.DIRECTORY
        train_dataset = MoonDataset.MoonDataset(n_train, noise=0.1, transform=transformation)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = MoonDataset.MoonDataset(n_test, transform=transformation)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    elif dataset == 'MNIST':
        directory = '/home/space/datasets/MNIST'
        train_dataset = torchvision.datasets.MNIST(directory, train=True, transform=transformation, download=True)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = torchvision.datasets.MNIST(directory, train=False, transform=transformation, download=True)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    else:
        print('Dataset not found')
        train_loader = None
        test_loader = None

    return train_loader, test_loader


train, test = load_data('MoonDataset', transformation=None, n_train=100, n_test=100)
model_rnvp = RNVP(2, 1)
sortie_array=[]
for element in train:
    sortie = model_rnvp(element)
    print(sortie)
    sortie = sortie.detach().numpy()
    print(sortie)
    sortie_array.append(sortie[0])
sortie_array = np.array(sortie_array)
print(sortie_array)
print(np.size(sortie_array))
plt.plot(sortie_array[:,0], sortie_array[:,1], '.')
plt.show()




