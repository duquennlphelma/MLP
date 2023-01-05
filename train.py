import os
import torch
import torch.utils.data as data
import torchvision
from model.rnvp.RNVP2 import RNVP
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from data import MoonDataset, FunDataset
from utils import show, train_one_epoch, index_statistics

path_data_cluster = '/home/space/datasets/MNIST'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset: str, transformation=None, n_train=None, n_test=None, noise=None, batch_size=32, shuffle=True,
              download=False):
    """
    Loading of the dataset
    :param dataset: Name of the datset (MoonDataset, FunDataset or MNIST)
    :param transformation
    :param n_train: number of training points generated
    :param n_test: number of testing points generated
    :param noise: noise added to the data
    :param batch_size : size of the batches
    :param shuffle: if True the data points are shuffled in the batches
    :param download
    :return Dataloader & Dataset
    """

    # Default settings
    if not n_train:
        n_train = 100
    if not n_test:
        n_test = 100

    if dataset == 'FunDataset':
        directory = FunDataset.DIRECTORY
        train_dataset = FunDataset.FunDataset(n_train, noise=noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = FunDataset.FunDataset(n_test, noise=noise, transform=transformation, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MoonDataset':
        directory = MoonDataset.DIRECTORY
        train_dataset = MoonDataset.MoonDataset(n_train, noise=noise, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = MoonDataset.MoonDataset(n_test, noise=noise, transform=transformation, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MNIST':
        directory = path_data_cluster
        train_dataset = torchvision.datasets.MNIST(directory, train=True, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = torchvision.datasets.MNIST(directory, train=False, transform=transformation, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        print('Dataset not found')
        train_dataset = None
        test_dataset = None
        train_loader = None
        test_loader = None

    return train_dataset, train_loader, test_dataset, test_loader


def train_apply(model, dataset: str, epochs=10, batch_size=32, lr=1e-4, momentum=0.0):
    """
    Training the model
    :param model: model chosen
    :param dataset: dataset to train on
    :param epochs: number of epochs to train on
    :param batch_size: size of the batches to put the dat into
    :param lr: learning rate
    :param momentum:
    :return: list of loss value per epoch
    """
    _, train_loader, _, test_loader = load_data(dataset, transformation=None, n_train=1000, n_test=100, noise=0.1,
                                                batch_size=batch_size, shuffle=True, download=False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # Training metrics
    epoch_loss = []

    # Train the model epochs * times & Collect metrics progress over the training
    for i in range(epochs):
        epoch_loss_i = train_one_epoch(model, train_loader, optimizer)
        epoch_loss.append(epoch_loss_i)

    arr_epoch_loss = np.array(epoch_loss)

    return arr_epoch_loss


if __name__ == "__main__":

    epochs=200
    batch_size=100
    dataset= 'FunDataset'
    samples_train=1000
    samples_test=1000
    noise=0.1
    learning_rate=0.01
    # Dowload a MoonDataset example
    #data_Moon, train_Moon, _, _ = load_data('MoonDataset', transformation=None, n_train=100, n_test=100, noise=0.1,
    #                                        download=False)
    # Dowload a FunDataset example
    _, _, data_Fun, test_Fun = load_data('FunDataset', transformation=None, n_train=samples_train, n_test=samples_test, noise=noise,
                                         download=False)

    # Plotting example of the data
    #ata_Fun_array = [data_Fun[i] for i in range(len(data_Fun))]
    #show(data_Fun_array, 'plot_before_training_Fun_Dataset')


    # Creating the model
    model_rnvp = RNVP(2, 1)
    # Training
    out = train_apply(model_rnvp, dataset, epochs, batch_size=batch_size, lr=learning_rate)

    #Ploting the loss for each epoch
    directory = '/home/pml_07/MLP'
    file_name = 'epoch_loss' + '.png'
    path = os.path.join(directory, file_name)
    plt.figure()
    plt.plot(out, '.')
    plt.savefig(path)
    plt.show()

    # Test

    # Passing MoonData into the model
    exit_array_test = np.array([[0, 0]])
    for element in test_Fun:
        exit_data = model_rnvp(element)
        exit_data = exit_data[0].detach().numpy()
        exit_array_test = np.concatenate((exit_array_test, exit_data))

    # Plot the data
    exit_array_test = np.array(exit_array_test[1:])

    show(exit_array_test, 'plot_after_training_Fun_Dataset')


    #Pass the data in the other way after training : from normal distribution to fun dataset
    #z = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample(1000)
    z= torch.from_numpy(np.float32(np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)))
    dataset_recreated = model_rnvp.inverse(z)
    exit_data = dataset_recreated[0].detach().numpy()

    # Plot the data

    exit_array_bis = np.array(exit_data)
    show(exit_array_bis, 'plot_dataset_recreated')

    #Validation test

    mean, std, skew, kurtosis=index_statistics(torch.tensor(exit_array_test))
    print('dataset:',dataset,' epochs:',epochs,' batch_size:', batch_size, 'n_train:', samples_train, ' n_test:', samples_train, ' lr:', learning_rate, 'noise:', noise)
    print('mean\n', mean)
    print('std\n', std)
    print('skew\n', skew)
    print('kurtosis\n',kurtosis)
