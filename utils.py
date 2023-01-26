import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data
import torch.nn as nn
import torch
from model.rnvp.loss_function import NLL


def show(x, outfile=None):
    """
    Plot the Data
    :param x: The data as an array
    :param outfile: if True save PNG file with the plot
    """

    x_array = np.array(x)

    if outfile is not None:
        directory = '/home/pml_07/MLP'
        file_name = outfile + '.png'
        path = os.path.join(directory, file_name)
        plt.figure()
        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.savefig(path)
        plt.show()
    else:
        plt.figure()
        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.show()


def train_one_epoch(model: nn.Module, train_loader: data.DataLoader, optimizer):
    """
    Training the model on one epoch
    :param model: model chosen
    :param train_loader: Dataloader of the training data
    :param optimizer: chosen optimizer
    :return: loss value on the epoch
    """
    losses = []

    for x in train_loader:
        optimizer.zero_grad()

        # forward pass
        y, det_J = model(x)

        # Loss function
        loss = NLL()
        output = loss(y, det_J)

        # update the model
        output.backward()
        optimizer.step()

        # collect statistics
        losses.append(output.detach())

    epoch_loss = torch.mean(torch.tensor(losses))

    return float(epoch_loss)

def train_one_epoch_image(model: nn.Module, train_loader: data.DataLoader, optimizer):
    """
    Training the model on one epoch
    :param model: model chosen
    :param train_loader: Dataloader of the training data
    :param optimizer: chosen optimizer
    :return: loss value on the epoch
    """
    losses = []

    for i, x in enumerate(train_loader):
        print('------start new iteration epoch--------')
        x=x[0]
        print('data x', x.size())
        #print('data i', i.size())
        optimizer.zero_grad()
        #print ('data x[0].size', x[0].size())
        #print ('data i', i)
        # forward pass
        y, det_J = model(x)

        # Loss function
        loss = NLL()
        output = loss(y, det_J)


        # update the model
        output.backward()
        optimizer.step()

        # collect statistics
        losses.append(output.detach())

    epoch_loss = torch.mean(torch.tensor(losses))
    print('epoch loss', epoch_loss)


    return float(epoch_loss)


def index_statistics(samples):
    """
    Calculates with simple indexes (mean, std, kurtosis, skewness) the resemblance between the sample distribution and a
    Normal distribution.
    :param samples: samples in 2D to calculate the index on
    :return: indexes
    """

    mean = torch.mean(samples)
    diffs = samples - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skew = torch.mean(torch.pow(zscores, 3.0))
    kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0

    return mean, std, skew, kurtosis


def checkerboard_mask(h, w, reverse_mask=False):
    x, y = torch.arange(h, dtype=torch.int32), torch.arange(w, dtype=torch.int32)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = torch.fmod(xx + yy, 2)
    mask = mask.to(torch.float32).view(1, 1, h, w)
    if reverse_mask:
        mask = 1 - mask
    return mask


def channel_mask(n_channels, reverse_mask=False):
    mask = torch.cat([torch.ones(n_channels//2, dtype=torch.float32),
                      torch.zeros(n_channels-n_channels//2, dtype=torch.float32)])
    mask = mask.view(1, n_channels, 1, 1)
    if reverse_mask:
        mask = 1-mask
    return mask


def squeeze(x):
    """converts a (batch_size,1,4,4) tensor into a (batch_size,4,2,2) tensor"""
    batch_size, c, h, w = x.size()
    x = x.reshape(batch_size, c, h//2, 2, w//2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(batch_size, c*4, h//2, w//2)
    return x


def unsqueeze(x):
    """converts a (batch_size,4,2,2) tensor into a (batch_size,1,4,4) tensor"""
    batch_size, c, h, w = x.size()
    x = x.reshape(batch_size, c//4, 2, 2, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.reshape(batch_size, c//4, h*2, w*2)
    return x

