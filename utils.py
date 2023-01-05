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
    losses = []

    for x in train_loader:
        # forward pass
        optimizer.zero_grad()
        y, det_J = model(x)
        loss = NLL()
        output = loss(y, det_J)

        # print('OUTPUT - train one epoch loss\n', output)

        # update the model
        output.backward()
        optimizer.step()

        # collect statistics
        losses.append(output.detach())

    epoch_loss = torch.mean(torch.tensor(losses))

    return float(epoch_loss)
