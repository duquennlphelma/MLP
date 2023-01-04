import matplotlib.pyplot as plt
import numpy as np
import os
import torch.utils.data as data
import torch.nn as nn
import torch
from model.rnvp.loss_function import loss_log


def show(x, outfile=None):
    """
    Plot the Data
    :param x: The data as an array
    :param outfile: if True save PNG file with the plot
    """

    #x_array = np.array(x)

    if outfile is not None:
        directory = '/home/pml_07/MLP'
        file_name = outfile + '.png'
        path = os.path.join(directory, file_name)

        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.savefig(path)
        plt.show()
    else:
        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.show()


def train_one_epoch(model: nn.Module, train_loader: data.DataLoader, optimizer):
    losses = []
    predictions = []

    for x in train_loader:
        # forward pass
        optimizer.zero_grad()
        y , det_J= model(x)
        loss = loss_log

        #size = [np.size(y, 0), np.size(y, 1)]
        #target = torch.randn(size[0], size[1])
        #var = torch.ones(size[0], size[1], requires_grad=True)
        output = loss(y, det_J)
        print('OUTPUT - train one epoch loss')
        print(output)

        # update the mosel
        output.backward()
        optimizer.step()

        # collect statistics

        # detach() returns a new tensor that doesn't share the history of the original Tensor
        losses.append(output.detach())

    epoch_loss = torch.mean(torch.tensor(losses))

    return float(epoch_loss)
