import matplotlib.pyplot as plt
import numpy as np
import os

def show(x, outfile=None):
    """

    :param x: The data as a torch.Tensor
    :param outfile:
    :return:
    """

    x_array = np.array(x)


    if outfile != None :
        directory = '/Users/louiseduquenne/Documents/BERLIN/Cours/machine_learning_project/MLP/sandbox'
        file_name = outfile + '.png'
        path = os.path.join(directory, file_name)

        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.savefig(path)
        plt.show()
    else:
        plt.plot(x_array[:, 0], x_array[:, 1], '.')
        plt.show()


directory = '/Users/louiseduquenne/Documents/BERLIN/Cours/machine_learning_project/MLP/data/fun_100_10_None.csv'
data = np.loadtxt(directory, dtype=float, delimiter=',')
show(data, 'data_test')

