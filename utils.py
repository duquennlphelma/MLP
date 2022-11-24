import matplotlib.pyplot as plt
import numpy as np
import os

def show(x, outfile=None):
    """
    Plot the Data
    :param x: The data as an array
    :param outfile: if True save PNG file with the plot
    """

    x_array = np.array(x)

    if outfile is not None:
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

