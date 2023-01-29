import os
import click
import torch
import torch.utils.data as data
import torchvision
from model.rnvp.RNVP2 import RNVP as RNVP2
from model.rnvp.RNVP import RNVP
from random import randrange

from model.rnvp.loss_function import NLL
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from data import MoonDataset, FunDataset
from utils import show, train_one_epoch, index_statistics, train_one_epoch_image

# GLOBAL VARIABLES
DIR_DATA_CLUSTER = '/home/space/datasets'
CURRENT_DIR = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataset: str, transformation=None, n_train=None, n_test=None, noise=None, batch_size=32, shuffle=True,
              download=False):
    """
    Loading of the dataset
    :param dataset: Name of the dataset (MoonDataset, FunDataset or MNIST)
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
        train_dataset = FunDataset.FunDataset(n_train, noise=noise, transform=None, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = FunDataset.FunDataset(n_test, noise=noise, transform=None, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MoonDataset':
        directory = MoonDataset.DIRECTORY
        train_dataset = MoonDataset.MoonDataset(n_train, noise=noise, transform=None, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = MoonDataset.MoonDataset(n_test, noise=noise, transform=None, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    elif dataset == 'MNIST':
        directory = DIR_DATA_CLUSTER
        train_dataset = torchvision.datasets.MNIST(directory, train=True, transform=transformation, download=download)
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_dataset = torchvision.datasets.MNIST(directory, train=False, transform=transformation, download=download)
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        print('DATASET NOT FOUND')
        train_dataset = None
        test_dataset = None
        train_loader = None
        test_loader = None

    return train_dataset, train_loader, test_dataset, test_loader


def train_apply(model, dataset: str, n_train=1000, epochs=10, batch_size=32, lr=1e-4, momentum=0.0, transformation=None):
    """
    Training the model
    :param model: model chosen
    :param dataset: dataset to train on
    :param n_train: number of samples to train on
    :param epochs: number of epochs to train on
    :param batch_size: size of the batches to put the dat into
    :param lr: learning rate
    :param momentum:
    :return: list of loss value per epoch
    """
    _, train_loader, _, test_loader = load_data(dataset, transformation=transformation, n_train=n_train, n_test=1000, noise=0.1,
                                                batch_size=batch_size, shuffle=True, download=False)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=lr)

    # Training metrics
    epoch_loss = []

    # Train the model epochs * times & Collect metrics progress over the training
    for i in range(epochs):
        print('epoch number :', i)
        if dataset == "MNIST":
            epoch_loss_i = train_one_epoch_image(model, train_loader, optimizer)
        else:
            epoch_loss_i = train_one_epoch(model, train_loader, optimizer)
        epoch_loss.append(epoch_loss_i)

    arr_epoch_loss = np.array(epoch_loss)

    return arr_epoch_loss


@click.command()
@click.option("--train", default=True, help="Whether to train a new model or not")
@click.option("--save", default=True, help="Whether to save the trained model parameters")
@click.option("--load_path", default=None, help="Path to the already trained model parameters to load")
@click.option("--dataset", default='MNIST', type=click.Choice(['MoonDataset', 'FunDataset', 'MNIST']),
              help="Dataset chosen : MNIST, FunDataset, MoonDataset.")
@click.option("--epoch", default=10, help="Number of epochs for the training.")
@click.option("--batch_size", default=128, help="Size of the batch for the training and the test.")
@click.option("--sample_train", default=1000, help="Number od samples for training for FunDataset or MoonDataset.")
@click.option("--sample_test", default=1000, help="Number od samples for test for FunDataset or MoonDataset.")
@click.option("--noise", default=0.1,
              help="Standard deviation of gaussian noise added to the samples for FunDataset or MoonDataset.")
@click.option("--learning_rate", default=1e-4, help="Learning rate for the training.")
@click.option("--momentum", default=0, help="Momentum for the training.")
def main(train, load_path, save, dataset, epoch, batch_size, sample_train, sample_test, noise, learning_rate, momentum):
    # CREATING THE MODEL
    if dataset == 'MNIST':
        model_rnvp = RNVP()
    else:
        model_rnvp = RNVP2()

    if load_path:
        # LOAD PARAMETERS OF THE MODEL
        print(f'LOADING THE MODEL PARAMETERS FROM : {load_path}')
        model_rnvp.load_state_dict(torch.load(load_path))

    if train:
        # TRAINING
        print('TRAINING THE MODEL')
        out = train_apply(model=model_rnvp, n_train=sample_train, dataset=dataset, epochs=epoch, batch_size=batch_size,
                          lr=learning_rate, transformation=transforms.Compose([transforms.ToTensor()]))
        # PLOT - LOSS PER EPOCH
        print('FIGURE - TRAINING LOSS PER EPOCH')
        directory = CURRENT_DIR
        file_name = f'epochs_loss_{dataset}_{epoch}_epochs_{batch_size}_batchsize_{learning_rate}_lr.png'
        path = os.path.join(directory, file_name)
        plt.figure()
        plt.title(f'Loss per epoch for {dataset} dataset\n'
                  f'Hyper-parameters :{epoch} epochs, batch_size = {batch_size} & lr={learning_rate} (Adam)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(out, '-')
        plt.savefig(path)
        plt.show()

        if save:
            # SAVE THE CURRENT PARAMETERS OF THE MODEL
            print('SAVING THE MODEL')
            directory = CURRENT_DIR
            file_name = f'model_trained_{dataset}_{epoch}_epochs_{batch_size}_batchsize_{learning_rate}_lr.pth'
            path = os.path.join(directory, file_name)
            torch.save(model_rnvp.state_dict(), path)

    # TESTING THE MODEL
    print('TESTING THE MODEL')
    _, _, test_set, test_loader = load_data(dataset, n_train=sample_train, n_test=sample_test, noise=noise, download=False,
                                            batch_size=batch_size, transformation=transforms.Compose([transforms.ToTensor()]))

    if dataset == 'MNIST':
        # PASSING DATA THROUGH THE MODEL
        print('PASSING TEST DATA THROUGH THE MODEL')
        rand_in_batch = randrange(0, batch_size)
        for x, i in test_loader:
            out_im_batch = model_rnvp(x)
            out_im = out_im_batch[0][rand_in_batch].detach().numpy()

            directory = CURRENT_DIR
            file_name = f'test_output_MNIST_{epoch}_epochs_{batch_size}_batchsize_{learning_rate}_lr.png'
            path = os.path.join(directory, file_name)
            plt.figure()
            plt.suptitle(f'Image in the latent space for MNIST dataset\n'
                         f'Hyper-parameters :{epoch} epochs, batch_size = {batch_size} & lr={learning_rate} (Adam)')
            plt.subplot(1, 2, 1)
            plt.title('Data space')
            plt.imshow(x[rand_in_batch].detach().numpy()[0], cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title('Latent space')
            plt.imshow(out_im[0], cmap='gray')
            plt.savefig(path)
            break

        # TEST OF THE INVERTED MODEL
        print('TEST OF THE INVERTED MODEL')
        gauss_im_batch = torch.from_numpy(np.float32([[np.random.randn(test_loader.dataset[0][0][0].size(0),
                                                                       test_loader.dataset[0][0][0].size(1))]]))
        gauss_im = gauss_im_batch[0][0]
        im_invert_batch = model_rnvp.inverse(gauss_im_batch)
        im_invert = im_invert_batch[0].detach().numpy()[0][0]
        directory = CURRENT_DIR
        file_name = f'test_invert_MNIST_{epoch}_epochs_{batch_size}_batchsize_{learning_rate}_lr.png'
        path = os.path.join(directory, file_name)
        plt.figure()
        plt.suptitle(f'Data recreated by inverting the model for MNIST dataset\n'
                     f'Hyper-parameters :{epoch} epochs, batch_size = {batch_size} & lr={learning_rate} (Adam)')
        plt.subplot(1, 2, 1)
        plt.title('Latent space')
        plt.imshow(gauss_im, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Data space')
        plt.imshow(im_invert, cmap='gray')
        plt.savefig(path)
        plt.savefig(path)

    elif dataset == 'FunDataset' or dataset == 'MoonDataset':
        # PASSING DATA THROUGH THE MODEL
        print('PASSING TEST DATA THROUGH THE MODEL')
        exit_data_array = np.array([[0, 0]])
        for element in test_loader:
            print('ELEMENT VALIDATION', element)
            exit_data, det_exit = model_rnvp(element)
            exit_data = exit_data.detach().numpy()
            exit_data_array = np.concatenate((exit_data_array, exit_data))
        exit_data_array = np.array(exit_data_array[1:])

        directory = CURRENT_DIR
        file_name = f'test_output_{dataset}_{epoch}_epochs_{sample_train}_points_{batch_size}_batchsize_{learning_rate}_lr.png'
        path = os.path.join(directory, file_name)
        plt.figure()
        plt.suptitle(f'Points in the latent space for {dataset} dataset\n'
                     f'Hyper-parameters :{epoch} epochs, {sample_train} points, batch_size = {batch_size} '
                     f'& lr={learning_rate} (Adam)')
        plt.subplot(1, 2, 1)
        plt.title('Data space')
        plt.plot(test_set[:, 0], test_set[:, 1], 'b.')
        plt.subplot(1, 2, 2)
        plt.title('Latent space')
        plt.plot(exit_data_array[:, 0], exit_data_array[:, 1], 'r.')
        plt.savefig(path)

        # TEST OF THE INVERTED MODEL
        print('TEST OF THE INVERTED MODEL')
        gauss_np = np.float32(np.random.multivariate_normal(np.zeros(2), np.eye(2), sample_test))
        gauss_sample = torch.from_numpy(gauss_np)
        sample_invert = model_rnvp.inverse(gauss_sample)[0].detach().numpy()
        directory = CURRENT_DIR
        file_name = f'test_invert_{dataset}_{epoch}_epochs_{sample_train}_points_{batch_size}_batchsize_{learning_rate}_lr.png'
        path = os.path.join(directory, file_name)
        plt.figure()
        plt.suptitle(f'Data recreated by inverting the model for {dataset} dataset\n'
                     f'Hyper-parameters :{epoch} epochs, {sample_test} points, batch_size = {batch_size} '
                     f'& lr={learning_rate} (Adam)')
        plt.subplot(1, 2, 1)
        plt.title('Latent space')
        plt.plot(gauss_np[:, 0], gauss_np[:, 1], 'r.')
        plt.subplot(1, 2, 2)
        plt.title('Data space')
        plt.plot(sample_invert[:, 0], sample_invert[:, 1], 'b.')
        plt.savefig(path)


if __name__ == '__main__':

    main()

    #Pass the data in the other way after training : from normal distribution to fun dataset
    #z = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)).sample(1000)
    #z= torch.from_numpy(np.float32(np.random.multivariate_normal(np.zeros(2), np.eye(2), 1000)))
    #dataset_recreated = model_rnvp.inverse(z)
    #exit_data = dataset_recreated[0].detach().numpy()

    # Plot the data

    #exit_array_bis = np.array(exit_data)
    #show(exit_array_bis, 'plot_dataset_recreated_MNIST')



    """
    #Validation test

    mean, std, skew, kurtosis=index_statistics(torch.tensor(exit_array_test))
    print('dataset:',dataset,' epochs:',epochs,' batch_size:', batch_size, 'n_train:', samples_train, ' n_test:', samples_train, ' lr:', learning_rate, 'noise:', noise)
    print('mean\n', mean)
    print('std\n', std)
    print('skew\n', skew)
    print('kurtosis\n',kurtosis)

    #Plot for validity and choose of hyperparameters
    #The mean : a Normal distribution is centered on 0.
    #The standard deviation : a Normal distribution is reduced so its standard deviation is equal to 1.
    #The kurtosis : a Normal distribution as a kurtosis equal to 0.
    #The skewness : a Normal distribution is symmetric so has a skewness equal to 0.

    dataset = 'MoonDataset'
    #the number times that the learning algorithm will work through the entire training dataset.
    epoch_array = [i * 25 for i in range(1,10)]
    print("epoch_array", epoch_array)
    #the number of samples to work through before updating the internal model parameters.
    batch_size_array =[i * 10 for i in range(1,10)]
    print("batch_size_array", batch_size_array)
    samples_train_array = [i * 500 for i in range(1,8)]
    print("sample_train_array", samples_train_array)
    samples_test = 1000
    noise = 0.1
    learning_rate_array = [0, 1e-6, 1e-5, 1e-4, 1e-3]
    momentum=0

    if dataset == 'FunDataset':

         # Dowload a FunDataset example
        _, _, _, test_loader = load_data('FunDataset', transformation=None, n_train=1000,
                                                    n_test=1000,noise=0.1,download=False)
    if dataset == 'MoonDataset':
        #Dowload a MoonDataset example
        _, _, _, test_loader = load_data('MoonDataset', transformation=None, n_train=1000,
                                                    n_test=1000,noise=0.1,download=False)
        

    #Plot evolution of statistical indexes different Hyperparameters:

    #EPOCHS
    means=[]
    stds=[]
    skews=[]
    kurtosiss=[]
    losses=[]
    for e in epoch_array:
        # Creating the model
        model_rnvp = RNVP(2, 1)
        loss = NLL()
        # Training
        out = train_apply(model_rnvp, dataset, epochs=e, batch_size=100, lr=0.001)
        print('model has been trained')
        # Passing MoonData into the model
        exit_data_array = np.array([[0, 0]])
        loss_data_array = []
        for element in test_loader:
            print('one element')
            exit_data, det_exit = model_rnvp(element)
            loss_data = loss(exit_data, det_exit)
            loss_data_array.append(loss_data)
            exit_data = exit_data.detach().numpy()
            exit_data_array = np.concatenate((exit_data_array, exit_data))

        loss_test = float(torch.mean(torch.tensor(loss_data_array)))
        exit_data_array = np.array(exit_data_array[1:])
        print('exit_data_array', exit_data_array)
        mean, std, skew, kurtosis = index_statistics(torch.tensor(exit_data_array))
        means.append(mean)
        stds.append(std)
        skews.append(skew)
        kurtosiss.append(kurtosis)
        losses.append(loss_test)


    # Ploting the statistical indexes for each epoch
    directory = '/home/pml_07/MLP'
    file_name = 'stat_index_accd_epochs' + '.png'
    path = os.path.join(directory, file_name)
    plt.figure()
    plt.plot(epoch_array, means, 'r', label='mean')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(epoch_array, stds,'b', label='std')
    plt.axhline(y=1, color='b', linestyle='--')
    plt.plot(epoch_array, skews, 'g', label='skew')
    plt.axhline(y=0, color='g', linestyle=':')
    plt.plot(epoch_array, kurtosiss, 'm', label='kurtosis')
    plt.axhline(y=0, color='m', linestyle='dashed')
    plt.plot(epoch_array, losses, 'c', label='losses')

    plt.xlabel('number of epochs')
    plt.title('Evolution of statistical indexes regarding epochs')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    plt.savefig(path)
    plt.show()
 

    # BATCH_SIZE
    means = []
    stds = []
    skews = []
    kurtosiss = []
    losses = []
    for e in batch_size_array:
        # Creating the model
        model_rnvp = RNVP(2, 1)
        loss = NLL()
        # Training
        out = train_apply(model_rnvp, dataset, epochs=200, batch_size=e, lr=0.001)
        # Passing MoonData into the model
        exit_data_array = np.array([[0, 0]])
        loss_data_array = []
        for element in test_loader:
            exit_data, det_exit = model_rnvp(element)
            loss_data = loss(exit_data, det_exit)
            loss_data_array.append(loss_data)
            exit_data = exit_data.detach().numpy()
            exit_data_array = np.concatenate((exit_data_array, exit_data))

        loss_test = float(torch.mean(torch.tensor(loss_data_array)))
        exit_data_array = np.array(exit_data_array[1:])
        mean, std, skew, kurtosis = index_statistics(torch.tensor(exit_data_array))
        means.append(mean)
        stds.append(std)
        skews.append(skew)
        kurtosiss.append(kurtosis)
        losses.append(loss_test)

    # Ploting the statisticak indexes for each batch size
    directory = '/home/pml_07/MLP'
    file_name = 'stat_index_accd_batch_size' + '.png'
    path = os.path.join(directory, file_name)
    plt.figure()
    plt.figure()
    plt.plot(batch_size_array, means, 'r', label='mean')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(batch_size_array, stds,'b', label='std')
    plt.axhline(y=1, color='b', linestyle='--')
    plt.plot(batch_size_array, skews, 'g', label='skew')
    plt.axhline(y=0, color='g', linestyle=':')
    plt.plot(batch_size_array, kurtosiss, 'm', label='kurtosis')
    plt.axhline(y=0, color='m', linestyle='dashed')
    plt.plot(batch_size_array, losses, 'c', label='losses')

    plt.xlabel('batch size')
    plt.title('Evolution of statistical indexes regarding batch size')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    plt.savefig(path)
    plt.show()

    # NUMBER OF SAMPLES
    means = []
    stds = []
    skews = []
    kurtosiss = []
    losses = []
    for e in samples_train_array:
        # Creating the model
        model_rnvp = RNVP(2, 1)
        loss = NLL()
        # Training
        out = train_apply(model_rnvp, dataset, n_train=e, epochs=200, batch_size=100, lr=0.001)
        # Passing MoonData into the model
        exit_data_array = np.array([[0, 0]])
        loss_data_array = []
        for element in test_loader:
            exit_data, det_exit = model_rnvp(element)
            loss_data = loss(exit_data, det_exit)
            loss_data_array.append(loss_data)
            exit_data = exit_data.detach().numpy()
            exit_data_array = np.concatenate((exit_data_array, exit_data))

        loss_test = float(torch.mean(torch.tensor(loss_data_array)))
        exit_data_array = np.array(exit_data_array[1:])
        mean, std, skew, kurtosis = index_statistics(torch.tensor(exit_data_array))
        means.append(mean)
        stds.append(std)
        skews.append(skew)
        kurtosiss.append(kurtosis)
        losses.append(loss_test)

    # Ploting the loss for each epoch
    directory = '/home/pml_07/MLP'
    file_name = 'stat_index_accd_n_train' + '.png'
    path = os.path.join(directory, file_name)
    plt.figure()
    plt.figure()
    plt.plot(samples_train_array, means, 'r', label='mean')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(samples_train_array, stds, 'b', label='std')
    plt.axhline(y=1, color='b', linestyle='--')
    plt.plot(samples_train_array,skews, 'g', label='skew')
    plt.axhline(y=0, color='g', linestyle=':')
    plt.plot(samples_train_array,kurtosiss, 'm', label='kurtosis')
    plt.axhline(y=0, color='m', linestyle='dashed')
    plt.plot(samples_train_array,losses, 'c', label='losses')

    plt.xlabel('number of samples')
    plt.title('Evolution of statistical indexes regarding number of samples')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    plt.savefig(path)
    plt.show()


    #LEARNING RATE
    means=[]
    stds=[]
    skews=[]
    kurtosiss=[]
    losses=[]
    for e in learning_rate_array:
        # Creating the model
        model_rnvp = RNVP(2, 1)
        loss = NLL()
        # Training
        out = train_apply(model_rnvp, dataset, epochs=200, batch_size=100, lr=e)
        # Passing MoonData into the model
        exit_data_array = np.array([[0, 0]])
        loss_data_array = []
        for element in test_loader:
            exit_data, det_exit = model_rnvp(element)
            loss_data = loss(exit_data, det_exit)
            loss_data_array.append(loss_data)
            exit_data = exit_data.detach().numpy()
            exit_data_array = np.concatenate((exit_data_array, exit_data))

        loss_test = float(torch.mean(torch.tensor(loss_data_array)))
        exit_data_array = np.array(exit_data_array[1:])
        mean, std, skew, kurtosis = index_statistics(torch.tensor(exit_data_array))
        means.append(mean)
        stds.append(std)
        skews.append(skew)
        kurtosiss.append(kurtosis)
        losses.append(loss_test)


    # Ploting the statistical indexes for each epoch
    directory = '/home/pml_07/MLP'
    file_name = 'stat_index_accd_lr' + '.png'
    path = os.path.join(directory, file_name)
    plt.figure()
    plt.plot(means, 'r', label='mean')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.plot(stds,'b', label='std')
    plt.axhline(y=1, color='b', linestyle='--')
    plt.plot(skews, 'g', label='skew')
    plt.axhline(y=0, color='g', linestyle=':')
    plt.plot(kurtosiss, 'm', label='kurtosis')
    plt.axhline(y=0, color='m', linestyle='dashed')
    plt.plot(losses, 'c', label='losses')

    plt.xlabel('learning rate')
    plt.title('Evolution of statistical indexes regarding learning rate')

    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper center')
    plt.savefig(path)
    plt.show()
    """
