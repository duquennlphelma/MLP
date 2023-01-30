import numpy as np
import matplotlib.pyplot as plt
import os

from pylab import rcParams
rcParams['figure.figsize'] = 10, 8
rcParams['figure.dpi'] = 300

import torch
from torch import nn
from torch import distributions
import torch.utils.data as data
from torch.utils.data import Dataset
from sklearn import datasets

import torchvision
import torchvision.transforms as transforms

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
    '''converts a (batch_size,1,4,4) tensor into a (batch_size,4,2,2) tensor'''
    batch_size, c, h, w = x.size()
    x = x.reshape(batch_size, c, h//2, 2, w//2, 2)
    x = x.permute(0,1,3,5,2,4)
    x = x.reshape(batch_size, c*4, h//2, w//2)
    return x

def unsqueeze(x):
    '''converts a (batch_size,4,2,2) tensor into a (batch_size,1,4,4) tensor'''
    batch_size, c, h, w = x.size()
    x = x.reshape(batch_size, c//4, 2, 2, h, w)
    x = x.permute(0,1,4,2,5,3)
    x = x.reshape(batch_size, c//4, h*2, w*2)
    return x

class MoonDataset(Dataset):
    def __init__(self, n_sample: int, shuffle=None, noise=None, random_state=None, transform=None, download=False):
        directory = 'a'
        file_name = 'moon_' + str(n_sample) + '_' + str(shuffle) + '_' + str(noise) + '_' + str(random_state) + '_' + str(transform) + '.csv'
        path = os.path.join(directory, file_name)

        if download:
            if os.path.exists(path):
                samples = np.loadtxt(path, dtype=np.float32, delimiter=',')
            else:
                samples, _ = datasets.make_moons(n_sample, shuffle=shuffle, noise=noise, random_state=random_state)
                samples = samples.astype(np.float32)
                np.savetxt(path, self.samples, delimiter=',', fmt='%f')

        else:
            samples, _ = datasets.make_moons(n_sample, shuffle=shuffle, noise=noise, random_state=random_state)
            samples = samples.astype(np.float32)

        samples[:, 0] = samples[:, 0] - np.mean(samples[:, 0])
        samples[:, 1] = samples[:, 1] - np.mean(samples[:, 1])
        samples[:, 0] = samples[:, 0] / (np.max(samples[:, 0]) - np.min(samples[:, 0]))
        samples[:, 1] = samples[:, 1] / (np.max(samples[:, 1]) - np.min(samples[:, 1]))
        self.samples = samples

        self.noise = noise
        self.transform = transform

    def __len__(self):
        """Returns the number of samples"""
        return np.size(self.samples, 0)

    def __getitem__(self, idx):
        """Get the idx sample of data"""
        return self.samples[idx]

    def show(self):
        plt.figure()
        plt.plot(self.samples[:, 0], self.samples[:, 1], '.')
        plt.title('MoonDataset for ' + str(len(self)) + ' samples with Gaussian noise std = ' + str(self.noise))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


class CouplingLayer(nn.Module):
    def __init__(self, input_channels, d_channels, mask_type, reverse=False):
        super().__init__()

        self.reverse = reverse
        self.d = d_channels
        self.input_size = input_channels
        self.mask_type = mask_type

        conv1 = nn.Conv2d(input_channels, d_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(d_channels, input_channels, kernel_size=3, padding=1)
        norm_in = nn.BatchNorm2d(input_channels)
        norm_out = nn.BatchNorm2d(input_channels)

        list_t = [norm_in, conv1, nn.ReLU(), conv2, norm_out]
        self.t = nn.Sequential(*list_t)
        list_s = [norm_in, conv1, nn.ReLU(), conv2, norm_out, nn.Tanh()]
        self.s = nn.Sequential(*list_s)

    def forward(self, x):
        #x = torch.Tensor(x)
        size = torch.Tensor.size(x)
        
        if self.mask_type=='checkerboard':
            b = checkerboard_mask(size[-2], size[-1], reverse_mask=self.reverse)
            
        if self.mask_type=='channel_wise':
            b = channel_mask(self.input_size, reverse_mask=self.reverse)
            
        b_x = torch.mul(x, b)
        s_x = self.s(b_x) * (1-b)
        t_x = self.t(b_x) * (1-b)
        z = (1 - b) * (x - t_x) * torch.exp(-s_x) + b_x      
        det_J = s_x.view(s_x.size(0), -1).sum(-1)

        return z, det_J
    
    def inverse(self, y):
        # y = torch.Tensor(y)
        size = torch.Tensor.size(y)
        if self.mask_type=='checkerboard':
            b = checkerboard_mask(size[-2], size[-1], reverse_mask=self.reverse)
            
        if self.mask_type=='channel_wise':
            b = channel_mask(self.input_size, reverse_mask=self.reverse)
        b_x = torch.mul(y, b)
        s_x = self.s(b_x)
        t_x = self.t(b_x)
        z = b_x + (1 - b) * (y * torch.exp(s_x) + t_x)
        det_J = (-s_x).view(s_x.size(0), -1).sum(-1)

        return z, det_J
    
class RNVP(nn.Module):
    def __init__(self, input_size, d):
        super().__init__()
        layer_0 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_1 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)
        layer_2 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        
        #self.squeeze = nn.Conv2d(1, 4, kernel_size=3, padding=1)
                
        layer_3 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=False)
        layer_4 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=True)
        layer_5 = CouplingLayer(4, 128, mask_type='channel_wise', reverse=False)
        
        #self.de_squeeze1 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        
        layer_6 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_7 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)
        layer_8 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=True)
        layer_9 = CouplingLayer(1, 64, mask_type='checkerboard', reverse=False)
        
        #self.de_squeeze2 = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        self.input_size = input_size
        self.d = d
        self.layers_check1 = nn.ModuleList([layer_0, layer_1, layer_2])
        self.layers_channel = nn.ModuleList([layer_3, layer_4, layer_5])
        self.layers_check2 = nn.ModuleList([layer_6, layer_7, layer_8, layer_9])


    def forward(self, x: torch.Tensor):
        #print('------enter RNVP forward-----')
        #x = x[0]
        y = x
        #print('data : ', y.size())
        sum_det_J = torch.zeros(len(y))
        for i in range(len(self.layers_check1)):
            y, det_J = self.layers_check1[i].forward(y)
            sum_det_J += det_J
        
        #print('after check1 : ', y.size())    
        y = squeeze(y)
        #print('after squeeze : ', y.size())
        
        for i in range(len(self.layers_channel)):
            y, det_J = self.layers_channel[i].forward(y)
            sum_det_J += det_J
            
        #print('after channel : ', y.size())    
        y = unsqueeze(y)
        #print('after unsqueeze : ', y.size())
        
        for i in range(len(self.layers_check2)):
            y, det_J = self.layers_check2[i].forward(y)
            sum_det_J += det_J
            
        #print('after check2 : ', y.size())
        return y, sum_det_J
    
    def inverse(self, y: torch.Tensor):
        x = y
        sum_det_J = torch.zeros(len(y))
        
        for i in range(1, len(self.layers_check2)+1):
            x, det_J = self.layers_check2[len(self.layers_check2)-i].inverse(x)
            sum_det_J += det_J
        
        x = squeeze(x)
        
        for i in range(1, len(self.layers_channel)+1):
            x, det_J = self.layers_channel[len(self.layers_channel)-i].inverse(x)
            sum_det_J += det_J
            
        x = unsqueeze(x) 
        
        for i in range(1, len(self.layers_check1)+1):
            x, det_J = self.layers_check1[len(self.layers_check1)-i].inverse(x)
            sum_det_J += det_J

        return x, sum_det_J
    
class NLL(nn.Module):
    def __init__(self):
        super(NLL, self).__init__()
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
    def forward(self, z, det_J):
        #print('-------enter loss-------')
        #print('size z : ', z.size())
        #print('self.prior.log_prob(z) : ', self.prior.log_prob(z).size())
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        #print('log_pz (128?): ', log_pz.size())
        log_px = det_J + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / np.prod(z.shape[1:])
        return bpd.mean()


def train_one_epoch(model: nn.Module, train_loader: data.DataLoader, optimizer):
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


def index_statistics(samples):
    mean = torch.mean(samples)
    diffs = samples - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skew = torch.mean(torch.pow(zscores, 3.0))
    kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0

    return mean, std, skew, kurtosis


def train_one_epoch_image(model: nn.Module, train_loader: data.DataLoader, optimizer):
    losses = []
    print('-------- ENTER TRAIN ONE EPOCH --------')

    for x, i in train_loader:
        #print('start new iteration on train loader')
        #print('data x : ', x.size())
        #print('data i : ', i.size())
        optimizer.zero_grad()

        # forward pass
        y, det_J = model(x)
        #print('output y : ', y.size())

        # Loss function
        loss = NLL()
        output = loss(y, det_J)

        # update the model
        output.backward()
        optimizer.step()

        # collect statistics
        losses.append(output.detach())
        #print('Loss of batch : ', output.detach())

    epoch_loss = torch.mean(torch.tensor(losses))
    print('Loss value : ', epoch_loss)
    print('--------- OUT TRAIN ONE EPOCH ---------')

    return float(epoch_loss)


# In[7]:


def train_apply(model, dataset: str, n_train=1000, epochs=10, batch_size=32, lr=1e-3, 
                momentum=0.0, transformation=None):
    
    train_dataset = torchvision.datasets.MNIST('/home/space/datasets', train=True,
                                               transform=transformation, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torchvision.datasets.MNIST('/home/space/datasets', train=False,
                                              transform=transformation, download=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
   
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad is True], lr=lr)

    # Training metrics
    epoch_loss = []

    # Train the model epochs * times & Collect metrics progress over the training
    for i in range(epochs):
        print('Epoch ' + str(i))
        if dataset == "MNIST":
            epoch_loss_i = train_one_epoch_image(model, train_loader, optimizer)
        epoch_loss.append(epoch_loss_i)

    arr_epoch_loss = np.array(epoch_loss)

    return arr_epoch_loss


# In[8]:

CURRENT_DIR = os.getcwd()

batch_size=800
epochs=50
print('CREATING THE MODEL')
model_rnvp = RNVP(1,4)
out = train_apply(model=model_rnvp, n_train=10, dataset='MNIST', epochs=epochs, batch_size=batch_size,
                      lr=1e-5, transformation = transforms.Compose([transforms.ToTensor()]))

print('FIGURE - TRAINING LOSS PER EPOCH')
directory = CURRENT_DIR
file_name = f'jupyter_epochs_loss_MNIST_{epochs}_epochs_{batch_size}_batchsize_1e-5_lr.png'
path = os.path.join(directory, file_name)
plt.figure()
plt.title('Loss per epoch for MNIST dataset\n'
          f'Hyper-parameters : {epochs} epochs, batch_size = {batch_size} & lr=1e-5 (Adam)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(out, '-')
plt.savefig(path)
plt.show()

print('SAVING THE MODEL')
directory = CURRENT_DIR
file_name = f'jupyter_model_trained_MNIST_{epochs}_epochs_{batch_size}_batchsize_1e-5_lr.pth'
path = os.path.join(directory, file_name)
torch.save(model_rnvp.state_dict(), path)


"""from random import randrange
test_set = torchvision.datasets.MNIST('/Users/alicebatte/Desktop', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = data.DataLoader(test_set, batch_size=128, shuffle=True)
rand_in_batch = randrange(0, 128)
print(rand_in_batch)
for x, i in test_loader:
    print('data x : ', x.size())
    print('data i : ', i.size())
    out_im = model_rnvp(x)
    print('out_im ', out_im[0].size())
    out_im = out_im[0][rand_in_batch].detach().numpy()
    print('image :', np.shape(out_im))

    plt.figure()
    plt.suptitle(f'Image passed through the model for MNIST dataset\n'
              f'Hyper-parameters :{10} epochs, batch_size = {128} & lr={1e-5} (Adam)')
    plt.subplot(1,2,1)
    plt.imshow(x[rand_in_batch].detach().numpy()[0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_im[0], cmap='gray')
    break
    
gauss_im = torch.from_numpy(np.float32([[np.random.randn(test_loader.dataset[0][0][0].size(0), 
                                                                       test_loader.dataset[0][0][0].size(1))]]))
im_invert = model_rnvp.inverse(gauss_im)
exit_data = im_invert[0].detach().numpy()[0][0]
plt.figure()
plt.imshow(exit_data)


# In[9]:


print('TESTING THE REVERSE MODEL')
#model_rnvp = RNVP(1,4)
#model_rnvp.load_state_dict(torch.load('/Users/alicebatte/Desktop/model_test'))
test_dataset = torchvision.datasets.MNIST('/Users/alicebatte/Desktop', train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=True)

n = np.random.randn(test_loader.dataset[0][0][0].size(0), test_loader.dataset[0][0][0].size(0))
y = torch.from_numpy(np.float32([[n]]))
reconstruct = model_rnvp.inverse(y)

plt.figure(2)
plt.title('image dataset')
plt.imshow(test_loader.dataset[18][0][0], cmap='gray')

plt.figure(3)
plt.title('RNormal distributed image that will pass through the reverse model')
plt.imshow(n, cmap='gray')

plt.figure(4)
plt.title('Reconstructed image from a normal distribution')
plt.imshow(reconstruct[0].detach().numpy()[0][0], cmap='gray')


# In[10]:


for i, x in enumerate(test_dataset):
    exit_data = model_rnvp(x[0])
    exit_data = exit_data[0].detach().numpy()
    plt.figure(i)
    plt.title('Test output of the trained network')
    plt.imshow(exit_data[0, 0], cmap='gray')
    if i > 2:
        break
        
        
plt.figure(10)
y = torch.from_numpy(np.float32(exit_data))
reconstruct = model_rnvp.inverse(y)
plt.title('Test reverse of the trained network')
plt.imshow(reconstruct[0].detach().numpy()[0][0], cmap='gray')"""




