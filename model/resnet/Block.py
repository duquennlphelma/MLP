import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    #creation of a block. The input size and output size are different because s and t are function from Rd in R(D-d) so we need to apply the "same" convolution
    #The convolutions are made with matrix of learnable weights
    def __init__(self, n_input_channel, n_output_channel):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channel, n_output_channel, kernel_size=3)
        self.norm_in=nn.BatchNorm2d(n_input_channel)
        self.norm_out=nn.BatchNorm2d(n_output_channel)
        self.conv2=nn.Conv2d(n_output_channel,n_output_channel, kernel_size=3)
        self.same_conv=nn.Conv2d(n_input_channel, n_output_channel, kernel_size=3)
    def forward(self, x):
        x_same=x.copy()
        x=self.norm_in(x)
        x=F.relu(x)
        x=self.conv1(x)
        x=self.norm_out(x)
        x=F.relu(x)
        x=self.conv2(x)

        x = x+ x_same

        return x