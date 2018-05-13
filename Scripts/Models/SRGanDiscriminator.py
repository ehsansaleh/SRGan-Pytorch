import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DiscMidLego(nn.Module):
    def __init__(self, inp_channels=64, n=64, k=3, s=1):
        super(DiscMidLego, self).__init__()
        p = int(k/2)
        self.conv = nn.Conv2d(inp_channels, n, kernel_size=k, padding=p, stride=s, bias=True)
        self.bn = nn.BatchNorm2d(n)
        self.lrelu = nn.LeakyReLU()
                
    def forward(self,x):
        out = self.lrelu(self.bn(self.conv(x)))
        return out


class DiscMidBlocks(nn.Module):
    def __init__(self, init_ch=64, B=4, k=3):
        super(DiscMidBlocks, self).__init__()
        
        assert B>=1
        
        init_ch=64
        ch_list=[init_ch*(2**i) for i in range(4)]
        p=int(k/2)
        self.first_conv = nn.Conv2d(3, init_ch, kernel_size=k, padding=p, stride=1, bias=True)
        self.first_lrelu = nn.LeakyReLU()
        
        self.building_Blocks = nn.Sequential()
        self.building_Blocks.add_module('Middle_Block_Second_0', DiscMidLego(inp_channels=init_ch, n=init_ch, k=k, s=2))
        
        for b in range(1,B):
            self.building_Blocks.add_module('Middle_Block_First_' + str(b) , 
                                            DiscMidLego(inp_channels=ch_list[b-1], n=ch_list[b], k=k, s=1))
            self.building_Blocks.add_module('Middle_Block_Second_' + str(b) , 
                                            DiscMidLego(inp_channels=ch_list[b]  , n=ch_list[b], k=k, s=2))
        
        
    def forward(self,x):
        y = self.first_conv(x)
        z = self.first_lrelu(y)
        out = self.building_Blocks.forward(z)
        return out

class Discriminator(nn.Module):
    """Discriminator Network

    Args:
        init_ch_expansion (int): The number of output channels for the first Lego
            It is 64 in the original paper, which is the default here.
        B (int): The number of Blocks at the beginning of the network.
            The default value is 4, which is the same as the original paper.
            We call each pair of legos a "block".
            In the original paper:
                -There is one pair of legos (i.e. one block) at the very beginning
                    (where it is unique, since the first lego does not have BN).
                -Afterwards, there are 3 pairs of legos(i.e. 3 blocks).
                    Each block doubles the number of channels, and halves channel dimensions.
        k (int): The kernel size for convolution layers in the First Blocks.
        
        fcn_kernel (int): The dense layers of the original paper were substituted by convolution layers, 
            with same functionality.
            
            This was done so that any input image with an arbitrary size can be fed to the network.
            Example: In order to create a dense layer with 1024 output neurons, suppose we have:
                - Orignial input image size of 96*96.
                - This means that the output of the first blocks would be 512 channels, each with a size of 6*6.
                - Then, in order to have a completely dense layer, we should set the convulotional kernel size equal to 6, and 
                    produce 1024 output channels.
            
            This fcn_kernel will only be important at training time, and deciding the network architecture hyperparams.
            Otherwise, our implementation lets any input image with an arbitrary size to be able to go through the network.
            
            If the network was created with the assumption that an image input size of 96 should get a dense layer in the
            middle, then the output for an image of size 97 would be the average output of four corner 96*96 images 
            of the input 97*97 image.
            
            For a complete dense layer in the middle, make sure the following condition holds
            
            fcn_kernel = input_image_dim / (2^B)
            
            For instance: 6 = 96 / (2^4)
            
        dense_nuerons (list of ints): The list of number of neurons in the dense layer. The number of layers is determined by the length 
        of this list.
        (Default is [1024], as is in the original paper)
        For example: [1024] :::: 512 Channels -> 1024 Channels(i.e. Neurons) -> 1 Output Channel(i.e. Neuron)
        For example: [1024, 256] :::: 512 Channels -> 1024 Channels(i.e. Neurons) 
                                      -> 256 Channels(i.e. Neuron) -> 1 Output Channel(i.e. Neuron)
            
            
    """
    def __init__(self, init_ch_expansion=64, B=4, k=3, fcn_kernel=6, dense_nuerons=[1024]):
        super(Discriminator, self).__init__()
        
        assert len(dense_nuerons)>0
        
        self.midblocks = DiscMidBlocks(init_ch=init_ch_expansion, B=B, k=k)
        self.pre_fcn_channels = init_ch_expansion * (2**(B-1))
        
        curr_ch = self.pre_fcn_channels
        curr_ker = fcn_kernel
        
        self.main = nn.Sequential()
        self.main.add_module('MidBlocks', self.midblocks)
        
        for i,layer_ch in enumerate(dense_nuerons):
            self.main.add_module('FCN_layer_' + str(i) , nn.Conv2d(curr_ch, layer_ch, padding = 0,
                                                                  kernel_size = curr_ker, bias=True))
            self.main.add_module('FCN_layer_lrelu_' + str(i) , nn.LeakyReLU())
            
            curr_ch = layer_ch
            curr_ker = 1
        
            
        self.main.add_module('FCN_Final_layer', nn.Conv2d(dense_nuerons[-1], 1, padding = 0,
                                                         kernel_size = 1, bias = True))
        
        
                
    def forward(self,x):
        output = self.main(x)
        return output


## 32 * 3 * 96*96 --> 32 * 3 * 1*1
## 32 * 3 * 97*97 --> 32 * 3 * 2*2

#m = Discriminator(init_ch_expansion=64, B=4, k=3, fcn_kernel=6, dense_nuerons=[1024])
#m.apply(conv_init)
#a = np.random.random((32,3,97,97))

#myinput = Variable(torch.from_numpy(a).float())
#output = m(myinput)

#print(output.data.numpy().shape)
#myout = output.view(-1).data.numpy()

##For Discriminator
#a =  (9 * 3   + 1) * 64
#a += (9 * 64  + 1) * 64  + 2 * 64
#a += (9 * 64  + 1) * 128 + 2 * 128
#a += (9 * 128 + 1) * 128 + 2 * 128
#a += (9 * 128 + 1) * 256 + 2 * 256
#a += (9 * 256 + 1) * 256 + 2 * 256
#a += (9 * 256 + 1) * 512 + 2 * 512
#a += (9 * 512 + 1) * 512 + 2 * 512
#a += (6 * 6 * 512 + 1) * 1024
#a += 1024 + 1
#print(count_parameters(net))
#print(a)
