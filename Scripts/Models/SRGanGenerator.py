import torch
import torch.nn as nn


class GenResLego(nn.Module):
    def __init__(self, inp_channels=64, n=64, k=3, s=1):
        super(GenResLego, self).__init__()
        p = int(k/2)
        self.conv1 = nn.Conv2d(inp_channels, n, kernel_size=k, padding=p, stride=s, bias=True)
        self.bn1 = nn.BatchNorm2d(n)
        self.prelu = nn.PReLU(n)
        self.conv2 = nn.Conv2d(n, n, kernel_size=k, padding=1, stride=s, bias=True)
        self.bn2 = nn.BatchNorm2d(n)
                
    def forward(self,x):
        y = self.prelu(self.bn1(self.conv1(x)))
        z = self.bn2(self.conv2(y))
        out = z + x
        return out

class GenResBlock(nn.Module):
    def __init__(self, inp_channels=64, B=16, n=64, k=3, s=1):
        super(GenResBlock, self).__init__()
        
        assert B>=1
        
        self.building_Blocks = nn.Sequential()
        self.building_Blocks.add_module('Residual_Block_0' , GenResLego(inp_channels=inp_channels, 
                                                                        n=n, k=k, s=s))
        for b in range(B-1):
            self.building_Blocks.add_module('Residual_Block_' + str(b+1) , 
                                            GenResLego(inp_channels=n, n=n, k=k, s=s))
        p = int(k/2)    
        self.final_conv = nn.Conv2d(n, n, kernel_size=k, padding=p, stride=s, bias=True)
        self.final_bn = nn.BatchNorm2d(n)
        
    def forward(self,x):
        y = self.building_Blocks.forward(x)
        z = self.final_conv(y)
        out = self.final_bn(z) + x
        return out


class GenUpsampleLego(nn.Module):
    def __init__(self, inp_channels=64, n=256, k=3, s=1, upscale_factor=2):
        super(GenUpsampleLego, self).__init__()
        
        p = int(k/2)
        self.conv = nn.Conv2d(inp_channels, n, kernel_size=k, padding=p, stride=s, bias=True)
        self.upsamp = nn.PixelShuffle(upscale_factor = upscale_factor)
        self.prelu = nn.PReLU(inp_channels)
        
    def forward(self,x):
        y = self.conv(x)
        z = self.upsamp(y)
        out = self.prelu(z)
        return out

class GenUpsampleBlock(nn.Module):
    def __init__(self, upsample_B=2, inp_channels=64, n=256, k=3, s=1, upscale_factor=2):
        super(GenUpsampleBlock, self).__init__()
        
        errorlog ='The upscale factor controls the pixel shuffler and changes then number of channels. ' 
        errorlog +='Make sure that you choose the hyperparams in a way that n == upscale_factor^2 * inp_channels.'
        assert n == upscale_factor * upscale_factor * inp_channels ,  errorlog
        
        
        self.building_Blocks = nn.Sequential()
        for b in range(upsample_B):
            self.building_Blocks.add_module('Upsample_Block_' + str(b) ,
                                            GenUpsampleLego(inp_channels=inp_channels, n=n,
                                                            k=k, s=s, upscale_factor=upscale_factor))
        
    def forward(self,x):
        out = self.building_Blocks.forward(x)
        return out


class Generator(nn.Module):
    def __init__(self, first_stage_hyperparams={'k':9, 'n':64, 's':1}, 
                 residual_blocks_hyperparams={'k':3, 'n':64, 's':1, 'B':16}, 
                 upsample_blocks_hyperparams={'k':3, 'n':256, 's':1, 'B':2, 'f':2}, 
                 last_stage_hyperparams={'k':9, 's':1}, ngpu=1):
        super(Generator, self).__init__()
        
        self.ngpu = ngpu
        
        fsh = first_stage_hyperparams
        rbh = residual_blocks_hyperparams
        ubh = upsample_blocks_hyperparams
        lsh = last_stage_hyperparams
        
        fsh['p']=int(fsh['k']/2)
        lsh['p']=int(lsh['k']/2)
        self.first_stage_conv = nn.Conv2d(3, fsh['n'], kernel_size=fsh['k'], 
                                          padding=fsh['p'], stride=fsh['s'], bias=True)
        self.first_stage_prelu = nn.PReLU(fsh['n'])
        self.ResBlocks = GenResBlock(inp_channels=fsh['n'], n=rbh['n'], k=rbh['k'], s=rbh['s'], B=rbh['B'])
        self.UpscaleBlocks = GenUpsampleBlock(upsample_B=ubh['B'], inp_channels=rbh['n'], n=ubh['n'], k=ubh['k'], 
                                              s=ubh['s'], upscale_factor=ubh['f'])
        self.last_stage_conv = nn.Conv2d(rbh['n'], 3, kernel_size=lsh['k'], 
                                          padding=lsh['p'], stride=lsh['s'], bias=True)
        
        self.main = nn.Sequential(self.first_stage_conv,
                                  self.first_stage_prelu,
                                  self.ResBlocks, 
                                  self.UpscaleBlocks,
                                  self.last_stage_conv)
    def forward(self,x):       
        output = self.main(x)
        return output

##For Generator
#a =  (81*3 + 1 ) *64 +64
#a += ( (9*64 + 1)*64*2 + 5*64)*16 + (9*64 + 1)*64 + 2*64
#a += ((9*64 + 1)*256 + 64)*2
#a += (81*64+1)*3
#print(count_parameters(net))
#print(a)
