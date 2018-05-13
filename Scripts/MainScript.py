
# coding: utf-8

# In[11]:


import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
from skimage.transform import resize, downscale_local_mean
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from Utilities.data import DataWrapper
from Models.SRGanGenerator import Generator
from Models.SRGanDiscriminator import Discriminator
from Operations.Losses import BCEWithLogitsLoss, SoftMarginLoss, VGGFeatureExtractor


# In[12]:


IS_GPU=True
RandomSeed = 123456
np.random.seed(RandomSeed)
torch.cuda.manual_seed_all(RandomSeed)
torch.manual_seed(RandomSeed)


# In[44]:


training = DataWrapper(batch_size=32, random_crop_size=96, down_scale_factor=4, 
                       root_dir='/mnt/a/u/sciteam/saleh1/work/srgan/Data/TrainingHR',
                       loader_shuffle=False, loader_workers=4)

evaluation = DataWrapper(batch_size=1, random_crop_size=None, down_scale_factor=4,
                         root_dir='/mnt/a/u/sciteam/saleh1/work/srgan/Data/EvaluationHR',
                         loader_shuffle=False, loader_workers=1)


# In[59]:


class ExtendedModelWithUtilities(object):
    def __init__(self, model_type = 'SRGAN', SRType='MSE', batch_loss_sum_or_mean = 'mean',
                 gen_learning_rate = 10 ** -6, beta1 = 0.9, beta2 = 0.999, 
                 adv2content_loss_ratio = None, disc_learning_rate = None, IS_GPU = True,
                 RandomSeed = 123456):
        
        self.model_type = model_type
        self.SRType = SRType
        self.batch_loss_sum_or_mean = batch_loss_sum_or_mean
        self.gen_learning_rate = gen_learning_rate
        if self.model_type == 'SRGAN':
            if adv2content_loss_ratio == None:    
                self.adv2content_loss_ratio = 10 ** -3
            else:
                self.adv2content_loss_ratio = adv2content_loss_ratio
                
            if disc_learning_rate == None:
                self.disc_learning_rate = 10 ** -6
            else:
                self.disc_learning_rate = disc_learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.IS_GPU = IS_GPU
        self.RandomSeed = RandomSeed
        
        self.SRTypeDict = {'VGG54': {'i':5, 'j':4, 'net':'vgg19', 'scale':12.75},
                           'VGG22': {'i':2, 'j':2, 'net':'vgg19', 'scale':12.75},
                           'MSE' : {'scale':1}}
        
        #Creating the Models
        self.CreateModels(gen_arch = {'first_stage_hyperparams' : {'k':9, 'n':64, 's':1}, 
                                      'residual_blocks_hyperparams' : {'k':3, 'n':64, 's':1, 'B':16}, 
                                      'upsample_blocks_hyperparams' : {'k':3, 'n':256, 's':1, 'B':2, 'f':2}, 
                                      'last_stage_hyperparams' : {'k':9, 's':1}},
                          
                          disc_arch = {'init_ch_expansion' : 64,
                                       'B' : 4, 
                                       'k' : 3,
                                       'fcn_kernel' : 6,
                                       'dense_nuerons' : [1024]},
                          
                          SRType = self.SRType)
        
        #Creating the Optimizers
        self.CreateOptimizers()
        
        #Initializing the Models
        self.InitializeModels(RandomSeed = self.RandomSeed)
        self.epoch = 0
        self.step = 0
        
        #Transporting to GPU if necessary
        if self.IS_GPU:
            self.trasnport_models_to_gpu()
        
        #Defining the losses
        self.CreateLosses()
        
        #Creating Statistics
        self.init_step_loss_statistics()
        self.init_epoch_loss_statistics()
            
        
                
    def CreateModels(self, gen_arch, disc_arch, SRType):
        self.ModelRegistry = {}
        
        self.gen_net = Generator(first_stage_hyperparams = gen_arch['first_stage_hyperparams'], 
                                 residual_blocks_hyperparams = gen_arch['residual_blocks_hyperparams'], 
                                 upsample_blocks_hyperparams = gen_arch['upsample_blocks_hyperparams'],
                                 last_stage_hyperparams = gen_arch['last_stage_hyperparams'])
        self.ModelRegistry['Generator'] = self.gen_net

        if self.model_type == 'SRGAN':
            self.disc_net = Discriminator(init_ch_expansion = disc_arch['init_ch_expansion'],
                                          B = disc_arch['B'],
                                          k = disc_arch['k'],
                                          fcn_kernel = disc_arch['fcn_kernel'], 
                                          dense_nuerons = disc_arch['dense_nuerons'])
            self.ModelRegistry['Discriminator'] = self.disc_net

        if SRType.upper().startswith('VGG'):
            self.FeatureExtractor = VGGFeatureExtractor(vggname = self.SRTypeDict[SRType]['net'],
                                                   i = self.SRTypeDict[SRType]['i'],
                                                   j = self.SRTypeDict[SRType]['j'])
            self.ModelRegistry['Feature Extractor'] = self.FeatureExtractor
    
    def CreateOptimizers(self):
        self.GenOptimizer = optim.Adam(self.gen_net.parameters(), lr=self.gen_learning_rate, 
                                       betas=(self.beta1, self.beta2))
        if self.model_type == 'SRGAN':
            self.DiscOptimizer = optim.Adam(self.disc_net.parameters(), lr=self.disc_learning_rate, 
                                            betas=(self.beta1, self.beta2))        

            
    def InitializeModels(self, RandomSeed = 123456):
        if not (RandomSeed == None):
            np.random.seed(RandomSeed)
            torch.cuda.manual_seed_all(RandomSeed)
            torch.manual_seed(RandomSeed)

        self.gen_net.apply(self.conv_init)
        if self.model_type == 'SRGAN':
            self.disc_net.apply(self.conv_init)
        
    def conv_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_uniform(m.weight, gain=np.sqrt(2))
            init.constant(m.bias, 0)
        elif classname.find('BatchNorm') != -1:
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def trasnport_models_to_gpu(self):
        import torch.backends.cudnn as cudnn
        
        self.gen_net = self.gen_net.cuda()
        self.gen_net = torch.nn.DataParallel(self.gen_net, device_ids=range(torch.cuda.device_count()))

        if self.model_type == 'SRGAN':
            self.disc_net = self.disc_net.cuda()
            self.disc_net = torch.nn.DataParallel(self.disc_net, device_ids=range(torch.cuda.device_count()))
            
        if self.SRType.upper().startswith('VGG'):
            self.FeatureExtractor = self.FeatureExtractor.cuda()
            self.FeatureExtractor = torch.nn.DataParallel(self.FeatureExtractor, device_ids=range(torch.cuda.device_count()))
            
        cudnn.benchmark = True
    
    def CreateLosses(self):
        ########################################################################
        # Define a Loss function and optimizer
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        my_size_average = (self.batch_loss_sum_or_mean == 'mean')
        self.NegLogSigmoidProdCriterion = SoftMarginLoss(size_average=my_size_average,reduce=True)
        self.BCEWithLogitCriterion = BCEWithLogitsLoss(size_average=my_size_average,reduce=True)
        self.MSECriterion = nn.MSELoss(size_average=my_size_average, reduce=True)
    
    def init_step_loss_statistics(self):
        self.step_loss_statistics = {}
        
        self.step_loss_statistics['Generator Perceptual Loss'] = []
        self.step_loss_statistics['Generator Content Loss'] = []
        
        if self.model_type == 'SRGAN':
            
            self.step_loss_statistics['Generator Adverserial Loss'] = []
            self.step_loss_statistics['Discriminator Total Loss'] = []
            self.step_loss_statistics['Discriminator Real Loss'] = []
            self.step_loss_statistics['Discriminator Fake Loss'] = []
            self.step_loss_statistics['Discriminator Real Logit Output'] = []
            self.step_loss_statistics['Discriminator Fake Logit Output'] = []
    
    def print_step_loss_statistics(self):
        information_dict = self.step_loss_statistics
        print('Epoch: '+ str(self.epoch) +'\tStep: ' + str(self.step))
        for key in information_dict.keys():
            print('\t' + key + ':\t\t\t' + str(information_dict[key][-1]))
        print('-------------------------------------------------')
        
    def print_epoch_loss_statistics(self):
        information_dict = self.epoch_loss_statistics
        print('Epoch: '+ str(self.epoch))
        for key in information_dict.keys():
            print('\t Average ' + key + ':\t\t\t' + str(information_dict[key]))
        print('-------------------------------------------------')
            
    def init_epoch_loss_statistics(self):
        self.epoch_loss_statistics = {}
        
    def update_epoch_loss_statistics(self):
        for key in self.step_loss_statistics.keys():
            self.epoch_loss_statistics[key] = np.mean(np.array(self.step_loss_statistics[key]))
        
    def UpdateDiscriminator(self, SR_im, HR_im):
        self.DiscOptimizer.zero_grad()

        disc_real_pred_logit = self.disc_net(HR_im)
        disc_fake_pred_logit = self.disc_net(SR_im.detach())

        disc_real_loss = self.BCEWithLogitCriterion(disc_real_pred_logit, torch.ones_like(disc_real_pred_logit))
        disc_fake_loss = self.BCEWithLogitCriterion(disc_fake_pred_logit, torch.zeros_like(disc_fake_pred_logit))

        disc_loss = disc_real_loss + disc_fake_loss

        disc_loss.backward()
        self.DiscOptimizer.step()
        
        #Reporting the statistics to the class
        self.step_loss_statistics['Discriminator Total Loss'].append(disc_loss.data[0])
        self.step_loss_statistics['Discriminator Real Loss'].append(disc_real_loss.data[0])
        self.step_loss_statistics['Discriminator Fake Loss'].append(disc_fake_loss.data[0])
        self.step_loss_statistics['Discriminator Real Logit Output'].append(disc_real_loss.data[0])
        self.step_loss_statistics['Discriminator Fake Logit Output'].append(disc_fake_loss.data[0])
        
    def UpdateGenerator(self, SR_im, HR_im):
        self.GenOptimizer.zero_grad()
        HR_im = Variable(HR_im.data, volatile=True, requires_grad=False)
        if self.model_type == 'SRGAN':
            if self.SRType.upper().startswith('VGG'):
                HR_features = self.FeatureExtractor(HR_im)
                SR_features = self.FeatureExtractor(SR_im)
                content_loss_gen = self.MSECriterion(SR_features, HR_features) / (self.SRTypeDict[self.SRType]['scale']**2)
            else:
                content_loss_gen = self.MSECriterion(SR_im, HR_im)
            
            disc_fake_pred_logit = self.disc_net(SR_im)
            adverserial_loss_gen =  self.NegLogSigmoidProdCriterion(disc_fake_pred_logit, torch.ones_like(disc_fake_pred_logit))
            
            perceptual_loss_gen = content_loss_gen + adverserial_loss_gen * self.adv2content_loss_ratio
        else:
            if self.SRType.upper().startswith('VGG'):
                HR_features = self.FeatureExtractor(HR_im)
                SR_features = self.FeatureExtractor(SR_im)
                perceptual_loss_gen = self.MSECriterion(SR_features, HR_features) / (self.SRTypeDict[self.SRType]['scale']**2)
            else:
                perceptual_loss_gen = self.MSECriterion(SR_im, HR_im)
            
        perceptual_loss_gen.backward()
        self.GenOptimizer.step()
        
        #Reporting the statistics to the class
        self.step_loss_statistics['Generator Perceptual Loss'].append(perceptual_loss_gen.data[0])
        if self.model_type == 'SRGAN':
            self.step_loss_statistics['Generator Content Loss'].append(content_loss_gen.data[0])
            self.step_loss_statistics['Generator Adverserial Loss'].append(adverserial_loss_gen.data[0])
        else:
            self.step_loss_statistics['Generator Content Loss'].append(perceptual_loss_gen.data[0])
    
        
        
        


# In[60]:


def printimg2file(np_img, outpath):
    curr_img = np_img.reshape(3, np_img.shape[-2], np_img.shape[-1])
    curr_img = np.transpose(curr_img,[1,2,0])
    curr_img = (curr_img * 255).astype(np.uint8)
    io.imsave(outpath , curr_img)
    


# In[61]:


Model = ExtendedModelWithUtilities(model_type = 'SRResNet', SRType='VGG54', batch_loss_sum_or_mean = 'mean',
                                   gen_learning_rate = 10 ** -6, beta1 = 0.9, beta2 = 0.999, 
                                   adv2content_loss_ratio = 10 ** -3, disc_learning_rate = 10 ** -6, IS_GPU = True)


# In[65]:


EPOCHS=2
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    Model.init_step_loss_statistics()
    Model.step = 0
    
    for i, data in enumerate(training.loader):
        
        # get the inputs
        HR_im = data['High']
        LR_im = data['Low']

        if IS_GPU:
            HR_im = HR_im.cuda()
            LR_im = LR_im.cuda()

        # wrap them in Variable
        HR_im = Variable(HR_im)
        LR_im = Variable(LR_im)
        
        SR_im = Model.gen_net(LR_im)
        
        #Discriminator Training
        if Model.model_type == 'SRGAN':
            Model.UpdateDiscriminator(SR_im, HR_im)
            
        # Generator Training        
        Model.UpdateGenerator(SR_im, HR_im)
        
        # Print Statistics
        Model.print_step_loss_statistics()
        Model.step = Model.step + 1
        
        if i>=2:
            break
    
    eval_out_ims=[]
    for i, data in enumerate(evaluation.loader):
        HR_im = data['High']
        LR_im = data['Low']
        
        if IS_GPU:
            LR_im = LR_im.cuda()

        LR_im = Variable(LR_im, requires_grad=False, volatile=True)
        SR_im = Model.gen_net(LR_im)
        
        eval_out_ims.append(torch.squeeze((SR_im.data+1)/2, dim=0))
        
        printimg2file(np_img = (SR_im.cpu().data.numpy() + 1) / 2,
                      outpath = '/mnt/a/u/sciteam/saleh1/work/srgan/Data/GeneratedOutput/Epoch_'+str(epoch)+'_img_'+str(i)+'.png')
        
    file_name='/mnt/a/u/sciteam/saleh1/work/srgan/Data/GeneratedOutput/Epoch_'+str(epoch)+'_merged.png'
    torchvision.utils.save_image(eval_out_ims, file_name, 
                                 nrow=8, padding=2,
                                 normalize=False, range=None,
                                 scale_each=False, pad_value=0)
    
    Model.update_epoch_loss_statistics()
    Model.print_epoch_loss_statistics()
    Model.epoch = Model.epoch + 1


# In[ ]:




