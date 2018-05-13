import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io
from skimage.transform import resize, downscale_local_mean
import numpy as np

class HighResolutionDataset(Dataset):
    """Preparing High Resolution dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        assert os.path.isdir(root_dir), 'Root directory does not exist: '+str(root_dir)
        self.root_dir = root_dir
        self.imnames=[name for name in os.listdir(self.root_dir) if 
                      os.path.isfile(os.path.join(self.root_dir, name)) and
                     (name.lower().endswith('.png') or name.lower().endswith('.jpg') or
                      name.lower().endswith('.bmp') or name.lower().endswith('.tif') or
                      name.lower().endswith('.tiff')) ]
        assert len(self.imnames)>0, 'No known image types found in the root directory: '+str(root_dir)
        self.transform=transform

    def __len__(self):
        return len(self.imnames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,self.imnames[idx])
        image = io.imread(img_path)
        sample = {'High': image}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['High']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'High': image}


class AddLowResolution(object):
    """Crop randomly the image in a sample.

    Args:
        scale_factor (tuple or int): Desired downscaling factor. 
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        ****The scale_factor and output_size need to be provided mutually exclusive
        ****(i.e. exactly one of them should be presented).
    """

    def __init__(self, down_scale_factor=4, output_size=None):
        assert isinstance(output_size, (int, tuple)) or isinstance(down_scale_factor, int)
        if isinstance(down_scale_factor, (int, tuple)):
            self.down_scale_factor = down_scale_factor
            if isinstance(down_scale_factor, int):
                self.down_scale_factor = (down_scale_factor, down_scale_factor)
            self.factor_exist=True
        else:
            self.factor_exist=False
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size

    def __call__(self, sample):
        hr_image = sample['High']

        h, w = hr_image.shape[:2]
        if self.factor_exist:
            new_h = int(h/self.down_scale_factor[0])
            new_w = int(w/self.down_scale_factor[1])
            output_shape = (new_h , new_w)
        else:
            output_shape = self.output_size
            
        lr_image = resize(hr_image, output_shape, order=1, mode='reflect', 
                          clip=True, preserve_range=True)
        lr_image = np.rint(lr_image).astype(hr_image.dtype)
        sample['Low'] = lr_image

        return sample
    



class NormalizeRange(object):
    """Normalizes the high and low resolution ranges.
    """

    def __init__(self, in_range = (0,255), out_range=(0,1), image_key='Low', out_type = np.float):
        assert isinstance(out_range, tuple)
        assert isinstance(in_range, tuple)
        assert len(out_range) == 2 
        assert len(in_range) == 2
        
        self.out_min = np.float(out_range[0])
        self.out_max = np.float(out_range[1])
        
        self.in_min = np.float(in_range[0])
        self.in_max = np.float(in_range[1])
        
        #Output = m * Input + b
        self.m = (self.out_max - self.out_min)/(self.in_max - self.in_min)
        self.b = self.out_min - self.m * self.in_min
        
        self.image_key = image_key
        self.out_type = out_type

    def __call__(self, sample):
        image = sample[self.image_key]

        out_image = image.astype(np.float) * self.m + self.b
        out_image = out_image.astype(self.out_type)
        
        sample[self.image_key] = out_image

        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        for curr_attr in sample.keys():
            
            curr_val=sample[curr_attr]
            
            if curr_attr=='High' or curr_attr=='Low':
                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                sample[curr_attr] = torch.from_numpy( curr_val.transpose((2, 0, 1)) ).float()
            else:
                sample[curr_attr] = torch.from_numpy( curr_val ).float()
                
        return sample


class DataWrapper(object):
    """Creates a dataset
    """

    def __init__(self, batch_size=32, random_crop_size=96, down_scale_factor=4, 
                 root_dir='/u/sciteam/saleh1/work/srgan/Data/HighResolution',
                 loader_shuffle=True, loader_workers=4):
        self.batch_size = batch_size
        self.root_dir=root_dir
        self.loader_shuffle=loader_shuffle
        self.loader_workers=loader_workers
        
        transform_list=[]
        if random_crop_size:
            transform_list.append(RandomCrop(random_crop_size))
        
        transform_list.append(AddLowResolution(down_scale_factor=down_scale_factor))
        
        transform_list.append(NormalizeRange(in_range = (0,255), out_range=(0,1),
                                             image_key='Low', out_type = np.float))
        
        transform_list.append(NormalizeRange(in_range = (0,255), out_range=(-1,1),
                                             image_key='High', out_type = np.float))
        
        transform_list.append(ToTensor())
        
        self.full_transform = transforms.Compose(transform_list)
        
        
        self.dataset = HighResolutionDataset(root_dir=self.root_dir, transform = self.full_transform)
        
        self.loader = torch.utils.data.DataLoader(self.dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=self.loader_shuffle,
                                                  num_workers=self.loader_workers)


