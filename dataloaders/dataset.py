import cv2
import torch
import numpy as np
import glob
import nibabel as nib
from torch.utils.data import Dataset
from PIL import Image
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb

def random_rot_flip(image, label):

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # print(type(image))
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


        
class Defect(Dataset):
    """ Defect Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self.split = split
        self.transform = transform
        self.sample_list = []

        self.image_list = base_dir
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx] # index for image
        label_path = image_path.replace('jpg','png')
        image = np.array(Image.open(image_path), dtype='float32') 
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA) 
        label = np.array(Image.open(label_path), dtype='uint8') 
        label = cv2.resize(label, (256,256), interpolation=cv2.INTER_AREA) 
        label[label>=128]=255
        label[label<128]=0
        
        if self.split == 'val':
            image = image/255
            label = label/255

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}

class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return {'image': torch.from_numpy(np.expand_dims(sample['image'],axis=0)),
                'label': torch.from_numpy(np.expand_dims(sample['label'],axis=0))}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size): # label unlabel 8 4
        self.primary_indices = primary_indices # [0,1,...,7]
        self.secondary_indices = secondary_indices # [8,9,...,79]
        self.secondary_batch_size = secondary_batch_size # 4
        self.primary_batch_size = batch_size - secondary_batch_size # 4

        assert len(self.primary_indices) >= self.primary_batch_size > 0 # 8 > 4
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0 # 74 > 4

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices) # 随机排列序列
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size # 2 迭代两次


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)