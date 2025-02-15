import torch
import torch.nn
import numpy as np
import os.path
import cv2
from PIL import Image
import itertools
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import torch.nn.functional as F
from guided_diffusion.model import *

def random_rot_flip(image, label):

    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    # print(type(image))
    return image, label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return {'image': torch.from_numpy(sample['image']).unsqueeze(0).unsqueeze(0).to(device),
                'label': torch.from_numpy(sample['label']).unsqueeze(0).unsqueeze(0).to(device)}

class ToTensor1(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return {'image': torch.from_numpy(sample['image']).unsqueeze(0).to(device),
                'label': torch.from_numpy(sample['label']).unsqueeze(0).to(device)}
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
        
class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, test_flag=True, transform=None):
        self.transform = transform
        self.sample_list = []

        self.image_list = base_dir
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx] # 所有病例的名字
        label_path = image_path.replace('jpg','png')
        image = np.array(Image.open(image_path), dtype='float32') 
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA) #图像后续会做标准化处理

        label = np.array(Image.open(label_path), dtype='float32')
        label = cv2.resize(label, (256,256), interpolation=cv2.INTER_AREA) 
        label[label>=128]=255
        label[label<128]=0
        
        image = image/255
        label = label/255
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return (sample['image'], sample['label'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Defect(Dataset):
    """ Defect Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):

        def load_net(net, path):
            state = torch.load(str(path),map_location='cpu')
            net.load_state_dict(state['state_dict'])

        self.split = split
        self.transform = transform
        self.sample_list = []

        self.image_list = base_dir
        print("total {} samples".format(len(self.image_list)))
        self.model = U_Net(1)
        self.ema_model = U_Net(1)
        self.model = self.model.to(device)
        self.ema_model = self.ema_model.to(device)
        for param in self.model.parameters(): #不需要计算梯度
                param.detach_()   
        for param in self.ema_model.parameters():
            param.detach_()  
        pretrained_model = 'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_12_labeled/pre_train/20231101-232958blowhole/the_best_model_2249_0.1670.pth'
        selfpretrained_model = 'D:/HZX/Program files/cache/1761170395/FileRecv/leo_segmentation/Industrial_defect_12_labeled/self_train/20231101-234555blowhole/the_best_model_249_0.1227.pth'
        load_net(self.model, pretrained_model)
        load_net(self.ema_model, selfpretrained_model) #两个模型加载了一样的权重




    def __len__(self):
        return len(self.image_list)
    

    def __getitem__(self, idx):

        def get_cut_mask(out, thres=0.5, nms=0):
            probs = F.sigmoid(out)
            mask = (probs >= thres).type(torch.int64) # [2, 144, 240, 240]
            mask = mask.contiguous()
            if nms == 1:
                mask = LargestCC_pancreas(mask)
            return mask

        def LargestCC_pancreas(segmentation):
            N = segmentation.shape[0]
            batch_list = []
            for n in range(N):
                n_prob = segmentation[n].detach().cpu().numpy()
                # labels = label[n_prob]
                labels = label(n_prob)
                if labels.max() != 0:
                    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                else:
                    largestCC = n_prob
                batch_list.append(largestCC)

        image_path = self.image_list[idx] # 所有病例的名字
        label_path = image_path.replace('jpg','png')
        image = np.array(Image.open(image_path), dtype='float32') 
        image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA) 
        label = np.array(Image.open(label_path), dtype='uint8') 
        label = cv2.resize(label, (256,256), interpolation=cv2.INTER_AREA) 
        label[label>=128]=255
        label[label<128]=0
        
        image = image/255
        label = label/255

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        with torch.no_grad():
            self.model.eval()
            self.ema_model.eval()
            pre1 = self.model(sample['image'])
            pre1 = get_cut_mask(pre1)
            pre2 = self.ema_model(sample['image'])
            pre2 = get_cut_mask(pre2)
            subtra1 = abs(sample['label'] - pre1)
            subtra2 = abs(pre2 - pre1)

        return sample['image'],  sample['label'],  subtra1, subtra2,  pre2
    
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size): # label unlabel 8 4
        self.primary_indices = primary_indices # 8
        self.secondary_indices = secondary_indices # 72
        self.secondary_batch_size = secondary_batch_size # 4
        self.primary_batch_size = batch_size - secondary_batch_size # 4

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

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