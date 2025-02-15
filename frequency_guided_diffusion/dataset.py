import torch.utils.data as data
import torch
import cv2
import copy
from torch.utils.data.sampler import Sampler
import numpy as np
from PIL import Image
import itertools

class Dataset(data.Dataset):
    def __init__(self, file_path):
        super(Dataset, self).__init__()
        self.file_path = file_path

    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, index):

        data = self.file_path[index]
        # data = Image.open(data_path)
        # data = np.array(data, dtype='float32') / 255 # (512,512)
        # data = data.transpose(2,0,1)[0]
        data = cv2.resize(data, (512,512), interpolation=cv2.INTER_AREA) 
        label = copy.deepcopy(data)
        return torch.from_numpy(data).unsqueeze(0), torch.from_numpy(label).unsqueeze(0)

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