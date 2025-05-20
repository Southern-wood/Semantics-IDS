import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp

class Dataset_Setter(Dataset):
    def __init__(self, normal_path=None, attack_path=None, validation_path=None, labels_path=None, win_size=30, step=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        # self.normalization = StandardScaler()
        
        if normal_path is not None:
            self.train = np.load(normal_path)
            # self.normalization.fit(self.train)
            # self.train = self.normalization.transform(self.train)
        if attack_path is not None:
            self.test = np.load(attack_path)
            # self.test = self.normalization.transform(self.test)
        if validation_path is not None:
            self.val = np.load(validation_path) 
            # self.val = self.normalization.transform(self.val)
        if labels_path is not None:
            self.test_labels = np.load(labels_path)
            self.test_labels = (self.test_labels.max(axis=1) >= 1).astype(int)
        self.flag = mode

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), \
                np.int32(self.test_labels[index:index + self.win_size].max())
        
def get_loader_segment(normal_path=None, attack_path=None, validation_path=None, labels_path=None, batch_size=128, win_size=30, step=1, mode='train', dataset='SWaT', noshuffle=False):
    mp.set_start_method('spawn', force=True)
    dataset = Dataset_Setter(normal_path, attack_path, validation_path, labels_path, win_size, step, mode)

    shuffle = (mode == 'train') and not noshuffle


    generator = None
    if shuffle:
        generator = torch.Generator()

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=8,
                             generator=generator,
                             drop_last=True)
    return data_loader
