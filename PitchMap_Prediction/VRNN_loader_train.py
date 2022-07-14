#
# Copyright (c) Facebook, Inc. and its affiliates.
#
import os
from glob import glob
import warnings
import parse
import numpy as np
import cv2
from torch.utils.data import Dataset
from numpy import *
import pandas as pd


class load_data_train(Dataset):
    def __init__(self):
        self.batch_size = 64
        self.data = np.load('')
        num_of_seq = (self.data.shape[0]//self.batch_size)*self.batch_size
        self.data = self.data[:num_of_seq]
        del num_of_seq
        
    def __getitem__(self, index):
        x = self.data[index]
        x = np.array(x).astype(np.float32)
        return (x,index)
    
    def __len__(self):
#        print(len(self.data))
        return len(self.data)
