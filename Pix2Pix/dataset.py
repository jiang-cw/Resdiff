import os
from torch.utils.data import Dataset
import numpy as np
import config
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

def minmaxnorm(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def random_translation(data_1, data_2):
    data1 = np.zeros((256, 256, 128),dtype=np.float32)
    data2 = np.zeros((256, 256, 128),dtype=np.float32)
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    data1[2:253,2:253,2:125] = data_1[2+i:253+i,2+j:253+j,2+z:125+z]
    data2[2:253,2:253,2:125] = data_2[2+i:253+i,2+j:253+j,2+z:125+z]
    return data1, data2

class OneDataset(Dataset):
    def __init__(self, data_path = config.train, target_path = config.train_target, name = "train"):
        self.name = name
        self.data_path = data_path
        self.target_path = target_path
        self.data = os.listdir(data_path)
        self.length_dataset = len(self.data)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        data_name = self.data[index % self.length_dataset] #+ ".nii"
        # print(t1_name)
        data = os.path.join(self.data_path, data_name)
        data = nifti_to_numpy(data)
        target = os.path.join(self.target_path, data_name[:-5]+"1.nii")
        target = nifti_to_numpy(target)
        # data = downsample(data)
        # target = downsample(target)
        if self.name == "train":
            data, target = random_translation(data, target)
        return data, target, data_name