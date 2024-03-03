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

def random_translation(data):
    data1 = np.zeros((128, 128, 128),dtype=np.float32)
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    data1[2:125,2:125,2:125] = data[2+i:125+i,2+j:125+j,2+z:125+z]
    return data1

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
        # if self.name == "train":
        #     data = random_translation(data)
        return data, target, data_name
    

# class OneDataset(Dataset):
#     def __init__(self, root_t1 = config.whole_t1, task =  config.train, name = "train"):
#         self.root_t1 = root_t1
#         self.name = name
#         self.t1 = os.listdir(self.root_t1)
#         self.length_dataset = len(self.t1)

#     def __len__(self):
#         return self.length_dataset

#     def __getitem__(self, index):
#         t1_name = self.t1[index % self.length_dataset]
#         path_t1 = os.path.join(self.root_t1, t1_name)
#         t1 = nifti_to_numpy(path_t1)
#         if self.name == "train":
#             t1 = random_translation(t1)
#         return t1,t1_name

# class ThreeDataset(Dataset):
#     def __init__(self, root_FDG = config.whole_FDG, root_AV45 = config.whole_AV45, root_Tau = config.whole_Tau, name = "train"):
#         self.root_FDG = root_FDG
#         self.root_AV45 = root_AV45
#         self.root_Tau = root_Tau
#         self.name = name
#         self.FDG = os.listdir(self.root_FDG)
#         self.AV45 = os.listdir(self.root_AV45)
#         self.Tau = os.listdir(self.root_Tau)
#         self.length_dataset = min(len(self.FDG),len(self.AV45),len(self.Tau))

#     def __len__(self):
#         return self.length_dataset

#     def __getitem__(self, index):
#         FDG_name = self.FDG[index % self.length_dataset]
#         path_FDG = os.path.join(self.root_FDG, FDG_name)
#         FDG = nifti_to_numpy(path_FDG)
#         if self.name == "train":
#             FDG = random_translation(FDG)
#         AV45_name = self.AV45[index % self.length_dataset]
#         path_AV45 = os.path.join(self.root_AV45, AV45_name)
#         AV45 = nifti_to_numpy(path_AV45)
#         if self.name == "train":
#             AV45 = random_translation(AV45)
#         Tau_name = self.Tau[index % self.length_dataset]
#         path_Tau = os.path.join(self.root_Tau, Tau_name)
#         Tau = nifti_to_numpy(path_Tau)
#         if self.name == "train":
#             Tau = random_translation(Tau)
#         return FDG,AV45,Tau,FDG_name,AV45_name,Tau_name