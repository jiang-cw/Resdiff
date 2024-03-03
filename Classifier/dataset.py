import os
from torch.utils.data import Dataset
import numpy as np
import config
import nibabel as nib
import pandas as pd
import csv

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

# real
# def crop(data1):
#     return data1[10:170,18:210,10:170]

# def random_translation(data1):
#     i=np.random.randint(-2,3)
#     j=np.random.randint(-2,3)
#     z=np.random.randint(-2,3)
#     return data1[10+i:170+i,18+j:210+j,10+z:170+z]

#syn
# def crop(data1):
#     data = np.zeros_like(data1)
#     data[2:157,2:189,2:157] = data1[2:157,2:189,2:157] 
#     return data

# def random_translation(data1):
#     i=np.random.randint(-2,3)
#     j=np.random.randint(-2,3)
#     z=np.random.randint(-2,3)
#     data = np.zeros_like(data1)
#     data[2:157,2:189,2:157]  = data1[2+i:157+i,2+j:189+j,2+z:157+z]
#     return data

# def normalization(scan): #输入的图片已归一化到0-1
#     scan = (scan - np.float32(0.5))*np.float32(2)
#     return scan

# def minmaxnorm(scan):
#     scan = (scan-np.min(scan))/(np.max(scan)-np.min(scan))
#     return scan

def train_split(file,fold,name):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    x=int(len(p)/5)
    if name == "train":
        if fold == "0":
            p = p[:x*3]
        elif fold == "1":
            p = p[x:x*4]
        elif fold == "2":
            p = p[x*2:]
        elif fold == "3":
            p = p[x*3:]+p[:x]
        elif fold == "4":
            p = p[x*4:]+p[:x*2]
    elif name == "validation":
        if fold == "0":
            p = p[x*3:x*4]
        elif fold == "1":
            p = p[x*4:]
        elif fold == "2":
            p = p[:x]
        elif fold == "3":
            p = p[x:x*2]
        elif fold == "4":
            p = p[x*2:x*3]
    elif name == "test":
        if fold == "0":
            p = p[x*4:]
        elif fold == "1":
            p = p[:x]
        elif fold == "2":
            p = p[x:x*2]
        elif fold == "3":
            p = p[x*2:x*3]
        elif fold == "4":
            p = p[x*3:x*4]
    return p

class OneDataset(Dataset):
    def __init__(self, root_AV45 = config.syn_AV45,task = config.whole,name = "train",fold = "0"):
        self.root_AV45 = root_AV45
        self.task = task
        self.name = name
        self.images = train_split(task,fold,name)
        self.length_dataset = len(self.images)
        self.len = len(self.images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        name = self.images[index % self.len]+".nii"
        AV45_path = os.path.join(self.root_AV45, name)
        AV45 = nifti_to_numpy(AV45_path)
        # if self.name == "train":
        #     AV45=random_translation(AV45)
        # else:
        #     AV45=crop(AV45)
        #AV45 = minmaxnorm(AV45)   #分类任务的输入范围为0-1
        data = pd.read_csv("data_info/Abeta_SUVR.csv",encoding = "ISO-8859-1")
        label = data[data['ID'] == name[0:-4]]['A+']
        label=label.values
        label=label.astype(np.float32)
        return AV45,label
