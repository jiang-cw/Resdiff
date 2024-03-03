import os
from torch.utils.data import Dataset
import numpy as np
import config
import nibabel as nib
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage import zoom
import SimpleITK as sitk

def read_list(file):
    file=open(file,"r")
    S=file.read().split()
    p=list(str(i) for i in S)
    return p

def nifti_to_numpy(file):
    data = nib.load(file).get_fdata()
    data = data.astype(np.float32)
    return data

def shuffle_and_write():
    with open("./data_info/whole_raw.txt", 'r') as raw_file:
        lines = raw_file.readlines()
    random.shuffle(lines)

    with open("./data_info/whole.txt", 'w') as shuffled_file:
        shuffled_file.writelines(lines)

def dataset_split():
    with open("./data_info/whole.txt", 'r') as file:
        data = file.readlines()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, validation_data = train_test_split(train_data, test_size=0.1, random_state=42)

    with open("./data_info/train.txt", 'w') as file:
        file.writelines(train_data)

    with open("./data_info/validation.txt", 'w') as file:
        file.writelines(validation_data)

    with open("./data_info/test.txt", 'w') as file:
        file.writelines(test_data)

def minmaxnorm(data):
    return (data - np.min(data))/(np.max(data)-np.min(data))

def random_translation(data):
    data1 = np.zeros((512, 512, 256),dtype=np.float32)
    i=np.random.randint(-2,3)
    j=np.random.randint(-2,3)
    z=np.random.randint(-2,3)
    data1[2:509,2:509,2:253] = data[2+j:509+j,2+z:509+z,2+i:253+i]
    return data1

def crop(data):
    return data[127:383,127:383,:]

# def downsample(data):
#     scale_factor = (0.5, 0.5, 0.5)
#     downsampled_data = zoom(data, scale_factor, order=1)
#     return downsampled_data

def downsample(data):
    input_image = sitk.GetImageFromArray(data)
    output_size = [sz//2 for sz in input_image.GetSize()]
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing([2.0, 2.0, 2.0])  # 设置新的spacing
    resampler.SetSize(output_size)
    downsampled_data = resampler.Execute(input_image)
    downsampled_data = sitk.GetArrayFromImage(downsampled_data)
    return downsampled_data

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
        #data = downsample(data)
        #target = downsample(target)
        # if self.name == "train":
        #     data = random_translation(data)
        return data, target, data_name
    
if __name__ == "__main__":
    shuffle_and_write()
    dataset_split()