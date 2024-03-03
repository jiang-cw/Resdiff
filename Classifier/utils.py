import torch
import torch.nn as nn
import config
import nibabel as nib
import numpy as np
import csv
import random
import os
from sklearn.metrics import accuracy_score, f1_score,recall_score, precision_score, roc_auc_score, confusion_matrix

def specificity(Labels, Predictions):
    matrix = confusion_matrix(Labels, Predictions)
    tn_sum = matrix[0, 0]
    fp_sum = matrix[0, 1]
    Condition_negative = tn_sum + fp_sum + 1e-6
    Specificity = tn_sum / Condition_negative
    return Specificity

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def initial_model(model):
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight.detach())
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.detach())
            m.bias.detach().zero_()

def indicators(label,predict,auc_list,epoch,name="nonspecific"):
    csvfile = open("result/"+str(config.exp)+str(name)+".csv", 'a+',newline = '')
    writer = csv.writer(csvfile)
    if epoch == 0:
        writer.writerow(["Epoch","accuracy","precision","recall","F1_score","specificity_score","Roc_auc_score"])
    accuracy=round(accuracy_score(label,predict),3)
    precision=round(precision_score(label,predict,average="binary", pos_label=1),3)
    recall=round(recall_score(label,predict,average="binary", pos_label=1),3)
    F1_score=round(f1_score(label,predict,average="binary", pos_label=1),3)
    specificity_score=round(specificity(label,predict),3)
    Roc_auc_score=round(roc_auc_score(label,auc_list),3)
    average = (accuracy + recall + specificity_score +F1_score+Roc_auc_score)/5
    writer.writerow([epoch+1,accuracy,precision,recall,F1_score,specificity_score,Roc_auc_score])
    csvfile.close()
    return average

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
