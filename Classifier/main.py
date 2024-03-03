from glob import glob
import torch
from dataset import *
from utils import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from model import *
from config import *
import utils

def train(i):
    model = Classifier().to(config.DEVICE)
    opt_model= optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    average_now = 0

    for epoch in range(config.NUM_EPOCHS):
        print("epoch:",epoch)
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","loss"])
        dataset = OneDataset(root_AV45 = config.syn_AV45,task = config.whole,name = "train",fold = str(i))
        length = dataset.length_dataset
        loader = DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        loss_epoch=0
        
        for idx, (img,label) in enumerate(loop):
            label=label.to(config.DEVICE)
            img = np.expand_dims(img, axis=1)
            img = torch.tensor(img)
            img = img.to(config.DEVICE)
            pred = model(img)

            loss = bce(pred,label)

            model.zero_grad()
            loss.backward() 
            opt_model.step()
            
            loss_epoch=loss_epoch+loss

        writer.writerow([epoch,loss_epoch.item()/length])
        lossfile.close()

        dataset = OneDataset(root_AV45 = config.syn_AV45,task = config.whole,name = "validation",fold = str(i))
        loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.NUM_WORKERS,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)

        result_list = []
        label_list = []
        auc_list = []

        for idx, (img,label) in enumerate(loop):
            label=label.to(config.DEVICE)
            img = np.expand_dims(img, axis=1)
            img = torch.tensor(img)
            img = img.to(config.DEVICE)
            pred = model(img)

            pred = np.squeeze(pred) 
            label = np.squeeze(label)
            pred= nn.Sigmoid()(pred)
            pred = pred.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            auc_list.append(pred)
            if pred > 0.5:
                predict = 1
            else:
                predict = 0
            result_list.append(predict)
            label_list.append(int(label))
        
        average = indicators(label_list,result_list,auc_list,epoch,"validation")
        
        if average > average_now: 
            average_now = average
            save_checkpoint(model, opt_model, filename=config.checkpoint_model)

            dataset = OneDataset(root_AV45 = config.syn_AV45,task = config.whole,name = "test",fold = str(i))
            loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.NUM_WORKERS,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)

            result_list = []
            label_list = []
            auc_list = []

            for idx, (img,label) in enumerate(loop):
                label=label.to(config.DEVICE)
                img = np.expand_dims(img, axis=1)
                img = torch.tensor(img)
                img = img.to(config.DEVICE)
                pred = model(img)

                pred = np.squeeze(pred) 
                label = np.squeeze(label)
                pred= nn.Sigmoid()(pred)
                pred = pred.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                auc_list.append(pred)
                if pred > 0.5:
                    predict = 1
                else:
                    predict = 0
                result_list.append(predict)
                label_list.append(int(label))
            
            indicators(label_list,result_list,auc_list,epoch,"test")

if __name__ == "__main__":
    #utils.seed_torch()
    for i in range(5):
         train(i)