import os
import copy
import config
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import csv
from torch import optim
from utils import *
from model import *
from dataset import *
from torch.cuda.amp import autocast, GradScaler

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import warnings
warnings.filterwarnings("ignore")
from itertools import islice

def train():
    gpus = config.gpus
    image = nib.load(config.path)
    mse = nn.MSELoss()
    diffusion = Diffusion()
    average_now = 0

    # net = Net()#.to(config.device)
    Unet = UNet().to(config.device)
    opt_Unet= optim.AdamW(list(Unet.parameters()), lr=config.learning_rate)
    # ema = EMA(0.9999)
    # ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","Loss"])
        
        dataset = OneDataset(data_path = config.train, target_path = config.train_target, name = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.__len__()
        MSE_loss_epoch = 0

        for idx, (data, target, data_name) in enumerate(loop):
            opt_Unet.zero_grad()
            data = np.expand_dims(data, axis=1)
            data = torch.tensor(data)
            data = data.to(config.device)

            target = np.expand_dims(target, axis=1)
            target = torch.tensor(target)
            target = target.to(config.device)

            t = diffusion.sample_timesteps(data.shape[0]).to(config.device)
            target_t = diffusion.noise_images(target, data, t)
            target_out = Unet(target_t, t)

            MSE_loss = mse(target_out, target)
            MSE_loss.backward()
            # with open("model_gradients.txt", "w") as file:
            #     for name, param in Unet.named_parameters():
            #         if param.grad is not None:
            #             file.write(f"{name}: {param.grad}\n")
            opt_Unet.step()


            MSE_loss_epoch=MSE_loss_epoch+MSE_loss.item()

        writer.writerow([epoch+1,MSE_loss_epoch/length])
        lossfile.close()

        if epoch % 5 == 0:
            dataset = OneDataset(data_path = config.test, target_path = config.test_target, name = "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.__len__()

            mae_0_target = 0
            psnr_0_target = 0
            ssim_0_target = 0

            for idx, (data, target, data_name) in enumerate(loop):
                validation_file = open("result/"+str(config.exp)+"test.csv", 'a+', newline = '')
                writer = csv.writer(validation_file)
                if epoch == 0 and idx == 0:
                    writer.writerow(["Epoch","MAE","PSNR","SSIM"])
                
                with torch.no_grad():

                    data = np.expand_dims(data, axis=1)
                    data = torch.tensor(data)
                    data = data.to(config.device)

                    target = np.expand_dims(target, axis=1)
                    target = torch.tensor(target)
                    target = target.to(config.device)

                    t = diffusion.sample_timesteps(data.shape[0]).to(config.device)
                    target_t = diffusion.noise_images(target, data, t)
                    target_out = Unet(target_t, t)
                    target_out = torch.clamp(target_out, 0, 1)

                    target = target.detach().cpu().numpy().astype(np.float32)
                    target_out = target_out.detach().cpu().numpy().astype(np.float32)
                    target = np.squeeze(target)
                    target_out = np.squeeze(target_out)

                    fake_flatten = target_out.reshape(-1,128)
                    True_flatten = target.reshape(-1,128)
                    mae_0_target += mae(True_flatten,fake_flatten)
                    psnr_0_target += round(psnr(target,target_out),3)
                    ssim_0_target += round(ssim(target,target_out, data_range=1),3)

            writer.writerow([epoch,mae_0_target/length,psnr_0_target/length,ssim_0_target/length])
            validation_file.close()

            average = psnr_0_target/length + ssim_0_target/length * 10
            if average > average_now:
                average_now = average
                target_out = nib.Nifti1Image(target_out,image.affine)
                nib.save(target_out,"result/"+str(config.exp)+"target")
                save_checkpoint(Unet,opt_Unet,filename=config.CHECKPOINT_Unet)
                # save_checkpoint(net,opt_Unet,filename=config.CHECKPOINT_net)
               
if __name__ == '__main__':
    #utils.seed_torch()
    train()
