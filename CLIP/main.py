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
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import parameters_to_vector
from utils import *
from model import *
from dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
import warnings
warnings.filterwarnings("ignore")
from itertools import islice

def cosine_similarity(image1, image2):
    tensor1 = torch.flatten(image1)
    tensor2 = torch.flatten(image2)

    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    similarity = dot_product / (norm1 * norm2)
    return similarity

def train():
    gpus = config.gpus
    image = nib.load(config.path)
    mse = nn.MSELoss()
    diffusion = Diffusion()
    average_now = 0

    net = Net().to(config.device)
    Unet = UNet().to(config.device)
    opt_Unet= optim.AdamW(list(Unet.parameters())+list(net.parameters()), lr=config.learning_rate)
    ema = EMA(0.9999)
    ema_Unet = copy.deepcopy(Unet).eval().requires_grad_(False)
    data = pd.read_csv("../data_info/info.csv",encoding = "ISO-8859-1")

    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","G_loss", "D_loss"])
        
        dataset = OneDataset(root_t1 = config.whole_t1, task = config.train, name = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.__len__()
        G_loss_epoch = 0

        for idx, (t1, t1_name) in enumerate(loop):
            AV45_path = config.whole_AV45+"/"+t1_name[0]
            Tau_path = config.whole_Tau+"/"+t1_name[0]
            if os.path.exists(AV45_path) or os.path.exists(Tau_path):
                t1 = np.expand_dims(t1, axis=1)
                t1 = torch.tensor(t1)
                t1 = t1.to(config.device)

                Age = data[data['ID'] == t1_name[0][0:-4]]['Age']
                Age = Age.values
                Age = Age.astype(np.float32)
                Sex = data[data['ID'] == t1_name[0][0:-4]]['Sex']
                Sex = Sex.values
                MMSE = data[data['ID'] == t1_name[0][0:-4]]['MMSE']
                MMSE = MMSE.values
                MMSE = MMSE.astype(np.float32)
                Disease = data[data['ID'] == t1_name[0][0:-4]]['Disease']
                Disease = Disease.values

                t = diffusion.sample_timesteps(t1.shape[0]).to(config.device)

                if os.path.exists(AV45_path):
                    AV45_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                    AV45_text_feature = net(clip.tokenize(AV45_text).to(config.device))
                    AV45 = nifti_to_numpy(AV45_path)
                    AV45 = np.expand_dims(AV45, axis=0)
                    AV45 = np.expand_dims(AV45, axis=1)
                    AV45 = torch.tensor(AV45)
                    AV45 = AV45.to(config.device)
                    AV45_t = diffusion.noise_images(AV45, t1, t)
                    AV45_out = Unet(AV45_t, t, AV45_text_feature)
                else:
                    AV45_out = None

                if os.path.exists(Tau_path):
                    Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                    Tau_text_feature = net(clip.tokenize(Tau_text).to(config.device))
                    Tau = nifti_to_numpy(Tau_path)
                    Tau = np.expand_dims(Tau, axis=0)
                    Tau = np.expand_dims(Tau, axis=1)
                    Tau = torch.tensor(Tau)
                    Tau = Tau.to(config.device)
                    Tau_t = diffusion.noise_images(Tau, t1, t)
                    Tau_out = Unet(Tau_t, t, Tau_text_feature)
                else:
                    Tau_out = None

                MSE_loss = 0
                opt_Unet.zero_grad()

                if AV45_out is not None:
                    MSE_loss += mse(AV45_out, AV45)

                if Tau_out is not None:
                    MSE_loss += mse(Tau_out, Tau) 

                high_loss = 0
                if (AV45_out is not None) and (Tau_out is not None):
                    high_loss += mse(torch.abs(AV45-Tau),torch.abs(AV45_out-Tau_out))

                loss_G = MSE_loss * 10 + high_loss
                loss_G.backward()
                opt_Unet.step()
                ema.step_ema(ema_Unet, Unet)

                G_loss_epoch=G_loss_epoch+loss_G

        # print(MSE_loss)
        # print(high_loss)
        writer.writerow([epoch+1,G_loss_epoch.item()/length]) #, D_loss_epoch.item()/length
        lossfile.close()

        #validation
        if epoch % 5 == 0:
            dataset = OneDataset(root_t1 = config.whole_t1, task = config.validation, name = "test")
            loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.__len__()

            mae_0_AV45 = 0
            psnr_0_AV45 = 0
            ssim_0_AV45 = 0
            mae_0_Tau = 0
            psnr_0_Tau = 0
            ssim_0_Tau = 0
            count_AV45 = 0
            count_Tau = 0

            for idx, (t1,t1_name) in enumerate(loop):
                if idx < 20:
                    validation_file = open("result/"+str(config.exp)+"validation.csv", 'a+', newline = '')
                    writer = csv.writer(validation_file)
                    if epoch == 0 and idx == 0:
                        writer.writerow(["Epoch","Name","PSNR","SSIM"])

                    t1 = np.expand_dims(t1, axis=1)
                    t1 = torch.tensor(t1)
                    t1 = t1.to(config.device)

                    Age = data[data['ID'] == t1_name[0][0:-4]]['Age']
                    Age = Age.values
                    Age = Age.astype(np.float32)
                    Sex = data[data['ID'] == t1_name[0][0:-4]]['Sex']
                    Sex = Sex.values
                    MMSE = data[data['ID'] == t1_name[0][0:-4]]['MMSE']
                    MMSE = MMSE.values
                    MMSE = MMSE.astype(np.float32)
                    Disease = data[data['ID'] == t1_name[0][0:-4]]['Disease']
                    Disease = Disease.values

                    t = diffusion.sample_timesteps(t1.shape[0]).to(config.device)

                    AV45_path = config.whole_AV45+"/"+t1_name[0]
                    if os.path.exists(AV45_path):
                        AV45_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                        AV45_text_feature = net(clip.tokenize(AV45_text).to(config.device))
                        count_AV45 += 1
                        AV45 = nifti_to_numpy(AV45_path)
                        AV45_out = diffusion.sample(ema_Unet, t1, AV45_text_feature)
                        AV45_out = torch.clamp(AV45_out,0,1)
                        AV45_out = AV45_out.detach().cpu().numpy()
                        AV45_out = np.squeeze(AV45_out)
                        AV45_out = AV45_out.astype(np.float32)

                        fake_PET_flatten = AV45_out.reshape(-1,128)
                        True_PET_flatten = AV45.reshape(-1,128)
                        mae_0_AV45 += mae(True_PET_flatten,fake_PET_flatten)
                        psnr_0_AV45 += round(psnr(AV45,AV45_out),3)
                        ssim_0_AV45 += round(ssim(AV45,AV45_out),3)

                    Tau_path = config.whole_Tau+"/"+t1_name[0]
                    if os.path.exists(Tau_path):
                        Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                        Tau_text_feature = net(clip.tokenize(Tau_text).to(config.device))
                        count_Tau += 1
                        Tau = nifti_to_numpy(Tau_path)
                        Tau_out = diffusion.sample(ema_Unet, t1,Tau_text_feature)
                        Tau_out = torch.clamp(Tau_out,0,1)
                        Tau_out = Tau_out.detach().cpu().numpy()
                        Tau_out = np.squeeze(Tau_out)
                        Tau_out = Tau_out.astype(np.float32)

                        fake_PET_flatten = Tau_out.reshape(-1,128)
                        True_PET_flatten = Tau.reshape(-1,128)
                        mae_0_Tau += mae(True_PET_flatten,fake_PET_flatten)
                        psnr_0_Tau += round(psnr(Tau,Tau_out),3)
                        ssim_0_Tau += round(ssim(Tau,Tau_out),3)

            writer.writerow([epoch,mae_0_AV45/count_AV45,psnr_0_AV45/count_AV45,ssim_0_AV45/count_AV45])
            writer.writerow([epoch,mae_0_Tau/count_Tau,psnr_0_Tau/count_Tau,ssim_0_Tau/count_Tau])
            validation_file.close()

            #test
            average = psnr_0_AV45/count_AV45 + ssim_0_AV45/count_AV45 * 10 + psnr_0_Tau/count_Tau + ssim_0_Tau/count_Tau * 10
            if average > average_now:
                average_now = average
                dataset = OneDataset(root_t1 = config.whole_t1, task = config.test, name = "test")
                loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
                loop = tqdm(loader, leave=True)
                length = dataset.__len__()

                mae_0_AV45 = 0
                psnr_0_AV45 = 0
                ssim_0_AV45 = 0
                mae_0_Tau = 0
                psnr_0_Tau = 0
                ssim_0_Tau = 0
                count_AV45 = 0
                count_Tau = 0

                for idx, (t1,t1_name) in enumerate(loop):
                    if idx < 20:
                        test_file = open("result/"+str(config.exp)+"test.csv", 'a+', newline = '')
                        writer = csv.writer(test_file)
                        if epoch == 0 and idx == 0:
                            writer.writerow(["Epoch","Name","PSNR","SSIM"])

                        t1 = np.expand_dims(t1, axis=1)
                        t1 = torch.tensor(t1)
                        t1 = t1.to(config.device)

                        Age = data[data['ID'] == t1_name[0][0:-4]]['Age']
                        Age = Age.values
                        Age = Age.astype(np.float32)
                        Sex = data[data['ID'] == t1_name[0][0:-4]]['Sex']
                        Sex = Sex.values
                        MMSE = data[data['ID'] == t1_name[0][0:-4]]['MMSE']
                        MMSE = MMSE.values
                        MMSE = MMSE.astype(np.float32)
                        Disease = data[data['ID'] == t1_name[0][0:-4]]['Disease']
                        Disease = Disease.values

                        t = diffusion.sample_timesteps(t1.shape[0]).to(config.device)

                        AV45_path = config.whole_AV45+"/"+t1_name[0]
                        if os.path.exists(AV45_path):
                            AV45_text = f"Synthesize an Aβ-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                            AV45_text_feature = net(clip.tokenize(AV45_text).to(config.device))
                            count_AV45 += 1
                            AV45 = nifti_to_numpy(AV45_path)
                            AV45_out = diffusion.sample(ema_Unet, t1, AV45_text_feature)
                            AV45_out = torch.clamp(AV45_out,0,1)
                            AV45_out = AV45_out.detach().cpu().numpy()
                            AV45_out = np.squeeze(AV45_out)
                            AV45_out = AV45_out.astype(np.float32)

                            fake_PET_flatten = AV45_out.reshape(-1,128)
                            True_PET_flatten = AV45.reshape(-1,128)
                            mae_0_AV45 += mae(True_PET_flatten,fake_PET_flatten)
                            psnr_0_AV45 += round(psnr(AV45,AV45_out),3)
                            ssim_0_AV45 += round(ssim(AV45,AV45_out),3)

                        Tau_path = config.whole_Tau+"/"+t1_name[0]
                        if os.path.exists(Tau_path):
                            Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
                            Tau_text_feature = net(clip.tokenize(Tau_text).to(config.device))
                            count_Tau += 1
                            Tau = nifti_to_numpy(Tau_path)
                            Tau_out = diffusion.sample(ema_Unet, t1, Tau_text_feature)
                            Tau_out = torch.clamp(Tau_out,0,1)
                            Tau_out = Tau_out.detach().cpu().numpy()
                            Tau_out = np.squeeze(Tau_out)
                            Tau_out = Tau_out.astype(np.float32)

                            fake_PET_flatten = Tau_out.reshape(-1,128)
                            True_PET_flatten = Tau.reshape(-1,128)
                            mae_0_Tau += mae(True_PET_flatten,fake_PET_flatten)
                            psnr_0_Tau += round(psnr(Tau,Tau_out),3)
                            ssim_0_Tau += round(ssim(Tau,Tau_out),3)
                    
                writer.writerow([epoch,mae_0_AV45/count_AV45,psnr_0_AV45/count_AV45,ssim_0_AV45/count_AV45])
                AV45_out=nib.Nifti1Image(AV45_out,image.affine)
                nib.save(AV45_out,"result/"+str(config.exp)+str(epoch)+"_AV45_"+str(t1_name[0]))

                writer.writerow([epoch,mae_0_Tau/count_Tau,psnr_0_Tau/count_Tau,ssim_0_Tau/count_Tau])
                Tau_out=nib.Nifti1Image(Tau_out,image.affine)
                nib.save(Tau_out,"result/"+str(config.exp)+str(epoch)+"_Tau_"+str(t1_name[0]))
                
                test_file.close()
                save_checkpoint(ema_Unet,opt_Unet,filename=config.CHECKPOINT_model)

def test():
    image = nib.load(config.path)
    diffusion = Diffusion()
    Unet = UNet().to(config.device)
    opt_Unet= optim.AdamW(Unet.parameters(), lr=config.learning_rate)
    load_checkpoint(config.CHECKPOINT_model, Unet, opt_Unet, config.learning_rate)

    dataset = OneDataset(root_t1 = config.whole_t1, task = config.whole_for_classifier, name = "test")
    loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
    loop = tqdm(loader, leave=True)
    data = pd.read_csv("../data_info/info.csv",encoding = "ISO-8859-1")

    for idx, (t1,t1_name) in enumerate(loop):

        Age = data[data['ID'] == t1_name[0][0:-4]]['Age']
        Age = Age.values
        Age = Age.astype(np.float32)
        Sex = data[data['ID'] == t1_name[0][0:-4]]['Sex']
        Sex = Sex.values
        MMSE = data[data['ID'] == t1_name[0][0:-4]]['MMSE']
        MMSE = MMSE.values
        MMSE = MMSE.astype(np.float32)
        Disease = data[data['ID'] == t1_name[0][0:-4]]['Disease']
        Disease = Disease.values

        AV45_test_file = open("result/"+str(config.exp)+"AV45_test.csv", 'a+', newline = '')
        AV45_writer = csv.writer(AV45_test_file)
        if idx == 0:
            AV45_writer.writerow(["Name","Name","PSNR","SSIM"])
        
        Tau_test_file = open("result/"+str(config.exp)+"Tau_test.csv", 'a+', newline = '')
        Tau_writer = csv.writer(Tau_test_file)
        if idx == 0:
            Tau_writer.writerow(["Name","Name","PSNR","SSIM"])

        t1 = np.expand_dims(t1, axis=1)
        t1 = torch.tensor(t1)
        t1 = t1.to(config.device)

        # AV45_path = config.whole_AV45+"/"+t1_name[0]
        # if os.path.exists(AV45_path):
        #     AV45 = nifti_to_numpy(AV45_path)
        #     AV45_out = diffusion.sample(Unet, t1, torch.ones(1, dtype=torch.long).to(config.device))
        #     AV45_out = torch.clamp(AV45_out,0,1)
        #     AV45_out = AV45_out.detach().cpu().numpy()
        #     AV45_out = np.squeeze(AV45_out)
        #     AV45_out = AV45_out.astype(np.float32)

        #     fake_PET_flatten = AV45_out.reshape(-1,128)
        #     True_PET_flatten = AV45.reshape(-1,128)
        #     mae_0_AV45 = mae(True_PET_flatten,fake_PET_flatten)
        #     psnr_0_AV45 = round(psnr(AV45,AV45_out),3)
        #     ssim_0_AV45 = round(ssim(AV45,AV45_out),3)

        #     AV45_writer.writerow([str(t1_name[0]),mae_0_AV45,psnr_0_AV45,ssim_0_AV45])
        #     AV45_out=nib.Nifti1Image(AV45_out,image.affine)
        #     nib.save(AV45_out,"data/GANDM/"+str(t1_name[0]))

        Tau_path = config.whole_Tau+"/"+t1_name[0]
        if os.path.exists(Tau_path):
            Tau_text = f"Synthesize an Tau-PET scan for a {Age[0]}-year-old {Sex[0]} subject" # diagnoised as {Disease[0]} with Mini-Mental Status Examination of {MMSE[0]}
            Tau_text_feature = net(clip.tokenize(Tau_text).to(config.device))
            Tau = nifti_to_numpy(Tau_path)
            Tau_out = diffusion.sample(Unet, t1, Tau_text_feature)
            Tau_out = torch.clamp(Tau_out,0,1)
            Tau_out = Tau_out.detach().cpu().numpy()
            Tau_out = np.squeeze(Tau_out)
            Tau_out = Tau_out.astype(np.float32)

            fake_PET_flatten = Tau_out.reshape(-1,128)
            True_PET_flatten = Tau.reshape(-1,128)
            mae_0_Tau = mae(True_PET_flatten,fake_PET_flatten)
            psnr_0_Tau = round(psnr(Tau,Tau_out),3)
            ssim_0_Tau = round(ssim(Tau,Tau_out),3)

            Tau_writer.writerow([str(t1_name[0]),mae_0_Tau,psnr_0_Tau,ssim_0_Tau])
            Tau_out=nib.Nifti1Image(Tau_out,image.affine)
            nib.save(Tau_out,"data/GANDM_Tau/"+str(t1_name[0]))

    # FDG_test_file.close()
    AV45_test_file.close()
    Tau_test_file.close()


if __name__ == '__main__':
    #utils.seed_torch()
    train()
    test()
