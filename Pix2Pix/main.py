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
import csv
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error as mae
# os.environ['CUDA_VISIBLE_DEVICES'] ='0,1'
import warnings
warnings.filterwarnings("ignore")

def Pix2Pix():
    disc = Discriminator().to(config.device)
    gen = UNet().to(config.device)
    init_net(gen)
    init_net(disc)
    # gen = nn.DataParallel(gen,device_ids=gpus,output_device=gpus[0])
    # disc = nn.DataParallel(disc,device_ids=gpus,output_device=gpus[0])
    opt_disc = optim.Adam(disc.parameters(),lr= config.learning_rate,betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(),lr= config.learning_rate,betas=(0.5, 0.999))

    L1 = nn.L1Loss()
    bce = nn.BCEWithLogitsLoss()
    average = 0
    image = nib.load(config.path)
   
    for epoch in range(config.epochs):
        lossfile = open("result/"+str(config.exp)+"loss_curve.csv", 'a+',newline = '')
        writer = csv.writer(lossfile)
        if epoch == 0:
            writer.writerow(["Epoch","G_loss", "D_loss"])
        
        dataset = OneDataset(data_path = config.train, target_path = config.train_target, name = "train")
        loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
        loop = tqdm(loader, leave=True)
        length = dataset.__len__()
        G_loss_epoch = 0
        D_loss_epoch = 0

        for idx, (t1, PET, data_name) in enumerate(loop):
            t1 = np.expand_dims(t1, axis=1)
            t1 = torch.tensor(t1)
            t1 = t1.to(config.device)

            PET = np.expand_dims(PET, axis=1)
            PET = torch.tensor(PET)
            PET = PET.to(config.device)

            fake_PET = gen(t1)
            #fake
            set_requires_grad(disc, True)
            opt_disc.zero_grad()
            fake_AB = torch.cat((t1, fake_PET), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = disc(fake_AB.detach())

            loss_D_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            # Real
            real_AB = torch.cat((t1, PET), 1)
            pred_real = disc(real_AB)
            loss_D_real = bce(pred_real, torch.ones_like(pred_real))
            # combine loss and calculate gradients
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_disc.step() 

            set_requires_grad(disc, False)
            opt_gen.zero_grad()
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((t1, fake_PET), 1)
            pred_fake = disc(fake_AB)
            loss_G_GAN = bce(pred_fake, torch.ones_like(pred_fake))
            # Second, G(A) = B
            loss_G_L1 = L1(fake_PET, PET) 
            # combine loss and calculate gradients
            loss_G = loss_G_GAN + loss_G_L1 * 100  
            loss_G.backward()
            opt_gen.step()

            G_loss_epoch=G_loss_epoch+loss_G
            D_loss_epoch=D_loss_epoch+loss_D

        writer.writerow([epoch,G_loss_epoch.item()/length, D_loss_epoch.item()/length])
        lossfile.close()
        if epoch % 5 == 0:
            dataset = OneDataset(data_path = config.test, target_path = config.test_target, name = "test")
            loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=config.numworker,pin_memory=True,drop_last=True)
            loop = tqdm(loader, leave=True)
            length = dataset.__len__()

            mae_0_target = 0
            psnr_0_target = 0
            ssim_0_target = 0

            for idx, (t1, PET, data_name) in enumerate(loop):
                indicatorfile = open("result/"+str(config.exp)+"test.csv", 'a+', newline = '')
                writer = csv.writer(indicatorfile)
                if epoch == 0 and idx == 0:
                    writer.writerow(["Epoch","MAE","PSNR","SSIM"])
                with torch.no_grad():

                    t1 = np.expand_dims(t1, axis=1)
                    t1 = torch.tensor(t1)
                    t1 = t1.to(config.device)

                    PET = np.expand_dims(PET, axis=1)
                    PET = torch.tensor(PET)
                    PET = PET.to(config.device)

                    fake_PET = gen(t1)
                    fake_PET = torch.clamp(fake_PET,0,1)
                    fake_PET=fake_PET.detach().cpu().numpy()
                    fake_PET=np.squeeze(fake_PET)
                    fake_PET=fake_PET.astype(np.float32)

                    True_PET=PET.detach().cpu().numpy()
                    True_PET=np.squeeze(True_PET)
                    True_PET=True_PET.astype(np.float32)

                    fake_PET_flatten = fake_PET.reshape(-1,128)
                    True_PET_flatten = True_PET.reshape(-1,128)
                    mae_0_target += mae(True_PET_flatten,fake_PET_flatten)
                    psnr_0_target += round(psnr(True_PET,fake_PET),3)
                    ssim_0_target += round(ssim(True_PET,fake_PET),3)

                    if (idx == 0):
                        syn_PET=nib.Nifti1Image(fake_PET,image.affine)
                        nib.save(syn_PET,"result/"+str(config.exp)+"epoch_"+str(epoch))

            writer.writerow([epoch,mae_0_target/length, psnr_0_target/length, ssim_0_target/length])
            indicatorfile.close()
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_model)

def test():
    image = nib.load(config.path)
    gen = UNet().to(config.device)
    opt_gen = optim.Adam(gen.parameters(),lr= config.learning_rate,betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_model, gen, opt_gen, config.learning_rate)

    dataset = OneDataset(root_t1 = config.whole_t1, task = config.test, name = "test")
    loader = DataLoader(dataset,batch_size=config.batch_size,shuffle=True,num_workers=config.numworker,pin_memory=True,drop_last=True)
    loop = tqdm(loader, leave=True)
    
    gen.eval()

    for idx, (t1,t1_name) in enumerate(loop):
        FDG_test_file = open("result/"+str(config.exp)+"FDG_test.csv", 'a+', newline = '')
        FDG_writer = csv.writer(FDG_test_file)
        if idx == 0:
            FDG_writer.writerow(["Name","Name","PSNR","SSIM"])

        t1 = np.expand_dims(t1, axis=1)
        t1 = torch.tensor(t1)
        t1 = t1.to(config.device)

        FDG_path = config.whole_AV45+"/"+t1_name[0]
        if os.path.exists(FDG_path):
            FDG = nifti_to_numpy(FDG_path)
            FDG_out = gen(t1)
            FDG_out = torch.clamp(FDG_out,0,1)
            FDG_out = FDG_out.detach().cpu().numpy()
            FDG_out = np.squeeze(FDG_out)
            FDG_out = FDG_out.astype(np.float32)

            fake_PET_flatten = FDG_out.reshape(-1,128)
            True_PET_flatten = FDG.reshape(-1,128)
            mae_0_FDG = mae(True_PET_flatten,fake_PET_flatten)
            psnr_0_FDG = round(psnr(FDG,FDG_out),3)
            ssim_0_FDG = round(ssim(FDG,FDG_out),3)

            FDG_writer.writerow([str(t1_name[0]),mae_0_FDG,psnr_0_FDG,ssim_0_FDG])
            FDG_out=nib.Nifti1Image(FDG_out,image.affine)
            nib.save(FDG_out,"data/syn/AV45/"+str(t1_name[0]))

    FDG_test_file.close()

if __name__ == "__main__":
    Pix2Pix()
    test()

