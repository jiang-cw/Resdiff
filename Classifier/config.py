import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 10
LEARNING_RATE = 1e-4

NUM_WORKERS = 0
NUM_EPOCHS = 40
LOAD_MODEL = True
SAVE_MODEL = True
latent_dim = 1 
exp = "exp_11_GANDM_correct_sex/"

checkpoint_model = "result/"+exp+"classifier.pth.tar"

whole_AV45 = "D:/SynDM/data/whole_AV45/"
syn_AV45 = "D:/SynDM/Abeta_classifier/data/TransUNet_2" #"D:/SynDM/Abeta_classifier/data/TransUNet/"

whole = "data_info/whole.txt"