import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 1
numworker = 0
epochs = 1000

whole_t1 = "../data/whole_t1"
whole_FDG = "../data/whole_FDG"
whole_AV45 = "../data/whole_AV45"
whole_Tau = "../data/whole_Tau"
path = "../data/whole_t1/002S2010.nii"
# gpus = [0,1]
gpus = [0]

exp = "exp_3/"

CHECKPOINT_model = "result/"+exp+"model.pth.tar"
CHECKPOINT_disc = "result/"+exp+"disc.pth.tar"

train = "./data_info/train.txt"
validation = "./data_info/validation.txt"
test = "./data_info/test.txt"
whole = "./data_info/whole.txt"
whole_for_classifier = "../data_info/whole_Tau.txt"#"../data_info/whole_Abeta.txt"