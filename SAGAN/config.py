import torch

device = torch.device("cuda:0")
#device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 1e-4
batch_size = 1
numworker = 0
epochs = 200

train = "../data/train/input/"
train_target = "../data/train/target/"
test = "../data/test/input/"
test_target = "../data/test/target/"
path = "../data/train/input/686130_0.nii"

# train = "D:/Caiwen/data/train/input/"
# train_target = "D:/Caiwen/data/train/target/"
# test = "D:/Caiwen/data/test/input/"
# test_target = "D:/Caiwen/data/test/target/"
# path = "D:/Caiwen/data/train/input/686130_0.nii"

# gpus = [0,1]
gpus = [0]
exp = "exp_1/"

CHECKPOINT_model = "result/"+exp+"Unet.pth.tar"
CHECKPOINT_net = "result/"+exp+"net.pth.tar"