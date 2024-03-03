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

hidden_size = 1024
mlp_dim = 1024
num_heads = 16
dropout_rate = 0.1
attention_dropout_rate = 0.0
num_layers = 24
grid = (16, 16, 8)
lambda_adv = 1.0
lambda_A = 10.0
patches_size = (8, 8, 8)
resnet_num_layers = (3, 4, 9)
resnet_width_factor = 1
transformer_num_heads = 16
attention_dropout_rate = 0.0
transformer_dropout_rate = 0.1
decoder_channels = (256, 128, 64, 16)
skip_channels = [512, 256, 64, 16]
n_skip = 3