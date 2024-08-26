import torch
import numpy as np
import pandas as pd
from datetime import datetime

from SPIN_Train_Survival import train_SPIN
from DataLoader import *
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required = True)
parser.add_argument("--num", type = int, help = "The project number", required = True)
parser.add_argument("--data", type = str, help = "Data name", required = True)
parser.add_argument("--epoch", type = int, help = "Epoch Size", required=True)
parser.add_argument("--btch", type = int, help = "Batch Size", required=True)
parser.add_argument("--init", type = str, help = "Initializer", default="he_normal")
parser.add_argument("--act", type = str, help = "Activation", default="relu")
parser.add_argument("--opt", type = str, help = "Optimizer", default="Adam")
parser.add_argument("--dr", type = float, help = "Dropout", default=0.)
parser.add_argument("--lr", type = float, help = "Learning rate", default=1e-3)
parser.add_argument("--fac", type = float, help = "Learning rate factor", default=0.99)
parser.add_argument("--pat", type = float, help = "Learning rate patience", default=0.99)
parser.add_argument("--wd", type = float, help = "Weight Decay", default=1e-2)
args = parser.parse_args()

### GPU assign
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

date = datetime.today().strftime('%m%d')
num = args.num
data = args.data
###################################################################################################################################
### Set Load & Save path
data_path = '''Set the path to load the datasets'''
save_path = '''Set the path to save files & results'''
###################################################################################################################################
### Load Pathway Mask
### Net setting
if data == "LIHC":
    in_Nodes = 4360
    pathway_idx = load_sparse_indices("Load Pathway Mask")
elif data == "STAD":
    in_Nodes = 4369
    pathway_idx = load_sparse_indices("Load Pathway Mask")
elif data == "LUAD":
    in_Nodes = 4365
    pathway_idx = load_sparse_indices("Load Pathway Mask")
elif data == "LUSC":
    in_Nodes = 4366
    pathway_idx = load_sparse_indices("Load Pathway Mask")
elif data == "GBM_&_LGG":
    in_Nodes = 4350
    pathway_idx = load_sparse_indices("Load Pathway Mask")
    
pathway_Nodes = 173
hidden_Nodes = 100
out_Nodes = 1
n_experiments = 10
###################################################################################################################################
### Optimal Hyperparams Settings
### Obtained from HyperParams_Optimization.py
net_hparams = [in_Nodes, [pathway_Nodes, hidden_Nodes], out_Nodes, args.init, args.act, args.dr] ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-dropout
optim_hparams = [args.opt, args.lr, args.fac, args.pat, args.wd] ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay
experim_hparms = [args.epoch, args.btch] ### 0-max_epoch, 1-batch_size
###################################################################################################################################
### Start SPIN
test_cindex_list = []
for experiment in range(1, n_experiments + 1):
    trainData, trainEvent, trainTime = load_data("Load Train Dataset")
    validData, validEvent, validTime = load_data("Load Valid Dataset")
    testData, testEvent, testTime = load_data("Load Test Dataset")
    ###################################################################################################################################
    torch.cuda.empty_cache()
    test_cindex = train_SPIN(date, num, data, experiment, trainData, trainEvent, trainTime, validData, validEvent, validTime, testData, testEvent, testTime, pathway_idx, net_hparams, optim_hparams, experim_hparms)
    ###################################################################################################################################
    print("[%d] Test C-Index: %.3f" % (experiment, test_cindex))
    record = open("Save Test C-Index", 'a+')
    record.write("[%d] Test C-Index: %.3f\r\n" % (experiment, test_cindex))
    record.close()
    test_cindex_list.append(test_cindex)

record = open("Save Test C-Index", 'a+')
record.write("Average of C-Index: %.3f\t\tStandard Deviation of C-Index: %.4f\r\n" % (np.average(test_cindex_list), np.std(test_cindex_list)))
record.close()
