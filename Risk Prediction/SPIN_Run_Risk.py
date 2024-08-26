import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import class_weight

from SPIN_Train_Risk import train_SPIN
from SPIN_Utils import *
import argparse
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required = True)
parser.add_argument("--num", type = int, help = "The project number", required = True)
parser.add_argument("--data", type = str, help = "Data name", required = True)
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
if data == "8052":
    in_Nodes = 4394
    pathway_idx = load_sparse_indices("Load Pathway Mask")
elif data == "172367":
    in_Nodes = 3019
    pathway_idx = load_sparse_indices("Load Pathway Mask")    
pathway_Nodes = 173
hidden_Nodes = 100
out_Nodes = 1
###################################################################################################################################
### Optimal Hyperparams Settings
### Obtained from HyperParams_Optimization.py
net_hparams = [in_Nodes, [pathway_Nodes, hidden_Nodes], out_Nodes, args.init, args.act, args.dr] ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-dropout
optim_hparams = [args.opt, args.lr, args.fac, args.pat, args.wd] ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay
experim_hparms = [100, args.btch] ### 0-max_epoch, 1-batch_size
###################################################################################################################################
### Start SPIN
test_auc_list = []
for experiment in range(1, n_experiments + 1):
    print("#######################  %d experiment  #######################\n" % experiment)
    ### load train & validation & test data and label
    trainData = pd.read_csv("Load Train Data")
    trainLabel = pd.read_csv("Load Train Label")
    validData = pd.read_csv("Load Valid Data")
    validLabel = pd.read_csv("Load Valid Label")
    testData = pd.read_csv("Load Test Data")
    testLabel = pd.read_csv("Load Test Label")
    ###################################################################################################################################
    torch.cuda.empty_cache()
    test_auc = train_SPIN(date, num, data, experiment, trainData, trainLabel, validData, validLabel, testData, testLabel, pathway_idx, net_hparams, optim_hparams, experim_hparms)

    print("[%d] Test AUC: %.3f" % (experiment, test_auc))
    record = open("Save Test AUC", 'a+')
    record.write("[%d] Test AUC: %.3f\r\n" % (experiment, test_auc))
    record.close()
    test_auc_list.append(test_auc)

record = open("Save Test AUC", 'a+')
record.write("Average of AUC: %.3f\t\tStandard Deviation of AUC: %.4f\r\n" % (np.average(test_auc_list), np.std(test_auc_list)))
record.close()
