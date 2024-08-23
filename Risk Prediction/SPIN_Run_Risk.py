import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import class_weight

from SPIN_Train_Risk import train_SPIN
from SPIN_Utils import *
import os
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required = True)
parser.add_argument("--num", type = int, help = "The project number", required = True)
parser.add_argument("--data", type = str, help = "Data name", required = True)
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
optim_hparams = [args.opt, args.lr, args.fac, 5, args.wd] ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay
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
    test_auc = train_SPIN(date, num, data, experiment, trainData, trainLabel, valData, valLabel, testData, testLabel,
                          pathway_indices, in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes, dropout_Rates, initializer,
                          activation, opt_lr, opt_wd, lr_factor, lr_patience, n_epoch, 
                          step = step, optimizer = optimizer, learning_rate_scheduler = True)

    print("[%d] Test AUC: %.3f" % (experiment, test_auc))
    record = open(save_path + f"Result/[{date}_{num}]_SPIN_[GSE{data}]_Result.txt", 'a+')
    record.write("[%d] Test AUC: %.3f\r\n" % (experiment, test_auc))
    record.close()
    test_auc_list.append(test_auc)

record = open(save_path + f"Result/[{date}_{num}]_SPIN_[GSE{data}]_Result.txt", 'a+')
record.write("Average of AUC: %.3f\t\tStandard Deviation of AUC: %.4f\r\n" % (np.average(test_auc_list), np.std(test_auc_list)))
record.close()
