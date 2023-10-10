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
    pathway_indices = load_sparse_indices(data_path + f"TCGA_{data}_Pathway_Mask.npz")
elif data == "STAD":
    in_Nodes = 4369
    pathway_indices = load_sparse_indices(data_path + f"TCGA_{data}_Pathway_Mask.npz")
elif data == "LUAD":
    in_Nodes = 4365
    pathway_indices = load_sparse_indices(data_path + f"TCGA_{data}_Pathway_Mask.npz")
elif data == "LUSC":
    in_Nodes = 4366
    pathway_indices = load_sparse_indices(data_path + f"TCGA_{data}_Pathway_Mask.npz")
elif data == "GBM_&_LGG":
    in_Nodes = 4350
    pathway_indices = load_sparse_indices(data_path + f"TCGA_{data}_Pathway_Mask.npz")
    
pathway_Nodes = 173
hidden_Nodes = 100
out_Nodes = 1
###################################################################################################################################
### Initialize Settings
initializer = "he_uniform"
activation = "Relu"
dropout_Rates = 0.7
optimizer = "Adam"
opt_lr = '''Set the optimal learning rate'''
opt_wd = '''Set the optimal weight decay'''
lr_factor = '''Set the learning rate scheduler factor'''
lr_patience = '''Set the learning rate scheduler patientce'''
step = 10
n_experiments = 10
n_epoch = 5000
###################################################################################################################################
### Record Settings
record = open(save_path + f"Result/[{date}_{num}]_SPIN_[TCGA_{data}]_Result.txt", 'a+')
record.write("Input Nodes: %d\t\tPathway Nodes: %d\t\tHidden Nodes: %s\t\tOutput Nodes: %d\r\n" % (in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes))
record.write("Initializer: %s\t\tActivation: %s\t\tDropout Rates: %s\t\tOptimizer: %s\r\n" % (initializer, activation, str(dropout_Rates), optimizer))
record.write("Init LR: %s\t\tWeight Decay: %s\r\n" % (opt_lr, opt_wd))
record.write("Epoch: %d\t\tStep Size: %d\r\n" % (eval_epoch, step))
record.write("LR Factor: %s\t\tLR Patience: %s\r\n" % (lr_factor, lr_patience))
record.close()
###################################################################################################################################
### Start SPIN
test_cindex_list = []
for experiment in range(1, n_experiments + 1):
    print("#######################  %d experiment  #######################\n" % experiment)
    ### load train & validation & test data and label
    tcga_tr_x, tcga_tr_yevent, tcga_tr_ytime = load_data(data_path + f"TCGA_{data}_Train_{experiment}.csv")
    tcga_val_x, tcga_val_yevent, tcga_val_ytime = load_data(data_path + f"TCGA_{data}_Valid_{experiment}.csv")
    tcga_ts_x, tcga_ts_yevent, tcga_ts_ytime = load_data(data_path + f"TCGA_{data}_Test_{experiment}.csv")
###################################################################################################################################
    torch.cuda.empty_cache()
    test_cindex = train_SPIN(date, num, data, experiment, tcga_tr_x, tcga_tr_yevent, tcga_tr_ytime,
                          tcga_val_x, tcga_val_yevent, tcga_val_ytime,
                          tcga_ts_x, tcga_ts_yevent, tcga_ts_ytime, pathway_indices,
                          in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes, dropout_Rates, initializer,
                          activation, opt_lr, opt_wd, lr_factor, lr_patience, n_epoch, 
                          step = step, optimizer = optimizer, learning_rate_scheduler = True)

    print("[%d] Test C-Index: %.3f" % (experiment, test_cindex))
    record = open(save_path + f"Result/[{date}_{num}]_SPIN_[TCGA_{data}]_Result.txt", 'a+')
    record.write("[%d] Test C-Index: %.3f\r\n" % (experiment, test_cindex))
    record.close()
    test_cindex_list.append(test_cindex)

record = open(save_path + f"Result/[{date}_{num}]_SPIN_[TCGA_{data}]_Result.txt", 'a+')
record.write("Average of C-Index: %.3f\t\tStandard Deviation of C-Index: %.4f\r\n" % (np.average(test_cindex_list), np.std(test_cindex_list)))
record.close()
