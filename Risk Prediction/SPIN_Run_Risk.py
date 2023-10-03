import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.utils import class_weight

from SPIN_Train_Risk import train_SPIN
from SPIN_Utils_Risk import *
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
data_path = f"../../../nasdatafolder/MTL/Data/GSE_{data}_KEGG_Genes/"
save_path = f"../../../nasdatafolder/MTL/SPIN/GSE_{data}_Result/"
###################################################################################################################################
### Load Pathway Mask
### Net setting
if data == "8052":
    in_Nodes = 4394
    pathway_indices = load_sparse_indices(data_path + f"Asthma_GSE_{data}_Gene_KEGG_Mask.npz")
elif data == "172367":
    in_Nodes = 3019
    pathway_indices = load_sparse_indices(data_path + f"Asthma_GSE_{data}_Gene_KEGG_Mask.npz")
    
pathway_Nodes = 173
hidden_Nodes = 100
out_Nodes = 1
###################################################################################################################################
### Initialize Settings
initializer = "he_uniform"
activation = "Relu"
dropout_Rates = 0.7
optimizer = "Adam"
opt_lr = 7e-5
opt_wd = 9.9e-1
lr_factor = 0.99
lr_patience = 1
step = 10
n_experiments = 10
n_epoch = 5000
###################################################################################################################################
### Record Settings
record = open(save_path + f"Result/[{date}_{num}]_SPIN_[GSE{data}]_Result.txt", 'a+')
record.write("Input Nodes: %d\t\tPathway Nodes: %d\t\tHidden Nodes: %s\t\tOutput Nodes: %d\r\n" % (in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes))
record.write("Initializer: %s\t\tActivation: %s\t\tDropout Rates: %s\t\tOptimizer: %s\r\n" % (initializer, activation, str(dropout_Rates), optimizer))
record.write("Init LR: %s\t\tWeight Decay: %s\r\n" % (opt_lr, opt_wd))
record.write("Epoch: %d\t\tStep Size: %d\r\n" % (eval_epoch, step))
record.write("LR Factor: %s\t\tLR Patience: %s\r\n" % (lr_factor, lr_patience))
record.close()
###################################################################################################################################
### Start SPIN
test_auc_list = []
for experiment in range(1, n_experiments + 1):
    print("#######################  %d experiment  #######################\n" % experiment)
    ### load train & validation & test data and label
    trainData = pd.read_csv(data_path + f"Normed_Train_Data_{experiment}.csv")
    trainLabel = pd.read_csv(data_path + f"Normed_Train_Label_{experiment}.csv")
    valData = pd.read_csv(data_path + f"Normed_Valid_Data_{experiment}.csv")
    valLabel = pd.read_csv(data_path + f"Normed_Valid_Label_{experiment}.csv")
    testData = pd.read_csv(data_path + f"Normed_Test_Data_{experiment}.csv")
    testLabel = pd.read_csv(data_path + f"Normed_Test_Label_{experiment}.csv")
    ### Load data on GPU
    x_train = torch.from_numpy(trainData.values).to(dtype=torch.float).cuda()
    y_train = torch.from_numpy(trainLabel.values).to(dtype=torch.float).cuda()
    x_valid = torch.from_numpy(valData.values).to(dtype=torch.float).cuda()
    y_valid = torch.from_numpy(valLabel.values).to(dtype=torch.float).cuda()
    x_test = torch.from_numpy(testData.values).to(dtype=torch.float).cuda()
    y_test = torch.from_numpy(testLabel.values).to(dtype=torch.float).cuda()
###################################################################################################################################
    torch.cuda.empty_cache()
    test_auc = train_SPIN(date, num, data, experiment, train_x, train_y, valid_x, valid_y, test_x, test_y,
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
