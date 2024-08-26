import pandas as pd
import numpy as np
import random
import os

from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

from DataLoader import load_data, load_sparse_indices
from SPIN_Utils_Survival import *
from SPIN_Model_Survival import SPIN
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
arser = argparse.ArgumentParser()
parser.add_argument("--gpu", type = str, help = "GPU number", required = True)
parser.add_argument("--data", type = str, help = "Data name", required = True)
parser.add_argument("--epoch", type = int, help = "Epoch Size", required=True)
parser.add_argument("--btch", type = int, help = "Batch Size", required=True)
args = parser.parse_args()

### GPU assign
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

date = datetime.today().strftime('%m%d')
data = args.data

class Load_Dataset(Dataset):
    def __init__(self, data, event, time):
        self.data = data
        self.event = event
        self.time = time

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data.iloc[idx]).cuda(), torch.FloatTensor(self.event.iloc[idx]).cuda(), torch.FloatTensor(self.time.iloc[idx]).cuda()

def train_SPIN_Survival(config):
    n_experiments = 1
    ### Net setting
    ### Number of genes
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
    
    net_hparams = [in_Nodes, [173, 100], 1, config['Initializer'], config['Activation'], config['Dropout']] ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-dropout
    optim_hparams = ["Adam", config['LR'], config['LR_Factor'], config['LR_Patience'], config['Weight_Decay']] ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay
    experim_hparms = [args.epoch, args.btch] ### 0-max_epoch, 1-batch_size
    for experiment in range(1, n_experiments + 1):
        print("#######################  %d experiment  #######################\n" % experiment)
        ### load train & validation & test data and label
        trainData, trainEvent, trainTime = load_data("Load Train Data")
        validData, validEvent, validTime = load_data("Load Valid Data")
        ###############################################################################################################################################
        train_dataloader = DataLoader(Load_Dataset(trainData, trainEvent, trainTime), batch_size = experim_hparms[1], shuffle = False)
        valid_dataloader = DataLoader(Load_Dataset(validData, validEvent, validTime), batch_size = experim_hparms[1], shuffle = False)
        ###############################################################################################################################################
        net = SPIN(net_hparams, pathway_idx)
        if torch.cuda.is_available():
            net.cuda()
        ### Optimizer Setting
        ### 0-optimizer, 1-lr, 2-lr_scheduler, 3-lr_factor, 4-lr_patience
        if optim_hparams[0] == "SGD":
            opt = optim.SGD(net.parameters(), lr = optim_hparams[1], momentum = 1e-1, dampening = 0, weight_decay = optim_hparams[4], nesterov = True)
        elif optim_hparams[0] == "Adam":
            opt = optim.Adam(net.parameters(), lr = optim_hparams[1], betas = (0.99, 0.999), eps = 1e-8, weight_decay = optim_hparams[4], amsgrad = True)
        elif optim_hparams[0] == "AdamW":
            opt = optim.AdamW(net.parameters(), lr = optim_hparams[1], betas = (0.99, 0.999), eps = 1e-8, weight_decay = optim_hparams[4], amsgrad = True)
        ### Learning scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor = optim_hparams[2], patience = optim_hparams[3])
    ###############################################################################################################################################
        for epoch in range(1, experim_hparms[0] + 1):
            torch.cuda.empty_cache()
            net.train()
            train_step_loss = []
            for train_x, train_event, train_time in train_dataloader:
                ### forward
                train_pred, train_yevent, train_ytime = net(train_x, train_event, train_time)
                ### calculate loss
                loss = neg_par_log_likelihood(train_pred, train_yevent, train_ytime)
                ### reset gradients to zeros
                opt.zero_grad()
                ### calculate gradients
                loss.backward()
                ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
                if net.layer1_female.weight.grad is not None:
                    net.layer1_female.weight.grad = fixed_s_mask(net.layer1_female.weight.grad, pathway_indices)
                if net.layer1_male.weight.grad is not None:
                    net.layer1_male.weight.grad = fixed_s_mask(net.layer1_male.weight.grad, pathway_indices)
                ### update weights and biases
                opt.step()
                ### append train loss per step
                train_step_loss.append(loss.item())

            train_loss = np.mean(train_step_loss)

            net.eval()
            ### validation data
            valid_loss = 0
            with torch.no_grad():        
                for valid_x, valid_event, valid_time in valid_dataloader:
                    valid_pred, valid_yevent, valid_ytime = net(valid_x, valid_event, valid_time)
                    valid_loss += neg_par_log_likelihood(valid_pred, valid_yevent, valid_ytime).item()
            valid_loss /= len(valid_dataloader)

            scheduler.step(valid_loss)

            train.report({"Training loss": train_loss, "Validation loss": valid_loss})


config = {
    "Dropout": tune.uniform(0, 1),
    "LR": tune.loguniform(1e-6, 1e-2),
    "LR_Factor": tune.uniform(0, 1),
    "LR_Patience": tune.choice([5, 10, 20, 30]),
    "Activation": tune.choice(["sigmoid", "tanh", "relu", "lkrelu"]),
    "Initializer": tune.choice(["he_normal", "he_uniform", "xavier_normal", "xavier_uniform"]),
    "Weight_Decay": tune.uniform(0, 1e-1)
}

scheduler = ASHAScheduler(
    time_attr='training_iteration',
    metric="Validation loss",
    mode="min",
    max_t=100,
    grace_period=20,
    reduction_factor=2,
)

search_alg = OptunaSearch(metric="Validation loss", mode="min")
    
result = tune.run(
    train_SPIN_Survival,
    config=config,
    resources_per_trial={"cpu": 4, "gpu": 1},
    num_samples=500,
    scheduler=scheduler,
    search_alg=search_alg,
    storage_path='/home/koe3/CMS/Race_GroupNN/Hyperparameters/'
)

best_trial = result.get_best_trial("Validation loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['Validation loss']}")
