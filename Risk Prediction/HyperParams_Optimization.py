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

from SPIN_Model_Risk import SPIN
from Utils import *
from tqdm import tqdm
from sklearn.utils import class_weight
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import warnings
warnings.filterwarnings('ignore')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

date = datetime.today().strftime('%m%d')

class Load_Dataset(Dataset):
    def __init__(self, data, label, weight):
        self.data = data
        self.label = label
        self.weight = weight

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data.iloc[idx]).cuda(), torch.FloatTensor(self.label.iloc[idx]).cuda(), torch.FloatTensor(self.weight.iloc[idx]).cuda()

def train_Race_GroupNN(config):
    n_experiments = 1
    net_hparams = [23, [11, 10], 1, "", config['Activation'], config['Dropout']] ### 0-input_nodes, 1-hidden_nodes, 2-output_nodes, 3-initializer, 4-activation, 5-dropout
    optim_hparams = ["Adam", config['LR'], config['LR_Factor'], config['LR_Patience'], config['Weight_Decay']] ### 0-optimizer, 1-lr, 2-lr_factor, 3-lr_patience, 4-weight_decay
    experim_hparms = [100, 2**8] ### 0-max_epoch, 1-batch_size
    sparse_indices = load_sparse_indices("/home/koe3/CMS/Data/Mask_Group_Var_withoutRace.npz")
    model_path = f"/home/koe3/CMS/{state}_Result/GroupNN/Saved_Model/"
    for experiment in range(1, n_experiments + 1):
        print("#######################  %d experiment  #######################\n" % experiment)
        ### load pretrained model
        pretrained_model = torch.load(model_path + f"[0314_1]_[{experiment}]Opt_Model.pt")
        ### each dataset includes race information
        ### the race information placed on the last column
        ### 1-White, 2-Black, 3-Hispanic, 4-Asian, 5-Other
        trainData = pd.read_csv(f"/home/koe3/CMS/Data/[0307]{state}_SIDC_CORE_Normed_Train_Data_{experiment}.csv")
        trainLabel = pd.read_csv(f"/home/koe3/CMS/Data/[0307]{state}_SIDC_CORE_Normed_Train_Label_{experiment}.csv")
        validData = pd.read_csv(f"/home/koe3/CMS/Data/[0307]{state}_SIDC_CORE_Normed_Valid_Data_{experiment}.csv")
        validLabel = pd.read_csv(f"/home/koe3/CMS/Data/[0307]{state}_SIDC_CORE_Normed_Valid_Label_{experiment}.csv")
        # testData = pd.read_csv(f"../Data/[0307]{state}_SIDC_CORE_Normed_Test_Data_{experiment}.csv")
        # testLabel = pd.read_csv(f"../Data/[0307]{state}_SIDC_CORE_Normed_Test_Label_{experiment}.csv")
        ###############################################################################################################################################
        train_sample_weight = pd.DataFrame(class_weight.compute_sample_weight('balanced', trainLabel.values.ravel()))
        valid_sample_weight = pd.DataFrame(class_weight.compute_sample_weight('balanced', validLabel.values.ravel()))
        train_dataloader = DataLoader(Load_Dataset(trainData, trainLabel, train_sample_weight), batch_size = experim_hparms[1], shuffle = False)
        valid_dataloader = DataLoader(Load_Dataset(validData, validLabel, valid_sample_weight), batch_size = experim_hparms[1], shuffle = False)
        ###############################################################################################################################################
        net = Model(pretrained_model, net_hparams, sparse_indices)
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
        if optim_hparams[2]:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor = optim_hparams[2], patience = optim_hparams[3])
    ###############################################################################################################################################
        for epoch in range(1, experim_hparms[0] + 1):
            torch.cuda.empty_cache()
            net.train()
            train_step_loss = []
            for train_x, train_y, train_w in train_dataloader:
                ### train on each race
                # for race in train_x[:, -1].unique():
                #     train_race_idx = (train_x[:, -1] == race).nonzero(as_tuple=True)
                #     train_race_x = train_x[train_race_idx]
                #     train_race_y = train_y[train_race_idx]
                #     train_race_w = train_w[train_race_idx]
                ### forward
                # train_pred = net(train_race_x)
                train_pred, train_label, train_weight = net(train_x, train_y, train_w)
                ### calculate loss
                loss = F.binary_cross_entropy(train_pred, train_label, weight = train_weight)
                # train_batch_loss_list.append(train_loss.cpu().detach().numpy())
                ### reset gradients to zeros
                opt.zero_grad()
                ### calculate gradients
                loss.backward()
                ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
                if net.layer1_white.weight.grad is not None:
                    net.layer1_white.weight.data = fixed_s_mask(net.layer1_white.weight.data, sparse_indices)
                if net.layer1_black.weight.grad is not None:
                    net.layer1_black.weight.data = fixed_s_mask(net.layer1_black.weight.data, sparse_indices)
                if net.layer1_hispanic.weight.grad is not None:
                    net.layer1_hispanic.weight.data = fixed_s_mask(net.layer1_hispanic.weight.data, sparse_indices)
                if net.layer1_asian.weight.grad is not None:
                    net.layer1_asian.weight.data = fixed_s_mask(net.layer1_asian.weight.data, sparse_indices)
                if net.layer1_other.weight.grad is not None:
                    net.layer1_other.weight.data = fixed_s_mask(net.layer1_other.weight.data, sparse_indices)
                ### update weights and biases
                opt.step()
                ### append train loss per step
                train_step_loss.append(loss.item())

            train_loss = np.mean(train_step_loss)

            net.eval()
            ### validation data
            valid_loss = 0
            with torch.no_grad():        
                for valid_x, valid_y, valid_w in valid_dataloader:
                    valid_pred, valid_label, valid_weight = net(valid_x, valid_y, valid_w)
                    valid_loss += F.binary_cross_entropy(valid_pred, valid_label, weight = valid_weight).item()
            valid_loss /= len(valid_dataloader)

            scheduler.step(valid_loss)

            train.report({"Training loss": train_loss, "Validation loss": valid_loss})


config = {
    "Dropout": tune.uniform(0, 1),
    "LR": tune.loguniform(1e-6, 1e-2),
    "LR_Factor": tune.uniform(0, 1),
    "LR_Patience": tune.choice([5, 10, 20, 30]),
    "Activation": tune.choice(["sigmoid", "tanh", "relu", "lkrelu"]),
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

# first_config = {'Dropout': 0.06484167717898467, 'GridSize': 3, 'KANLayers': 0, 'KANMHA': 0, 'KANSize1': 64, 'KANSize2': 256, 'KANSize3': 16, 'Lr': 0.00023036437005511046, 'SplineOrder': 3}

search_alg = OptunaSearch(metric="Validation loss", mode="min")#, points_to_evaluate=[first_config])

#search_alg = OptunaSearch(metric="Validation loss", mode="min")
    
result = tune.run(
    train_Race_GroupNN,
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
# result.results_df.to_csv(f"")
