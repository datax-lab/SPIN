from SPIN_Model_Risk import SPIN
from SPIN_Utils_Risk import *
from tqdm import tqdm
from sklearn.utils import class_weight

import numpy as np
import pandas as pd
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Load_Dataset(Dataset):
    def __init__(self, data, label, weight):
        self.data = data
        self.label = label
        self.weight = weight

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data.iloc[idx]).cuda(), torch.FloatTensor(self.label.iloc[idx]).cuda(), torch.FloatTensor(self.weight.iloc[idx]).cuda()

def train_SPIN(date, num, data, experiment, trainData, trainLabel, validData, validLabel, testData, testLabel, pathway_idx, net_hparams, optim_hparams, experim_hparms):
    ### set save path
    save_path = '''Set the path to save files & results'''
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
    ### Compute sample weight
    train_sample_weight = pd.DataFrame(class_weight.compute_sample_weight('balanced', trainLabel.values.ravel()))
    valid_sample_weight = pd.DataFrame(class_weight.compute_sample_weight('balanced', validLabel.values.ravel()))
    ### Load Dataset
    train_dataloader = DataLoader(Load_Dataset(trainData, trainLabel, train_sample_weight), batch_size = experim_hparms[1], shuffle = False)
    valid_dataloader = DataLoader(Load_Dataset(validData, validLabel, valid_sample_weight), batch_size = experim_hparms[1], shuffle = False)
    ###############################################################################################################################################
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    layer2_sp_level_during_epoch = []
    ###################################################################################################################################
    for epoch in range(1, experim_hparms[0] + 1):
        torch.cuda.empty_cache()
        net.train()
        train_step_loss = []
        for train_x, train_y, train_w in train_dataloader:
            ### forward
            train_pred, train_label, train_weight = net(train_x, train_y, train_w)
            ### calculate loss
            loss = F.binary_cross_entropy(train_pred, train_label, weight = train_weight)
            ### reset gradients to zeros
            opt.zero_grad()
            ### calculate gradients
            loss.backward()
            ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
            if net.layer1_female.weight.grad is not None:
                net.layer1_female.weight.grad = fixed_s_mask(net.layer1_female.weight.grad, pathway_idx)
            if net.layer1_male.weight.grad is not None:
                net.layer1_male.weight.grad = fixed_s_mask(net.layer1_male.weight.grad, pathway_idx)
            ### update weights and biases
            opt.step()
            ### append train loss per step
            train_step_loss.append(loss.item())
        ###################################################################################################################################
        train_loss = np.mean(train_step_loss)
        ### sparse network - prunning connections
        net, sp_level_list = sparse_func_risk(net, train_dataloader)
        layer2_sp_level_during_epoch.append(sp_level_list[0][1])
        del sp_level_list
        gc.collect()
        ###################################################################################################################################
        net.eval()
        ### validation data
        valid_loss = 0
        with torch.no_grad():
            for valid_x, valid_y, valid_w in valid_dataloader:
                valid_pred, valid_label, valid_weight = net(valid_x, valid_y, valid_w)
                valid_loss += F.binary_cross_entropy(valid_pred, valid_label, weight = valid_weight).item()
        valid_loss /= len(valid_dataloader)
        ### adjust learning rate based on validation loss
        scheduler.step(valid_loss)
        ### save the optimal model at the best validation loss
        best_valid_loss = float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            opt_net = copy.deepcopy(net)
            
    torch.save(opt_net, f"[{date}_{num}]_[{experiment}]Save_Opt_Model")
    ############################################################## Test ##############################################################
    test_sample_weight = pd.DataFrame(class_weight.compute_sample_weight('balanced', testLabel.values.ravel()))
    test_dataloader = DataLoader(Load_Dataset(testData, testLabel, test_sample_weight), batch_size = experim_hparms[1], shuffle = False)
    opt_net.eval()
    test_pred_list = []
    test_true_list = []
    test_sample_weight_list = []
    for test_x, test_y, test_w in test_dataloader:
        test_pred, test_label, test_weight = opt_net(test_x, test_y, test_w)
        test_pred_list.append(test_pred.reshape(-1,))
        test_true_list.append(test_label.reshape(-1,))
        test_sample_weight_list.append(test_weight.reshape(-1,))

    test_pred_list = torch.cat(test_pred_list, dim = 0)
    test_true_list = torch.cat(test_true_list, dim = 0)
    test_sample_weight_list = torch.cat(test_sample_weight_list, dim = 0)
    test_auc = auc(test_true_list, test_pred_list, test_sample_weight_list)
    
    return test_auc
