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

def train_SPIN(date, num, data, experiment, train_x, train_y, valid_x, valid_y, test_x, test_y, pathway_indices, net_hparams, optim_hparams, experim_hparms):
    ### set save path
    save_path = '''Set the path to save files & results'''
    net = SPIN(net_hparams, pathway_indices)
    if torch.cuda.is_available():
        net.cuda()

    ### Optimizer Setting
    if optimizer == "SGD":
        opt = optim.SGD(net.parameters(), lr = learning_rate, weight_decay = weight_decay, nesterov = True)
    elif optimizer == "Adam":
        opt = optim.Adam(net.parameters(), lr = learning_rate, weight_decay = weight_decay, amsgrad = True)
    elif optimizer == "AdamW":
        opt = optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = weight_decay, amsgrad = True)
        
    if learning_rate_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor = lr_factor, patience = lr_patience)
###################################################################################################################################
    train_loss_list = []
    val_loss_list = []
    train_auc_list = []
    val_auc_list = []
    layer2_sp_level_during_epoch = []
    best_val_auc = 0.
    best_val_loss = torch.tensor(float('inf'))
    ### compute sample weight
    train_sample_weight = class_weight.compute_sample_weight('balanced', train_y.cpu().detach().numpy().ravel())
    val_sample_weight = class_weight.compute_sample_weight('balanced', val_y.cpu().detach().numpy().ravel())
    gpu_train_sample_weight = torch.tensor(train_sample_weight).reshape(-1,1).cuda()
    gpu_val_sample_weight = torch.tensor(val_sample_weight).reshape(-1,1).cuda()
###################################################################################################################################
    for epoch in tqdm(range(1, n_epochs + 1)):
        torch.cuda.empty_cache()
        net.train()
        ### forward
        pred = net(train_x)
        ### calculate loss
        loss = F.binary_cross_entropy(pred, train_y, weight = gpu_train_sample_weight)
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
###################################################################################################################################
        ### sparse network - prunning connections
        net, sp_level_list = sparse_func_risk(net, train_x, train_y)
        layer2_sp_level_during_epoch.append(sp_level_list[0][1])
        del sp_level_list
        gc.collect()
        
        if epoch % step == 0:
            torch.cuda.empty_cache()
            net.train()
            train_pred = net(train_x)
            train_loss = F.binary_cross_entropy(train_pred, train_y, weight = gpu_train_sample_weight)
            train_loss_list.append(train_loss.cpu().detach().numpy())
            train_auc = auc(train_y, train_pred)
            train_auc_list.append(train_auc)
###################################################################################################################################
            net.eval()
            ### validation data
            with torch.no_grad():
                val_pred = net(val_x)
                val_loss = F.binary_cross_entropy(val_pred, val_y, weight = gpu_val_sample_weight)
                val_loss_list.append(val_loss.cpu().detach().numpy())
                val_auc = auc(val_y, val_pred)
                val_auc_list.append(val_auc)
###################################################################################################################################
            ### adjust learning rate based on validation loss
            if learning_rate_scheduler:
                scheduler.step(val_loss)            
            ### update best auc and ratio at best auc
            if best_val_auc < val_auc:
                best_val_auc = val_auc
                best_val_auc_epoch = epoch
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                val_auc_at_best_val_loss = val_auc
                opt_net = copy.deepcopy(net)                
                print("[%d] Train Loss in [%d]: %.4f" % (experiment, epoch, train_loss))
                print("[%d] Valid Loss in [%d]: %.4f" % (experiment, epoch, val_loss))
                print("[%d] Train AUC in [%d]: %.3f" % (experiment, epoch, train_auc))
                print("[%d] Valid AUC in [%d]: %.3f" % (experiment, epoch, val_auc))
                print("[%d] Best Valid AUC in [%d]: %.3f" % (experiment, best_val_auc_epoch, best_val_auc))
            else:
                print("[%d] Train Loss in [%d]: %.4f" % (experiment, epoch, train_loss))
                print("[%d] Valid Loss in [%d]: %.4f" % (experiment, epoch, val_loss))
                print("[%d] Train AUC in [%d]: %.3f" % (experiment, epoch, train_auc))
                print("[%d] Valid AUC in [%d]: %.3f" % (experiment, epoch, val_auc))
                print("[%d] Best Valid Loss in [%d]: %.4f" % (experiment, best_val_loss_epoch, best_val_loss))
                print("[%d] Valid AUC at Best Valid Loss in [%d]: %.3f" % (experiment, best_val_loss_epoch, val_auc_at_best_val_loss))
                print("[%d] Best Valid AUC in [%d]: %.3f" % (experiment, best_val_auc_epoch, best_val_auc))
                
            del train_pred, val_pred, train_loss, val_loss
            del train_auc, val_auc
            gc.collect()
            
    torch.save(opt_net, save_path + f"Saved_Model/[{date}_{num}]_[{experiment}]_Opt_Model.pt")
############################################################## Test ##############################################################
    opt_net.eval()
    final_pred = opt_net(test_x)
    final_auc = auc(test_y, final_pred)
    ### save ground truth & prediction of test data
    pd.DataFrame(np.concatenate(([final_pred.cpu().detach().numpy().ravel()], [test_y.cpu().detach().numpy().ravel()]), axis = 0).T).to_csv(save_path + f"Pred_&_Truth/[{date}_{num}]_[{experiment}]_SPIN_Pred_Truth.csv", index = False)
###################################################################################################################################
    ### plot learning curve
    plot_learning_curve(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, train_loss_list, val_loss_list)
    ### plot auc
    plot_auc(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, train_auc_list, val_auc_list)
    ### plot sparse ratio
    layer2_sp_level_during_epoch = np.array(layer2_sp_level_during_epoch, dtype = np.float)
    plot_remaining_ratio(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, layer2_sp_level_during_epoch)
    
    return final_auc
