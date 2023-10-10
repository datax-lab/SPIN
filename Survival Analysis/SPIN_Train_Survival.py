from SPIN_Model_Survival import SPIN
from SPIN_Utils_Survival import *
from tqdm import tqdm

import numpy as np
import pandas as pd
import copy
import torch
import torch.optim as optim
import torch.nn.functional as F

def train_SPIN(date, num, data, experiment, train_x, train_yevent, train_ytime, valid_x, valid_yevent, valid_ytime, 
               test_x, test_yevent, test_ytime, pathway_indices,
               in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes, dropout_Rates, initializer, activation,
               learning_rate, weight_decay, lr_factor, lr_patience, n_epochs, 
               step = 1, optimizer = "Adam", learning_rate_scheduler = False):
    ### set save path
    save_path = '''Set the path to save files & results'''
    net = SPIN(in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes, pathway_indices, initializer, activation, dropout_Rates)
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
    train_cindex_list = []
    val_cindex_list = []
    
    layer2_sp_level_during_epoch = []
    best_val_loss = np.inf
    best_val_loss_epoch = step
    val_cindex_at_best_val_loss = 0.
    best_val_cindex = 0.
###################################################################################################################################
    for epoch in tqdm(range(1, n_epochs + 1)):
        torch.cuda.empty_cache()
        net.train()
        ### forward
        pred = net(train_x)
        ### calculate loss
        loss = neg_par_log_likelihood(pred, train_yevent, train_ytime)
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
        net, sp_level_list = sparse_func_survival(net, train_x, train_yevent, train_ytime)
        layer2_sp_level_during_epoch.append(float(sp_level_list[0][1]))
        del sp_level_list
        gc.collect()
        
        if epoch % step == 0:
            torch.cuda.empty_cache()
            net.train()
            train_pred = net(train_x)
            train_loss = neg_par_log_likelihood(train_pred, train_yevent, train_ytime).view(1,)
            train_loss_list.append(train_loss)
            train_cindex = c_index(train_pred, train_yevent, train_ytime)
            train_cindex_list.append(train_cindex)
###################################################################################################################################
            net.eval()
            ### validation data
            with torch.no_grad():
                val_pred = net(valid_x)
                val_loss = neg_par_log_likelihood(val_pred, valid_yevent, valid_ytime).cpu().detach().numpy()
                if np.isnan(val_loss):
                    val_loss = np.inf
                val_loss_list.append(val_loss)
                val_cindex = c_index(val_pred, valid_yevent, valid_ytime)
                val_cindex_list.append(val_cindex)
###################################################################################################################################
            ### adjust learning rate based on validation loss
            if learning_rate_scheduler:
                scheduler.step(val_loss)            
            ### update best cindex and ratio at best cindex
            if best_val_cindex < val_cindex:
                best_val_cindex = val_cindex
                best_val_cindex_epoch = epoch
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                val_auc_at_best_val_loss = val_auc
                opt_net = copy.deepcopy(net)
                print("[%d] Train Loss in [%d]: %.4f" % (experiment, epoch, train_loss))
                print("[%d] Valid Loss in [%d]: %.4f" % (experiment, epoch, val_loss))
                print("[%d] Train C-Index in [%d]: %.3f" % (experiment, epoch, train_cindex))
                print("[%d] Valid C-Index in [%d]: %.3f" % (experiment, epoch, val_cindex))
                print("[%d] Best Valid C-Index in [%d]: %.3f" % (experiment, best_val_cindex_epoch, best_val_cindex))
            else:
                print("[%d] Train Loss in [%d]: %.4f" % (experiment, epoch, train_loss))
                print("[%d] Valid Loss in [%d]: %.4f" % (experiment, epoch, val_loss))
                print("[%d] Train C-Index in [%d]: %.3f" % (experiment, epoch, train_cindex))
                print("[%d] Valid C-Index in [%d]: %.3f" % (experiment, epoch, val_cindex))
                print("[%d] Best Valid Loss in [%d]: %.4f" % (experiment, best_val_loss_epoch, best_val_loss))
                print("[%d] Valid C-Index at Best Valid Loss in [%d]: %.3f" % (experiment, best_val_loss_epoch, val_cindex_at_best_val_loss))
                print("[%d] Best Valid C-Index in [%d]: %.3f" % (experiment, best_val_cindex_epoch, best_val_cindex))
                
            del train_pred, val_pred, train_loss, val_loss
            del train_cindex, val_cindex
            gc.collect()
            
    torch.save(opt_net, save_path + f"Saved_Model/[{date}_{num}]_[{experiment}]_Opt_Model.pt")
############################################################## Test ##############################################################
    opt_net.eval()
    final_pred = opt_net(test_x)
    final_cindex = c_index(final_pred, test_yevent, test_ytime)
    ### save ground truth & prediction of test data
    pd.DataFrame(np.concatenate(([final_pred.cpu().detach().numpy().ravel()], [test_yevent.cpu().detach().numpy().ravel()], [test_ytime.cpu().detach().numpy().ravel()]), axis = 0).T).to_csv(save_path + f"Pred_&_Truth/[{date}_{num}]_[{experiment}]_Cox-SPIN_Pred_Truth.csv", index = False)
###################################################################################################################################
    ### plot learning curve
    plot_learning_curve(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, train_loss_list, val_loss_list)
    ### plot c-index
    plot_cindex(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, train_cindex_list, val_cindex_list)
    ### plot sparse ratio
    layer2_sp_level_during_epoch = np.array(layer2_sp_level_during_epoch, dtype = np.float)
    plot_remaining_ratio(date, num, data, experiment, learning_rate, weight_decay, dropout_Rates, layer2_sp_level_during_epoch)
    
    return final_cindex.cpu().detach().numpy()
