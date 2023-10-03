import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import copy

from scipy import sparse
from scipy.stats import bernoulli
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d
import gc

import torch.nn as nn
import torch.nn.functional as F

def load_sparse_indices(path):

    coo = sparse.load_npz(path)
    indices = np.vstack((coo.row, coo.col))

    return indices

def load_data(path):
    return pd.read_csv(path)

def upload_gpu(data, label):
    x = torch.from_numpy(data.values).to(dtype = torch.float).cuda()
    y = torch.from_numpy(label.values).to(dtype = torch.float).cuda()
    
    return x, y

def fixed_s_mask(w, idx):
    '''
    Input: 
        w: weight matrix.
        idx: the indices of having values (or connections).
    Output:
        returns the weight matrix that has been forced the connections.
    '''
    sp_w = torch.sparse_coo_tensor(idx, w[idx], w.size())
    
    return sp_w.to_dense()

def auc(y_true, y_pred, sample_weight = None):
    ###if gpu is being used, transferring back to cpu
    if torch.cuda.is_available():
        y_true = y_true.cpu().detach()
        y_pred = y_pred.cpu().detach()
        
    if sample_weight is None:
        auc = roc_auc_score(y_true.numpy(), y_pred.numpy())
    else:
        auc = roc_auc_score(y_true.numpy(), y_pred.numpy(), sample_weight = sample_weight)
        
    return auc

def plot_learning_curve(date, num, data, experiment, lr, wd, Dropout_Rates, *args):
    save_path = f"../../../nasdatafolder/MTL/SPIN/GSE_{data}_Result/"
    print("########### Learning Curve Plot ###########")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    weight_decay = r'$\lambda$'
    sparsity_level = r'$\alpha$'
    ax.set_title(f"LR: {lr}  {weight_decay}: {wd}  DR: {Dropout_Rates}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    for loss_list in args:
        ax.plot(loss_list)
        
    ax.legend(["Training Loss", "Validation Loss"])
    fig.savefig(save_path + f"Learning_Curve/[{date}_{num}]_[{experiment}]_Learning_Curve.png")        
    plt.close(fig)

def plot_auc(date, num, data, experiment, lr, wd, Dropout_Rates, *args):
    save_path = f"../../../nasdatafolder/MTL/SPIN/GSE_{data}_Result/"
    print("########### AUC Plot ###########")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    weight_decay = r'$\lambda$'
    sparsity_level = r'$\alpha$'
    ax.set_title(f"LR: {lr}  {weight_decay}: {wd}  DR: {Dropout_Rates}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    for auc_list in args:
        ax.plot(auc_list)
        
    ax.legend(["Training AUC", "Validation AUC"])
    fig.savefig(save_path + f"AUC_Plot/[{date}_{num}]_[{experiment}]_AUC.png")
    plt.close(fig)
    
def plot_remaining_ratio(date, num, data, experiment, lr, wd, Dropout_Rates, *args):
    save_path = f"../../../nasdatafolder/MTL/SPIN/GSE_{data}_Result/"
    print("########### Remaining Ratio Plot ###########")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    weight_decay = r'$\lambda$'
    sparsity_level = r'$\alpha$'
    ax.set_title(f"LR: {lr}  {weight_decay}: {wd}  DR: {Dropout_Rates}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sparsity Level")
    for layer_remaining_ratio in args:
        ax.plot(layer_remaining_ratio)
        
    ax.legend(["layer2 layer", "layer3 layer"])
    fig.savefig(save_path + f"Remaining_Ratio/[{date}_{num}]_[{experiment}]_Remaining_Ratio.png")        
    plt.close(fig)

def get_threshold(w, m, sp):
    '''Obtain the weight value associated to sparsity
    Input: 
        w: weight matrix
        m: the bi-adjacency matrix to indicate the weights that have been updated
        sp: sparsity level
    Output:
        returns the cutoff value (th in soft_threshold())
    '''
    pos_param = torch.abs(torch.masked_select(w, m))
    top_k = math.ceil(pos_param.size(0) * (100 - sp) * 0.01)
    '''if pos_param.sum() == 0:
        pos_parm = torch.Tensor([0] * top_k).cuda()'''
    return torch.topk(pos_param, top_k)[0][-1]

def soft_threshold(w, th):
    '''
    Soft-thresholding
    Input:
        w: weight matrix
        th: the cutoff value (output from get_threshold())
    Output:
        returns the shrinked weight matrix
    '''
    return torch.sign(w)*torch.clamp(abs(w) - th, min=0.0)

def get_sparse_weight(w, m, s):
    '''Generate the sparse weight matrix based on sparsity level'''
    epsilon = get_threshold(w, m, s)
    sp_w = soft_threshold(w, epsilon)
    return sp_w

def get_best_sparsity(sparse_set, loss_set):
    '''Estimate the best sparsity level by cubic interpolation'''
    interp_loss_set = interp1d(sparse_set, loss_set, kind = 'cubic')
    interp_sparse_set = torch.linspace(min(sparse_set), max(sparse_set), steps = 100)
    interp_loss = interp_loss_set(interp_sparse_set)
    best_sp = interp_sparse_set[np.argmin(interp_loss)]
    return best_sp

def small_net_mask(w, m_in_nodes, m_out_nodes):
    '''Generate the masks in order to locate the trained weights in the selected small sub-network'''
    nonzero_idx_in = m_in_nodes.nonzero()
    nonzero_idx_out = m_out_nodes.nonzero()
    gcd = math.gcd(nonzero_idx_in.shape[0], nonzero_idx_out.shape[0])
    if gcd == 1:
        sparse_row_idx = nonzero_idx_out.repeat(nonzero_idx_in.size()).transpose(1, -2)
        sparse_col_idx = nonzero_idx_in.repeat(nonzero_idx_out.size()).transpose(1, -2)
    else:
        row = rotate = nonzero_idx_out.repeat(torch.Size([int(nonzero_idx_in.size()[0] / gcd), 1]))
        while gcd > 1:
            rotate = torch.roll(rotate, -1)
            row = torch.cat((row, rotate))
            gcd -= 1        
        sparse_row_idx = row.transpose(1, -2)
        sparse_col_idx = nonzero_idx_in.repeat(nonzero_idx_out.size()).transpose(1, -2)
        
    idx = torch.cat((sparse_row_idx, sparse_col_idx), 0)
    val = torch.ones(nonzero_idx_out.size(0) * nonzero_idx_in.size(0))
    sparse_bool_mask = torch.sparse_coo_tensor(idx, val, w.size()).cuda()
    
    return sparse_bool_mask.to_dense().type(torch.uint8)

'''Sparse coding phrase: optimize the connections between intermediate layers sequentially'''
def sparse_func_survival(net, x_tr, yevent_tr, ytime_tr):
    torch.cuda.empty_cache()
    ###serializing net 
    net_state_dict = net.state_dict()
    copy_net = copy.deepcopy(net)
    copy_state_dict = copy_net.state_dict()
    sp_level_list = []
    for name, param in net_state_dict.items():
        torch.cuda.empty_cache()
        ###omit the param if it is not a weight matrix
        if not "weight" in name: continue
        ###omit gene layer
        if "layer1" in name: continue
        if "layer3" in name: continue
        if "bn1" in name: continue
        if "bn2" in name: continue
        if "layer2" in name:
            active_mask = small_net_mask(net.layer2.weight.data, torch.ones(net.layer2.weight.shape[1]), torch.ones(net.layer2.weight.shape[0]))
            copy_weight = copy.deepcopy(net.layer2.weight.data)
            
        S_set = torch.linspace(99, 0, 10)
        S_loss = []
        for S in S_set:
            sp_param = get_sparse_weight(copy_weight, active_mask, S.item())
            copy_state_dict[name].copy_(sp_param)
            torch.cuda.empty_cache()
            copy_net.train()
            pred_tmp = copy_net(x_tr)
            loss_tmp = neg_par_log_likelihood(pred_tmp, yevent_tr, ytime_tr)
            S_loss.append(loss_tmp.cpu().detach().numpy())
            
        ### apply cubic interpolation
        best_S = get_best_sparsity(S_set, S_loss)
        optimal_sp_param = get_sparse_weight(copy_weight, active_mask, best_S)
        copy_weight[active_mask] = optimal_sp_param[active_mask]
        sp_level_list.append([name, best_S.item()])
        
        ### update weights in copied net
        copy_state_dict[name].copy_(copy_weight)
        ### update weights in net
        net_state_dict[name].copy_(copy_weight)
        del active_mask, copy_weight
        gc.collect()
    
    del copy_net, copy_state_dict
    gc.collect()
    
    return net, sp_level_list

def R_set(x):
    '''Create an indicator matrix of risk sets, where T_j >= T_i.
    Note that the input data have been sorted in descending order.
    Input:
    x: a PyTorch tensor that the number of rows is equal to the number of samples.
    Output:
    indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
    '''
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    
    return indicator_matrix

def neg_par_log_likelihood(pred, yevent, ytime):
    '''Calculate the average Cox negative partial log-likelihood.
    Input:
    pred: linear predictors from trained model.
    ytime: true survival time from load_data().
    yevent: true censoring status from load_data().
    Output:
    cost: the cost that is to be minimized.
    '''
    n_observed = yevent.sum(0)
    #print(n_observed)
    ytime_indicator = R_set(ytime)
    ###if gpu is being used
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
        
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))
    #print(risk_set_sum)
    diff = pred - torch.log(risk_set_sum)
    #print(diff)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    #print(sum_diff_in_observed)
    cost = (-(sum_diff_in_observed / n_observed)).reshape((-1,))
    
    return cost

def c_index(pred, yevent, ytime):
    '''Calculate concordance index to evaluate models.
    Input:
    pred: linear predictors from trained model.
    ytime: true survival time from load_data().
    yevent: true censoring status from load_data().
    Output:
    concordance_index: c-index (between 0 and 1).
    '''
    n_sample = len(ytime)
    ytime_indicator = R_set(ytime)
    ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
    ###T_i is uncensored
    censor_idx = (yevent == 0).nonzero()
    zeros = torch.zeros(n_sample)
    ytime_matrix[censor_idx, :] = zeros
    ###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
    pred_matrix = torch.zeros_like(ytime_matrix)
    for j in range(n_sample):
        for i in range(n_sample):
            if pred[i] < pred[j]:
                pred_matrix[j, i]  = 1
            elif pred[i] == pred[j]:
                pred_matrix[j, i] = 0.5
                
    concord_matrix = pred_matrix.mul(ytime_matrix)
    ###numerator
    concord = torch.sum(concord_matrix)
    ###denominator
    epsilon = torch.sum(ytime_matrix)
    ###c-index = numerator/denominator
    concordance_index = torch.div(concord, epsilon)
    ###if gpu is being used
    if torch.cuda.is_available():
        concordance_index = concordance_index.cuda()
        
    return concordance_index
