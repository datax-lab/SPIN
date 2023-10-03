import torch

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
