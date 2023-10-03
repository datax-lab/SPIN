import numpy as np
import pandas as pd
import torch
from scipy import sparse

def sort_data(path):
    ''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
    Input:
    path: path to input dataset (which is expected to be a csv file).
    Output:
    x: sorted genomic inputs.
    ytime: sorted survival time (OS_MONTHS) corresponding to 'x'.
    yevent: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored.
    '''
    data = pd.read_csv(path)
    data.sort_values("OS_MONTHS", ascending = False, inplace = True)
    
    x = data.drop(["OS_STATUS", "OS_MONTHS"], axis = 1).values
    yevent = data.loc[:, ["OS_STATUS"]].values
    ytime = data.loc[:, ["OS_MONTHS"]].values
    
    return x, yevent, ytime

def load_data(path):
    '''Load the sorted data, and then covert it to a Pytorch tensor.
    Input:
    path: path to input dataset (which is expected to be a csv file).
    Output:
    X: a Pytorch tensor of 'x' from sort_data().
    YTIME: a Pytorch tensor of 'ytime' from sort_data().
    YEVENT: a Pytorch tensor of 'yevent' from sort_data().
    '''
    x, yevent, ytime = sort_data(path)
    
    gpu_x = torch.from_numpy(x).to(dtype = torch.float).cuda()
    gpu_yevent = torch.from_numpy(yevent).to(dtype = torch.float).cuda()
    gpu_ytime = torch.from_numpy(ytime).to(dtype = torch.float).cuda()
    
    return gpu_x, gpu_yevent, gpu_ytime

def load_sparse_indices(path):
    coo = sparse.load_npz(path)
    indices = np.vstack((coo.row, coo.col))
    
    return indices
