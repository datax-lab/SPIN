import numpy as np
import math
import torch
import torch.nn as nn
from SPIN_Utils_Survival import fixed_s_mask

class SPIN(nn.Module):
    def __init__(self, in_Nodes, pathway_Nodes, hidden_Nodes, out_Nodes, pathway_idx, initializer = "he_normal", activation = "Relu", dropout_Rates = .0):
        super(SPIN, self).__init__()        
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Relu":
            self.activation = nn.ReLU()
        
        self.pathway_idx = pathway_idx
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_Rates)
        ###########################################################################
        ### gene layer --> pathway layer
        ###########################################################################
        self.layer1_female = nn.Linear(in_Nodes, pathway_Nodes) ### for female
        self.layer1_male = nn.Linear(in_Nodes, pathway_Nodes) ### for male
        ### initialize weight & bias
        self.init_parameters(initializer, self.layer1_female.weight.data, self.layer1_female.bias)
        self.init_parameters(initializer, self.layer1_male.weight.data, self.layer1_male.bias)
        ###########################################################################
        ### pathway layer --> hidden layer
        ###########################################################################
        self.layer2 = nn.Linear(pathway_Nodes, hidden_Nodes)
        self.init_parameters(initializer, self.layer2.weight.data, self.layer2.bias)
        ###########################################################################
        ### hidden layer --> Output layer
        ###########################################################################
        self.layer3 = nn.Linear(hidden_Nodes, out_Nodes, bias = False)
        self.init_parameters(initializer, self.layer3.weight.data)
        ###########################################################################
        ### batch normalization & initialization
        ###########################################################################
        self.bn1 = nn.BatchNorm1d(pathway_Nodes, eps = 1e-3, momentum = 0.99)
        self.bn2 = nn.BatchNorm1d(hidden_Nodes, eps = 1e-3, momentum = 0.99)
    
    def init_parameters(self, init, weight, bias = None):
        ### initialize weight
        if init == "he_normal":
            nn.init.kaiming_normal_(weight, mode = 'fan_in')
        elif init == "he_uniform":
            nn.init.kaiming_uniform_(weight, mode = 'fan_in')
        elif init == "xavier_normal":
            nn.init.xavier_normal_(weight)
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(weight)
        ### initialize bias
        if bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(bias, -bound, bound)
            
    def forward(self, x):
        '''
        Input: 
            x: gene expression data, where the last column is sex information (e.g., 0 - female, 1 - male).            
        Output:
            return the SPIN's prediction
        '''
        torch.cuda.empty_cache()
        ##################################################################################
        ### gene layer --> pathway layer
        if (0 in x[:, -1]) and (1 in x[:, -1]):
            ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
            self.layer1_female.weight.data = fixed_s_mask(self.layer1_female.weight.data, self.pathway_idx)
            self.layer1_male.weight.data = fixed_s_mask(self.layer1_male.weight.data, self.pathway_idx)
            x_female = self.activation(self.layer1_female(x[:, :-1][x[:, -1] == 0]))
            x_male = self.activation(self.layer1_male(x[:, :-1][x[:, -1] == 1]))            
            x = self.bn1(torch.cat((x_female, x_male), 0))
        elif 0 in x[:, -1]:
            self.layer1_female.weight.data = fixed_s_mask(self.layer1_female.weight.data, self.pathway_idx)
            x = self.bn1(self.activation(self.layer1_female(x[:, :-1][x[:, -1] == 0])))
        elif 1 in x[:, -1]:
            self.layer1_male.weight.data = fixed_s_mask(self.layer1_male.weight.data, self.pathway_idx)
            x = self.bn1(self.activation(self.layer1_male(x[:, :-1][x[:, -1] == 1])))
        else:
            raise ValueError("No information on the task vector")
        ### dropout pathway layer
        if self.training:
            x = self.dropout(x)
        ##################################################################################
        ### pathway layer --> hidden layer
        x = self.bn2(self.activation(self.layer2(x)))
        ### dropout hidden layer
        if self.training:
            x = self.dropout(x)
        ##################################################################################
        ### hidden layer --> output layer
        x = self.layer3(x)
            
        return x
    
