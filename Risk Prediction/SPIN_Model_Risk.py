import numpy as np
import math
import torch
import torch.nn as nn
from SPIN_Utils_Risk import fixed_s_mask

class SPIN(nn.Module):
    def __init__(self, net_hparams, pathway_idx):
        super(SPIN, self).__init__()
        self.pathway_idx = pathway_idx
        ###########################################################################
        ### gene layer --> pathway layer
        ###########################################################################
        self.layer1_female = nn.Linear(net_hparams[0], net_hparams[1][0]) ### for female
        self.layer1_male = nn.Linear(net_hparams[0], net_hparams[1][0]) ### for male
        ### initialize weight & bias
        self.init_parameters(initializer, self.layer1_female.weight.data, self.layer1_female.bias)
        self.init_parameters(initializer, self.layer1_male.weight.data, self.layer1_male.bias)
        ###########################################################################
        ### pathway layer --> hidden layer
        ###########################################################################
        self.layer2 = nn.Linear(net_hparams[1][0], net_hparams[1][1])
        self.init_parameters(initializer, self.layer2.weight.data, self.layer2.bias)
        ###########################################################################
        ### hidden layer --> Output layer
        ###########################################################################
        self.layer3 = nn.Linear(net_hparams[1][1], net_hparams[2])
        self.init_parameters(initializer, self.layer3.weight.data, self.layer3.bias)
        ###########################################################################
        ### batch normalization & initialization
        ###########################################################################
        self.bn1 = nn.BatchNorm1d(net_hparams[1][0])
        self.bn2 = nn.BatchNorm1d(net_hparams[1][1])
        ###########################################################################
        ### activation
        ###########################################################################
        if net_hparams[4] == "sigmoid":
            self.layer1_activation = nn.Sigmoid()
            self.layer2_activation = nn.Sigmoid()
        elif net_hparams[4] == "tanh":
            self.layer1_activation = nn.Tanh()
            self.layer2_activation = nn.Tanh()
        elif net_hparams[4] == "relu":
            self.layer1_activation = nn.ReLU()
            self.layer2_activation = nn.ReLU()
        elif net_hparams[4] == "lkrelu":
            self.layer1_activation = nn.LeakyReLU()
            self.layer2_activation = nn.LeakyReLU()
        
        ### output activation for binary classification
        self.output_activation = nn.Sigmoid()

        ### dropout
        if net_hparams[5] != 0.:
            self.on_Dropout = True
            self.dropout = nn.Dropout(net_hparams[5])
        else:
            self.on_Dropout = False
    
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
            
    def forward(self, x, y, w):
        '''
        Input: 
            x: gene expression data, where the last column is sex information (e.g., 0 - female, 1 - male).            
        Output:
            return the SPIN's prediction
        '''
        torch.cuda.empty_cache()
        ##################################################################################
        ### gene layer --> pathway layer
        if (0 not in x[:, -1]) and (1 not in x[:, -1]):
            raise ValueError("No Sex Information")
        else:
            x_layer1 = []
            labels = []
            weights = []
            if 0 in x[:, -1]:
                ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
                self.layer1_female.weight.data = fixed_s_mask(self.layer1_female.weight.data, self.pathway_idx)
                idx_female = (x[:, -1] == 0).nonzero(as_tuple=True)
                x_female = self.layer1_activation(self.layer1_female(x[:, :-1][idx_female]))
                x_layer1.append(x_female)
                labels.append(y[idx_female])
                weights.append(w[idx_female])
            if 1 in x[:, -1]:
                ### force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
                self.layer1_male.weight.data = fixed_s_mask(self.layer1_male.weight.data, self.pathway_idx)
                idx_male = (x[:, -1] == 0).nonzero(as_tuple=True)
                x_male = self.layer1_activation(self.layer1_male(x[:, :-1][idx_male]))
                x_layer1.append(x_male)
                labels.append(y[idx_male])
                weights.append(w[idx_male])
            
            self.layer1_female.weight.data = fixed_s_mask(self.layer1_female.weight.data, self.pathway_idx)
            self.layer1_male.weight.data = fixed_s_mask(self.layer1_male.weight.data, self.pathway_idx)
            x_female = self.activation(self.layer1_female(x[:, :-1][x[:, -1] == 0]))
            x_male = self.activation(self.layer1_male(x[:, :-1][x[:, -1] == 1]))            
            x = self.bn1(torch.cat((x_female, x_male), 0))
        ### concatenate all sex group into a layer
        x = torch.cat(x_layer1, dim = 0)
        labels = torch.cat(labels, dim = 0)
        weights = torch.cat(weights, dim = 0)

        try:
            x = self.bn1(x)
        except:
            x = x
        if self.training & self.on_Dropout:
            x = self.dropout(x)
        ### pathway --> hidden layer
        x = self.layer2_activation(self.layer2(x))
        try:
            x = self.bn2(x)
        except:
            x = x
        if self.training & self.on_Dropout:
            x = self.dropout(x)
        ### hidden --> output layer
        x = self.output_activation(self.layer3(x))
        
        return x, labels, weights
