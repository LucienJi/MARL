import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
def shape_calc(input_shape):
    n = len(input_shape)
    res = 1
    for i in range(n):
        res *= input_shape[i]
    return res

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, input:torch.Tensor):
        return input.view(input.size(0), -1)

def attention(query,key,value,mask = None):
    """
    query = batch_size,k_channel,num_agents,d_k
    mask = batch_size,k_channel,num_agents,
    """
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    if mask is not  None:
        scores = scores.masked_fill(mask==0,-1e9)
    p_attn = F.softmax(scores,dim=-1)
    ## p_attn = batch_size,k_channel, num_agent,num_agent

    return torch.matmul(p_attn,value),p_attn

class Attention_Module(nn.Module):
    def __init__(self,view_dim,feature_dim,output_dim,max_agents,h = 1):
        super(Attention_Module, self).__init__()
        self.view_dim = shape_calc(view_dim)
        self.feature_dim = shape_calc(feature_dim)
        self.input_dim = self.view_dim + self.feature_dim
        self.max_agents = max_agents
        self.flatten = Flatten()
        self.key_encoder = nn.Linear(in_features=self.input_dim,out_features=output_dim)
        self.query_encoder = nn.Linear(in_features=self.input_dim,out_features=output_dim)
        self.value_encoder = nn.Linear(in_features=self.input_dim,out_features=output_dim)
        self.last_encoder = nn.Linear(in_features=output_dim,out_features=output_dim)
        self.d_k = output_dim//h
        self.h = h
        self.attn = None



    def forward(self, view:torch.Tensor,feature:torch.Tensor):
        """
        view = nbatch,nagent,view_dim,
        feature = nbatch,nagent,feature_dim
        """
        shape = view.shape
        nbatch = shape[0]
        nagent = shape[1]
        view = view.view(nbatch,nagent,-1)
        feature = feature.view(nbatch,nagent,-1)
        input = torch.cat((view,feature),dim=-1)
        query = self.query_encoder(input).view(nbatch,-1,self.h,self.d_k).transpose(1,2)
        key = self.key_encoder(input).view(nbatch,-1,self.h,self.d_k).transpose(1,2)
        value = self.value_encoder(input).view(nbatch,-1,self.h,self.d_k).transpose(1,2)

        new_value,self.attn = attention(query=query,key=key,value=value,mask=None)
        new_value = new_value.transpose(1,2).contiguous().view(nbatch,-1,self.h*self.d_k)
        new_value = self.last_encoder(new_value) ## batch_size,num_agent,output_dim
        return new_value


