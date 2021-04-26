import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def shape_calc(input_shape):
    n = len(input_shape)
    res = 1
    for i in range(n):
        res *= input_shape[i]
    return res
class ResidualBlock(nn.Module):
    def __init__(self,input_dim,out_dim):
        super(ResidualBlock, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(in_features=input_dim,out_features=input_dim),
            nn.ReLU(),
            nn.Linear(in_features=input_dim,out_features=input_dim),
        )
        self.relu1 = nn.ReLU()
        self.trans1 = nn.Linear(in_features=input_dim,out_features=out_dim)

        self.net2 = nn.Sequential(
            nn.Linear(in_features=out_dim,out_features=out_dim),
            nn.ReLU(),
            nn.Linear(in_features=out_dim,out_features=out_dim)
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.net1(x)
        out +=residual
        out = self.relu1(out)
        out = self.trans1(out)
        residual = out
        out = self.net2(out)
        out +=residual
        out = self.relu2(out)
        return out


class SharedEncoder(nn.Module):
    """
    Input: nbatch,nagent,obs_dim + nbatch,nagent,policy + nbatch,nagent,reward, + nbatch,last_dim
    1. nbatch,nagent, (obs_dim + last_dim + policy + reward)
    2. generate weight matrix: nbatch,maxagent
    3. generate random projection: maxagent, agents
    4. weighted somme: nbatch,(obs_dim + last_dim + policy + reward)
    5. final latent : nbatch, latent_dim
    """
    def __init__(self,max_agents,input_dim,latent_dim,action_dim):
        super(SharedEncoder, self).__init__()
        self.max_agents =max_agents
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.action_dim = shape_calc(action_dim)
        self.total_input_dim = self.input_dim + self.action_dim + self.latent_dim + 1

        ## Preprocess
        self.policy_preprocess = nn.Sequential(
            nn.Linear(in_features=self.action_dim,out_features=self.action_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.action_dim,out_features=self.action_dim)
        )
        ## Para weighted
        self.weight_generator1 = nn.Sequential(
            nn.Linear(in_features=self.total_input_dim,out_features=max_agents),
            nn.ReLU(),
            nn.Linear(in_features=max_agents,out_features=max_agents)
        )
        self.weight_generator2 = nn.Sequential(
            nn.Linear(in_features=self.total_input_dim,out_features=max_agents),
            nn.ReLU(),
            nn.Linear(in_features=max_agents,out_features=max_agents)
        )
        self.weight_generator3 = nn.Sequential(
            nn.Linear(in_features=self.total_input_dim, out_features=max_agents),
            nn.ReLU(),
            nn.Linear(in_features=max_agents, out_features=max_agents)
        )
        self.final_layer = nn.Linear(in_features=3*max_agents,out_features=max_agents)

        ## Encoder Part
        dim1 = int(self.total_input_dim/4)
        dim2 = int(dim1/4)
        dim3 = int(dim2/4)
        self.intention_encoder1 = nn.Sequential(
            ResidualBlock(input_dim = self.total_input_dim,out_dim=dim1),
            ResidualBlock(input_dim=dim1,out_dim=dim2),
            ResidualBlock(input_dim=dim2,out_dim=dim3),
            nn.Linear(in_features=dim3,out_features=latent_dim)
        )
    def weight_generate(self,x):
        """
        x = nbatch,ngent,total_dim
        """
        x = torch.sum(x,dim=1)
        x1 = self.weight_generator1(x)
        x2 = self.weight_generator2(x)
        x3 = self.weight_generator3(x)
        x = torch.cat((x1,x2,x3),dim=-1)
        weight = self.final_layer(x)
        return weight


    def generate_convert_matrix(self,num_agent):
        weights = torch.Tensor(np.ones((self.max_agents,)))
        sample = torch.multinomial(weights, num_agent, replacement=False)
        convert_matrix = F.one_hot(sample,num_classes=self.max_agents).transpose(0,1)
        return convert_matrix

    def forward(self,obs_input:torch.Tensor,last_latent:torch.Tensor,policy:torch.Tensor,reward:torch.Tensor):
        """
        input: nbatch,nums_agents,dim

        intention = nbatch,dim
        after repeat = nbatch,nagents,dim
        """
        nbatch = obs_input.shape[0]
        num_agent = obs_input.shape[1]
        policy = self.policy_preprocess(policy.double())
        last_latent = last_latent.repeat(1,num_agent,1)
        #print(obs_input.shape)
        #print(last_latent.shape)
        #print(policy.shape)
        #print(reward.shape)


        total_input = torch.cat((obs_input.double(),last_latent,policy,reward.double()),dim=-1)

        initial_weight  = self.weight_generate(total_input) ## nbatch,max_agents
        convert_matrix = self.generate_convert_matrix(num_agent) ## max_agents,num_agents
        converted_weights = torch.matmul(initial_weight,convert_matrix.double()).view(nbatch,num_agent,1) ## nbatch,num_agents


        weighted_input = torch.sum(total_input*converted_weights,dim=1) ## nbatch,d
        intention = self.intention_encoder1(weighted_input)
        #print(intention.shape)
        #intention = intention.repeat(1,num_agent,1)

        return intention



class Decoder(nn.Module):
    """
    input: nbatch, (shared_latent + private_latent) + nbatch,nagents,(view+feature) at step t ;;; nbatch,num_agent,(view + feature) at step t+k
    for i in nagents:
        nbatch,(shared_latent + private_latent + view + feature) at step t -> at step t+k

    output: nbatch,
    """
    def __init__(self,view_dim,feature_dim,latent_dim):
        super(Decoder, self).__init__()
        self.view_dim = shape_calc(view_dim)
        self.feature_dim = shape_calc(feature_dim)

        self.net1 = nn.Sequential(
            nn.Linear(2*latent_dim+self.view_dim+self.feature_dim,out_features=self.view_dim + self.feature_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.view_dim + self.feature_dim,out_features=self.view_dim + self.feature_dim)
        )


    def forward(self,shared_latent,private_latent,view:torch.Tensor,feature:torch.Tensor):
        nbatch = view.shape[0]
        nagent = view.shape[1]
        shared_latent = shared_latent.repeat(1,nagent,1)
        private_latent = private_latent.repeat(1,nagent,1)

        total_latent = torch.cat((shared_latent,private_latent),dim = -1).double()
        ##print("total_latent",total_latent.shape)
        view  = view.view(nbatch,nagent,-1)
        feature = feature.view(nbatch,nagent,-1)

        obs = torch.cat((view,feature),dim = -1).double()
        #print("raw_obs",obs.shape)
        splited_obs = torch.chunk(obs,nagent,dim=1)
        splited_total_latent = torch.chunk(total_latent,nagent,dim=1)

        #print("split",len(splited_obs),splited_obs[0].shape)

        outputs = []

        for i in range(nagent):
            obs = splited_obs[i]
            lat = splited_total_latent[i]
            obs = obs.view(nbatch,-1)
            lat = lat.view(nbatch,-1)
            #print("obs",obs.shape)
            input = torch.cat((lat,obs),dim=-1)
            #print("input",input.shape)
            outputs.append(self.net1(input))
        outputs = torch.cat(outputs,dim=-1)
        outputs = outputs.view(nbatch,nagent,-1)

        return outputs