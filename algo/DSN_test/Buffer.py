import numpy as np
import torch
class DSN_Group:
    """
    1. nbatch,nagent,view
    2. nbatch,nagent,feature
    3. nbatch,nagent,policy
    5. nbatch,nagent,reward
    5. nbatch,latent_var
    6.
    """
    def __init__(self,next_k):
        self.view = []
        self.feature = []
        self.policy = []
        self.reward = []
        self.latent_var = []
        self.point = -1
        self.next_k = next_k

    def push(self,data):
        self.view.append(np.array([data['view']]))
        self.feature.append(np.array([data['feature']]))
        self.policy.append(np.array([data['old_policy']]))
        self.reward.append(np.array([data['reward']]))
        self.latent_var.append(np.array([data['this_latent']]))
        self.point+=1

    def pull(self):
        assert self.point>=1

        data = {}
        data['view']  = torch.from_numpy(self.view[self.point])
        data['feature'] = torch.from_numpy(self.feature[self.point])
        data['last_latent'] = torch.from_numpy(self.latent_var[self.point-1])
        data['old_policy'] = torch.from_numpy(self.policy[self.point])
        data['reward'] = torch.from_numpy(self.reward[self.point])
        data['this_latent'] = torch.from_numpy(self.latent_var[self.point])
        self.point-=1
        return data

    def pull_next_k(self):
        assert self.point+self.next_k <len(self.view)
        data = {}
        data['view'] = torch.from_numpy(self.view[self.point+self.next_k])
        data['feature'] = torch.from_numpy(self.feature[self.point+self.next_k])
        data['last_latent'] = torch.from_numpy(self.latent_var[self.point - 1+self.next_k])
        data['old_policy'] = torch.from_numpy(self.policy[self.point+self.next_k])
        data['reward'] = torch.from_numpy(self.reward[self.point+self.next_k])
        data['this_lat'] = torch.from_numpy(self.latent_var[self.point+self.next_k])
        return data

    def clear(self):
        self.view = []
        self.feature = []
        self.policy = []
        self.reward = []
        self.latent_var = []
        self.point = -1

    def __len__(self):
        return self.point+1



