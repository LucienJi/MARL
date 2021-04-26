import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from python import magent
import algo.DSN_test1.DSN_test1 as DSN
import algo.DSN_test1.Replaybuffer as replaybuffer

class Agent:
    def __init__(self,env: magent.gridworld.GridWorld, handle, batch_size=1024,learning_rate=1e-5, discount_coef=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = discount_coef
        self.batch_size = batch_size
        self.handle = handle
        """ Basic Environment Info """
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.action_space = env.get_action_space(handle)[0]

        """ Network """

        self.replaybuffer = replaybuffer.GroupMemory(self.view_space, self.feature_space, self.action_space, max_buffer_size=2**10, batch_size=self.batch_size, sub__len=2**10,
                 n_step=3, gamma=0.99)

        self.agent = DSN.ActorCritic(self.view_space,self.feature_space,self.action_space,embedded_size=16)
        self.pi_optimizer = torch.optim.Adam(self.agent.policy.parameters(),lr=learning_rate)
        self.q_optimizer  = torch.optim.Adam(self.agent.qvalue.parameters(),lr=learning_rate)
        self.target_encoder_optimizer = torch.optim.Adam(self.agent.target_private_encoder.parameters(),lr=learning_rate)
        self.source_encoder_optimizer = torch.optim.Adam(self.agent.source_private_encoder.parameters(),lr = learning_rate)
        self.shared_encoder_optimizer = torch.optim.Adam(self.agent.shared_encoder.parameters(),lr = learning_rate)
        self.decoder_optimizer = torch.optim.Adam(self.agent.decoder.parameters(),lr=learning_rate)
    def compute_loss_pi(self,data,clip_ratio = 0.2):
        """
        You have to set data as tensor
        policy = logits
        """
        view,feature,action,reward,log_pi,q = data['view'],data['feature'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']

        a_indice = F.one_hot(action.long(),num_classes=int(self.action_space))
        adv = (reward - torch.sum(q*torch.exp(log_pi),dim=1))
        #print("test",type(view),type(feature))
        now_log_pi = self.agent.eval_policy(view,feature)
        now_log_pi = torch.sum(a_indice * now_log_pi,dim = 1)
        log_pi = torch.sum(a_indice * log_pi,dim = 1)

        ratio = torch.exp(now_log_pi - log_pi)
        clip_adv = torch.clamp(ratio,1-clip_ratio,1+clip_ratio)*adv
        loss_pi = -(torch.min(ratio*adv,clip_adv)).mean()
        return loss_pi


    def compute_loss_q(self,data):
        """
        You have to set data as tensor
        """
        view,feature,action,reward,log_pi,q = data['view'],data['feature'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']
        q = self.agent.eval_qvalue(view,feature)
        q_value = torch.sum(q*torch.exp(log_pi),dim=1)
        loss_q = torch.mean((reward - q_value)**2)

        return loss_q

    def update(self):
        num_agents = self.env.get_agent_id(self.handle)
        for i in num_agents:
            source_data,target_data = self.replaybuffer.sample(_id=i)
            """ Q and Policy network """
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(source_data)
            loss_pi.backward()
            self.pi_optimizer.step()

            self.q_optimizer.zero_grad()
            loss_q = self.compute_loss_q(source_data)
            loss_q.backward()
            self.q_optimizer.step()

            """ Encoder network 
            self.target_encoder_optimizer = torch.optim.Adam(self.agent.target_private_encoder,lr=learning_rate)
            self.source_encoder_optimizer = torch.optim.Adam(self.agent.source_private_encoder,lr = learning_rate)
            self.shared_encoder_optimizer = torch.optim.Adam(self.agent.shared_encoder,lr = learning_rate)
            self.decoder_optimizer = torch.optim.Adam(self.agent.decoder,lr=learning_rate)
            """
            self.target_encoder_optimizer.zero_grad()
            self.source_encoder_optimizer.zero_grad()
            self.shared_encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            s_recons_loss = self.agent.compute_recons_loss(source_data['view'],source_data['feature'],method='Source')
            t_recons_loss = self.agent.compute_recons_loss(target_data['view'],target_data['feature'],method='Target')

            s_diff_loss = self.agent.compute_difference_loss(source_data['view'],source_data['feature'],method='Source')
            t_diff_loss = self.agent.compute_difference_loss(target_data['view'],target_data['feature'],method='Target')

            similarity_loss = self.agent.compute_similarity_loss(target_data['view'],target_data['feature'],source_data['view'],source_data['feature'])

            total_loss = s_recons_loss + t_recons_loss + s_diff_loss + t_diff_loss + similarity_loss
            total_loss.backward()
            self.target_encoder_optimizer.step()
            self.source_encoder_optimizer.step()
            self.shared_encoder_optimizer.step()
            self.decoder_optimizer.step()

            return loss_q.detach().numpy(),total_loss.detach().numpy()

    def act(self, data):
        view = data['view']
        feature = data['feature']
        res = self.agent.act(torch.as_tensor(view), torch.as_tensor(feature)).astype(np.int32)

        return res

    def step(self, data):
        view = data['view']
        feature = data['feature']
        a, pi, q = self.agent.step(torch.as_tensor(view), torch.as_tensor(feature))
        return pi, q

    def save(self, path,i=1):
        file_path = os.path.join(path, "Test1" + '_%d') % i
        torch.save(self.agent.state_dict(), file_path)
        print("Saved")

    def load(self, path,i=1):
        file_path = os.path.join(path, "Test1" + '_%d') % i
        self.agent.load_state_dict(torch.load(file_path))
        print("Loaded")




