from algo.DSN_test.Attention import Attention_Module
from algo.DSN_test.Encoder import SharedEncoder,Decoder
from algo.DSN_test.Buffer import DSN_Group
import torch.nn as nn
import torch
import numpy as np

class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE,self).__init__()
    def forward(self,pred,real):
        diffs = torch.add(pred,-real)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2)/(n**2)
        return simse
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse
""" Difference Loss for Private and Shared """
class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()
    def forward(self,input1,input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size,-1)
        input2 = input2.view(batch_size,-1)

        input1_l2norm = torch.norm(input1,p=2,dim=1,keepdim=True).detach()
        input2_l2norm = torch.norm(input2,p=2,dim=1,keepdim=True).detach()

        input1_l2 = input1.div(input1_l2norm.expand_as(input1)+1e-6)
        input2_l2 = input2.div(input2_l2norm.expand_as(input2)+1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        return diff_loss

class MMD(nn.Module):
    def __init__(self):
        super(MMD,self).__init__()

    def _mix_rbf_kernel(self,X, Y, sigma_list):
        assert (X.size(0) == Y.size(0))
        m = X.size(0)

        Z = torch.cat((X, Y), 0)
        ZZT = torch.mm(Z, Z.t())
        diag_ZZT = torch.diag(ZZT).unsqueeze(1)
        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += torch.exp(-gamma * exponent)

        return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

    def mix_rbf_mmd2(self,X, Y, sigma_list, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigma_list)
        # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

    def _mmd2(self,K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = K_XX.size(0)  # assume X, Y are same shape

        # Get the various sums of kernels that we'll use
        # Kts drop the diagonal, but we don't need to compute them explicitly
        if const_diagonal is not False:
            diag_X = diag_Y = const_diagonal
            sum_diag_X = sum_diag_Y = m * const_diagonal
        else:
            diag_X = torch.diag(K_XX)  # (m,)
            diag_Y = torch.diag(K_YY)  # (m,)
            sum_diag_X = torch.sum(diag_X)
            sum_diag_Y = torch.sum(diag_Y)

        Kt_XX_sums = K_XX.sum(dim=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = K_YY.sum(dim=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = K_XY.sum(dim=0)  # K_{XY}^T * e

        Kt_XX_sum = Kt_XX_sums.sum()  # e^T * \tilde{K}_XX * e
        Kt_YY_sum = Kt_YY_sums.sum()  # e^T * \tilde{K}_YY * e
        K_XY_sum = K_XY_sums_0.sum()  # e^T * K_{XY} * e

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2.0 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2.0 * K_XY_sum / (m * m))

        return mmd2
    def forward(self,input1,input2):
        sigma_list = [1]
        return self.mix_rbf_mmd2(input1,input2,sigma_list)
class DSN:
    def __init__(self,view_dim,feature_dim,action_dim,latent_dim1,latent_dim2,max_agents,learning_rate):
        self.attention = Attention_Module(view_dim=view_dim,feature_dim=feature_dim,output_dim=latent_dim1,max_agents=max_agents,h=1)
        self.shared_encoder = SharedEncoder(max_agents=max_agents,input_dim=latent_dim1,latent_dim=latent_dim2,action_dim=action_dim).double()
        self.private_encoder = SharedEncoder(max_agents=max_agents,input_dim=latent_dim1,latent_dim=latent_dim2,action_dim=action_dim).double()
        self.decoder = Decoder(view_dim=view_dim,feature_dim=feature_dim,latent_dim=latent_dim2).double()

        self.latent_dim2 = latent_dim2

        """Loss Definition"""
        self.mmd = MMD()
        self.diff = DiffLoss()
        self.simse = SIMSE()
        self.mse = MSE()

        """ Group Memory """
        self.next_k = 1
        self.memory = DSN_Group(next_k=self.next_k)

        """ Optimizer """
        self.shared_encoder_optimizer = torch.optim.Adam(self.shared_encoder.parameters(),lr=learning_rate)
        self.private_encoder_optimizer = torch.optim.Adam(self.private_encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)

    def compute_similartiy_loss(self,data,data_next_t):
        view = data['view']
        feature = data['feature']
        last_latent = data['last_latent']
        log_pi = data['old_policy']
        reward = data['reward']

        atte_output = self.attention(view,feature)
        latent = self.shared_encoder(atte_output,last_latent, log_pi, reward)

        view = data_next_t['view']
        feature = data_next_t['feature']
        last_latent = data_next_t['last_latent']
        log_pi = data_next_t['old_policy']
        reward = data_next_t['reward']
        atte_output = self.attention(view, feature)
        latent_next_t = self.shared_encoder(atte_output, last_latent, log_pi, reward)
        similarity_loss = self.mmd(latent,latent_next_t)
        return similarity_loss

    def compute_reconstrction_loss(self,data):
        view = data['view']
        feature = data['feature']
        last_latent = data['last_latent']
        log_pi = data['old_policy']
        reward = data['reward']

        nbatch  = view.shape[0]
        nagent = view.shape[1]

        atte_output = self.attention(view, feature)
        shared_latent = self.shared_encoder(atte_output, last_latent, log_pi, reward)
        private_latent = self.private_encoder(atte_output,last_latent,log_pi,reward)

        next = self.decoder(shared_latent, private_latent, view, feature)
        view = view.view(nbatch, nagent, -1)
        feature = feature.view(nbatch, nagent, -1)
        view = view.view(nbatch, nagent, -1)
        feature = feature.view(nbatch, nagent, -1)

        obs = torch.cat((view, feature), dim=-1).double()

        loss1 = self.simse(next,obs)
        loss2 = self.mse(next,obs)
        return loss2 + loss1

    def compute_difference_loss(self,data):
        view = data['view']
        feature = data['feature']
        last_latent = data['last_latent']
        log_pi = data['old_policy']
        reward = data['reward']

        atte_output = self.attention(view, feature)
        shared_latent = self.shared_encoder(atte_output, last_latent, log_pi, reward)
        private_latent = self.private_encoder(atte_output, last_latent, log_pi, reward)
        loss = self.diff(shared_latent,private_latent)
        return loss

    def push(self,data):
        view = data['view']
        nagents = len(view)
        feature = data['feature']
        policy = data['old_policy']
        reward = data['reward']
        reward = np.reshape(reward,newshape=(nagents,1))
        data['reward'] = reward

        if len(self.memory)==0:
            last_lat = np.random.rand(self.latent_dim2).reshape(self.latent_dim2)
            data['this_latent'] = last_lat
            #print("first",last_lat.shape)
        else:
            ##print(feature.shape)
            atte_output = self.attention(torch.from_numpy(np.array([view])), torch.from_numpy(np.array([feature])))
            this_latent = self.shared_encoder(atte_output, torch.from_numpy(self.memory.latent_var[self.memory.point-1]),torch.from_numpy(np.array([policy])), torch.from_numpy(np.array([reward])))
            this_latent = this_latent.view(self.latent_dim2)
            this_latent = this_latent.detach().numpy()
            #print("after_latent",this_latent.shape)
            data['this_latent']  = this_latent
        self.memory.push(data)


    def update(self):
        l = len(self.memory)
        loss_log = 0
        self.memory.point-=self.next_k + 1

        for i in range(l-1-self.next_k-1):
            next_k_data = self.memory.pull_next_k()
            data = self.memory.pull()

            self.shared_encoder_optimizer.zero_grad()
            self.private_encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            similarity_loss = self.compute_similartiy_loss(data,next_k_data)
            recons_loss = self.compute_reconstrction_loss(data)
            diff_loss = self.compute_difference_loss(data)

            loss = similarity_loss+recons_loss+diff_loss
            loss_log+=loss
            loss.backward()
            self.shared_encoder_optimizer.step()
            self.private_encoder_optimizer.step()
            self.decoder_optimizer.step()

        self.memory.clear()
        loss_log/=l-1-self.next_k
        return loss_log







