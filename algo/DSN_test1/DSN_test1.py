import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)
def preprocess(input_size,output_size):
    net = nn.Sequential(
        Flatten(),
        nn.Linear(in_features=input_size,out_features=output_size)
    )

    return net

def mlp(input_size,embedded_size):
    net = nn.Sequential(
        nn.Linear(in_features=input_size,out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size, out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size,out_features=embedded_size)

    )

    return net
def shape_calc(input_shape):
    n = len(input_shape)
    res = 1
    for i in range(n):
        res *= input_shape[i]
    return res



class Private_source_encoder(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size=16):
        super(Private_source_encoder,self).__init__()
        v_space = shape_calc(view_space)
        f_space = shape_calc(feature_space)
        a_space = action_space

        self.net1 = preprocess(v_space,embedded_size*2)
        self.net2 = preprocess(f_space,embedded_size*2)

        self.net3 = mlp(embedded_size*4,embedded_size)
    def forward(self,view,feature):
        v = self.net1(view)
        f = self.net2(feature)
        h = torch.cat((v,f),dim=1)
        res = self.net3(h)
        return res



class Private_target_encoder(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size=16):
        super(Private_target_encoder,self).__init__()
        v_space = shape_calc(view_space)
        f_space = shape_calc(feature_space)
        a_space = action_space

        self.net1 = preprocess(v_space, embedded_size * 2)
        self.net2 = preprocess(f_space, embedded_size * 2)

        self.net3 = mlp(embedded_size * 4, embedded_size)

    def forward(self, view, feature):
        v = self.net1(view)
        f = self.net2(feature)
        h = torch.cat((v, f), dim=1)
        res = self.net3(h)
        return res

class Shared_encoder(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size=16):
        super(Shared_encoder,self).__init__()
        v_space = shape_calc(view_space)
        f_space = shape_calc(feature_space)
        a_space = action_space

        self.net1 = preprocess(v_space, embedded_size * 2)
        self.net2 = preprocess(f_space, embedded_size * 2)

        self.net3 = mlp(embedded_size * 4, embedded_size)

    def forward(self, view, feature):
        v = self.net1(view)
        f = self.net2(feature)
        h = torch.cat((v, f), dim=1)
        res = self.net3(h)
        return res

class Decoder(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size):
        super(Decoder,self).__init__()
        self.view_space = view_space
        self.feature_space = feature_space
        self.action_space = action_space
        self.embedded_size = embedded_size

        v_space = shape_calc(view_space)
        f_space = shape_calc(feature_space)
        self.net1 = nn.Sequential(
            nn.Linear(in_features=embedded_size,out_features=2*embedded_size),
            nn.ReLU(),
            nn.Linear(in_features=2*embedded_size,out_features=4*embedded_size),
        )
        self.net2 = nn.Sequential(
            nn.Linear(in_features=4*embedded_size,out_features=v_space),
            nn.ReLU(),
            nn.Linear(in_features=v_space,out_features=v_space),
            nn.ReLU(),
            nn.Linear(in_features=v_space,out_features=v_space)
        )
        self.net3 = nn.Sequential(
            nn.Linear(in_features=4 * embedded_size, out_features=f_space),
            nn.ReLU(),
            nn.Linear(in_features=f_space, out_features=f_space),
            nn.ReLU(),
            nn.Linear(in_features=f_space, out_features=f_space)
        )
    def forward(self,embedded_var):
        batch_size = embedded_var.size(0)

        h = self.net1(embedded_var)
        h_view = self.net2(h)
        h_feature = self.net3(h)

        recons_view = torch.reshape(h_view,(batch_size,)+self.view_space)
        recons_feature = torch.reshape(h_feature,(batch_size,)+self.feature_space)
        return recons_view,recons_feature



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


class Policy_network(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size):
        super(Policy_network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_size,2*embedded_size),
            nn.ReLU(),
            nn.Linear(2*embedded_size,action_space),
            nn.ReLU(),
            nn.Linear(action_space,action_space)
        )

    def forward(self,input_embdded_var):
        h = self.net(input_embdded_var)
        log_pi = F.log_softmax(h,dim=1)
        return log_pi

class Qvalue_network(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size):
        super(Qvalue_network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_size, 2 * embedded_size),
            nn.ReLU(),
            nn.Linear(2 * embedded_size, action_space),
            nn.ReLU(),
            nn.Linear(action_space, action_space)
        )
    def forward(self,input_embd_var):
        q = self.net(input_embd_var)
        return q

class ActorCritic(nn.Module):
    def __init__(self,view_space,feature_space,action_space,embedded_size):
        super(ActorCritic,self).__init__()
        self.policy = Policy_network(view_space,feature_space,action_space,embedded_size)
        self.qvalue = Qvalue_network(view_space,feature_space,action_space,embedded_size)
        self.source_private_encoder = Private_source_encoder(view_space,feature_space,action_space,embedded_size)
        self.target_private_encoder = Private_target_encoder(view_space,feature_space,action_space,embedded_size)
        self.shared_encoder = Shared_encoder(view_space,feature_space,action_space,embedded_size)
        self.decoder = Decoder(view_space,feature_space,action_space,embedded_size)

        self.loss_diff = DiffLoss()
        self.loss_recons1 = MSE()
        self.loss_recons2 = SIMSE()
        self.loss_similarity = MMD()

    def eval_policy(self,view,feature):
        emb = self.shared_encoder(view,feature)
        log_pi = self.policy(emb)
        return log_pi

    def eval_qvalue(self,view,feature):
        emb = self.shared_encoder(view,feature)
        q = self.qvalue(emb)
        return q

    def eval_encoder(self,view,feature):
        with torch.no_grad():
            shared_var = self.shared_encoder(view,feature)
            s_private = self.source_private_encoder(view,feature)
            t_private = self.target_private_encoder(view,feature)
        return shared_var.numpy(),s_private.numpy(),t_private.numpy()


    def step(self,view,feature):
        with torch.no_grad():
            embedded_vars = self.shared_encoder(view,feature)
            log_pi = self.policy(embedded_vars)
            q = self.qvalue(embedded_vars)
            pi = torch.exp(log_pi)
            a = Categorical(logits=log_pi).sample()
        #print("Test:",np.shape(log_pi.numpy()))
        return q.numpy(),log_pi.numpy(),a.numpy()

    def act(self,view,feature):
        q,log_pi,a = self.step(view,feature)
        return a


    def compute_recons_loss(self,view,feature,method = 'Source'):
        shared_embedded_vars = self.shared_encoder(view,feature)
        if method == 'Source':
            private_embedded_vars = self.source_private_encoder(view,feature)
            embedded_vars = shared_embedded_vars + private_embedded_vars
        else:
            private_embedded_vars = self.target_private_encoder(view,feature)
            embedded_vars = shared_embedded_vars + private_embedded_vars

        re_view,re_feature = self.decoder(embedded_vars)
        recons_loss = self.loss_recons1(re_view,view) + self.loss_recons1(re_feature,feature)
        recons_loss +=self.loss_recons2(re_view,view) + self.loss_recons2(re_feature,feature)

        return recons_loss

    def compute_difference_loss(self,view,feature,method = 'Source'):
        shared_embedded_vars = self.shared_encoder(view,feature)
        if method == 'Source':
            private_embedded_vars = self.source_private_encoder(view,feature)
        else:
            private_embedded_vars = self.target_private_encoder(view,feature)
        diff = self.loss_diff(shared_embedded_vars,private_embedded_vars)
        return diff


    def compute_similarity_loss(self,target_view,target_feature,source_view,source_feature):
        source_shared_embedded_var = self.shared_encoder(source_view,source_feature)
        target_shared_embedded_var = self.shared_encoder(target_view,target_feature)
        loss = self.loss_similarity(source_shared_embedded_var,target_shared_embedded_var)
        return loss






