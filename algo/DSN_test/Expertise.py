import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os

def discount_cumsum(x,discount):
    keep = 0
    N = len(x)
    for i in reversed(range(N)):
        x[i] = keep*discount + x[i]
        keep = x[i]

    return x

class SumTree(object):
    def __init__(self, r, v, alpha):

        """
        data = array of priority
        :param data: pos 0 no ele; ele from [1 to 2N[
        """

        td = ((r - v) * (r - v)) ** alpha
        td = td / np.sum(td)
        data = td
        l = len(data)
        if l % 2 == 0:
            self.N = l
        else:
            self.N = l - 1
        self.data = np.zeros(2 * self.N)
        self.data[self.N:2 * self.N] = data[0:self.N]

    def _sum(self, i):
        if 2 * i >= self.N:
            self.data[i] = self.data[2 * i] + self.data[2 * i + 1]
            return self.data[i]
        else:
            self.data[i] = self._sum(2 * i) + self._sum(2 * i + 1)
            return self.data[i]

    def build(self):
        self.total_p = self._sum(1)

    def find(self, p):
        idx = 1
        while idx < self.N:
            l = 2 * idx
            r = l + 1
            if self.data[l] >= p:
                idx = l
            else:
                idx = r
                p = p - self.data[l]
        return idx - self.N

    def sample(self, batchsize):
        real_index = []
        interval = self.total_p / batchsize
        for i in range(batchsize):
            try:
                p = np.random.uniform(i * interval, (i + 1) * interval)
            except OverflowError:
                print("OverflowError\n")
                print("Check interval:",interval)
                real_index.append(i)
            else:
                real_index.append(self.find(p))

        return real_index

class MetaBuffer(object):

    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0
        self.shape = shape
        self.max_len = max_len
        self.dtype = dtype

    # this is overridee?
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):

        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def add(self, value):
        self.data[self._flag] = value
        self._flag += 1
        self.length += 1

    def reset_new(self, start, value):
        self.data[start:self.length] = value

    def clear(self):
        self.data = np.zeros((self.max_len,) + self.shape).astype(self.dtype)
        self.start = 0
        self.length = 0
        self._flag = 0


class AgentMemory(object):
    """
    as the name indicate, it is only for a single agent which make it possible
    to single agent
    """

    def __init__(self, view_space, feature_space, action_space, n_step, gamma, max_buffer_size):
        self.view_space = view_space
        self.feature_space = feature_space
        self.action_space = action_space
        self.max_buffer_size = max_buffer_size
        self.n = n_step
        self.gamma = gamma

        """ Agent  Buffer """
        self.view_buffer = MetaBuffer(shape=self.view_space, max_len=self.max_buffer_size)
        self.feature_buffer = MetaBuffer(shape=self.feature_space, max_len=self.max_buffer_size)
        self.action_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size, dtype='int16')
        self.reward_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)
        self.adv_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)
        self.old_policy = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)


    def append(self, view, feature, action, reward, old_policy, old_qvalue):
        #print("test",old_policy.shape)
        self.view_buffer.append(np.array([view]))
        self.feature_buffer.append(np.array([feature]))
        self.action_buffer.append(np.array([action], dtype=np.int32))
        self.reward_buffer.append(np.array([reward]))
        self.old_policy.append(np.array([old_policy]))
        self.old_qvalue.append(np.array([old_qvalue]))


    def pull(self):
        res = {
            'view': self.view_buffer.pull(),
            'feature': self.feature_buffer.pull(),
            'action': self.action_buffer.pull(),
            'reward': self.reward_buffer.pull(),
            'old_policy': self.old_policy.pull(),
            'old_qvalue': self.old_qvalue.pull(),
            'adv':self.adv_buffer.pull()
        }

        return res

    def clear(self):
        self.view_buffer.clear()
        self.feature_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.old_qvalue.clear()
        self.old_policy.clear()
        self.adv_buffer.clear()

        return
    def reshape(self):
        rews = self.reward_buffer.pull()
        q = self.old_qvalue.pull()
        pi = self.old_policy.pull()
        vals = np.sum(q*np.exp(pi),axis=-1)
        lam = 0.95

        # GAE-Lambda adv
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buffer.append(discount_cumsum(deltas,self.gamma * lam))

        # reward to go
        self.reward_buffer.reset_new(0,discount_cumsum(rews,self.gamma))

        self.reward_buffer.length-=1
        assert self.reward_buffer.length == self.adv_buffer.length
        self.old_policy.length-=1
        self.old_qvalue.length-=1
        self.view_buffer.length-=1
        self.feature_buffer.length-=1
        self.action_buffer.length-=1

        return





    """
    def reshape(self):
        if self.n != -1:
            gamma = self.gamma
            n = self.n
            N = len(self.reward_buffer)
            reward = self.reward_buffer.pull()
            assert len(reward) == N
            action = self.action_buffer.pull()
            action_indice = np.eye(self.action_space)[action]

            old_qvalue = self.old_qvalue.pull()
            value = np.sum(action_indice * old_qvalue, axis=-1)
            r = reward[0:N - 1 - n]
            v = value[0:N - 1 - n]
            for i in range(1, n):
                r = r + reward[i:N - 1 - n + i] * (gamma ** i)
            r = r + v * (gamma ** n)
            reward[0:N - 1 - n] = r
            reward[N - 1 - n:N] = reward[N - 1 - n:N] + gamma * value[N - 1 - n:N]
            self.reward_buffer.reset_new(0, reward)
        else:
            gamma = self.gamma
            N = len(self.reward_buffer)
            reward = self.reward_buffer.pull()
            action = self.action_buffer.pull()
            action_indice = np.eye(self.action_space[0])[action]
            old_qvalue = self.old_qvalue.pull()
            keep = np.sum(action_indice * old_qvalue, axis=-1)
            keep = keep[-1]
            for i in reversed(range(N)):
                keep = reward[i] + keep * gamma
                reward[i] = keep
            self.reward_buffer.reset_new(0, reward)

        return
    """

class GroupMemory(object):
    """
    we try to mix the all agents'memory together

    by calling methods, we no longer distinguish the step memory of certain agent

    we collect memory of single agent in the container <<AgentMemory>>,and put it together

    """

    def __init__(self, view_space, feature_space, action_space, max_buffer_size, batch_size, sub__len,
                 n_step=3, gamma=0.99):
        # config to define the memory agent
        # batch_size: define the size of returned sample()
        # sub_len: define length for agents' memory
        self.view_space = view_space
        self.feature_space = feature_space
        self.action_space = action_space
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.sub_len = sub__len
        self.n = n_step
        self.gamma = gamma
        # big container for whole group
        """ Agent  Buffer """
        self.view_buffer = MetaBuffer(shape=self.view_space, max_len=self.max_buffer_size)
        self.feature_buffer = MetaBuffer(shape=self.feature_space, max_len=self.max_buffer_size)
        self.action_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size, dtype='int16')
        self.reward_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)
        self.old_policy = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)
        self.adv_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)


        # the container for the agents
        # we can retrieve the agent's memory by this
        self.agents = dict()
        self.buffer_length = 0

    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['id']):
            if self.agents.get(_id) is None:
                self.agents[_id] = AgentMemory(self.view_space, self.feature_space, self.action_space,
                                               n_step=self.n,
                                               max_buffer_size=self.max_buffer_size, gamma=self.gamma)

            self.agents[_id].append(view=kwargs['view'][i],
                                    feature=kwargs['feature'][i],
                                    action=kwargs['action'][i],
                                    reward=kwargs['reward'][i],
                                    old_policy=kwargs['old_policy'][i],
                                    old_qvalue=kwargs['old_qvalue'][i],
                                )

    def _flush(self, **kwargs):
        """

        :param kwargs: kwargs: {"views" : get by agent.pull(),
                        "actions":
                        "rewards":,
                        "alives":,
                        }
        :return: make the agent's memory together into the big container
        """
        self.view_buffer.append(kwargs["view"])
        self.feature_buffer.append(kwargs["feature"])

        self.action_buffer.append(kwargs["action"])
        self.reward_buffer.append(kwargs["reward"])
        self.old_qvalue.append(kwargs['old_qvalue'])
        self.old_policy.append(kwargs['old_policy'])
        self.adv_buffer.append(kwargs["adv"])

    def tight(self):
        """
        eat all agent's memory
        put it in the bigger container, we nolonger consider the order
        :return: enrich the big container
        """

        all_agents = list(self.agents.keys())

        # disturb the order
        np.random.shuffle(all_agents)
        for id in all_agents:
            """
            pull call the method in AgentMemory
            and call the method in Metabuffer
            """

            self.agents[id].reshape()
            agent_memory = self.agents[id].pull()
            self.buffer_length += len(agent_memory["reward"])
            self._flush(**agent_memory)

        # clear the agent's memory
        self.agents = dict()

        action = self.action_buffer.pull()
        action_indice = np.eye(self.action_space)[action]

        # old_policy = self.old_policy.pull()
        old_qvalue = self.old_qvalue.pull()
        value = np.sum(action_indice * old_qvalue, axis=-1)

        self.tree = SumTree(self.reward_buffer.pull(), value, 1)
        self.tree.build()

    def sample(self, batch_size, priority_replay=True):
        """

        :return: give the caller all kinds of training sample from the big container

        IMPORTANT!!! the return size is fixed !!! it's bathch size
        (s,a,s',r,mask)

        """
        self.tight()
        if priority_replay == True:
            ids = self.tree.sample(batch_size)
            ids = np.array(ids, dtype=np.int32)

        else:
            ids = np.random.choice(self.length_buffer, size=self.batch_size if batch_size is None else batch_size)

        # prevent outflow
        # next_ids = (ids + 1) % self.length_buffer

        buffer = {}

        buffer['view'] = torch.as_tensor(self.view_buffer.sample(ids),dtype=torch.double)
        buffer['feature'] = torch.as_tensor(self.feature_buffer.sample(ids),dtype=torch.double)
        buffer['adv'] = torch.as_tensor(self.adv_buffer.sample(ids),dtype=torch.double)
        buffer['action'] = torch.as_tensor(self.action_buffer.sample(ids),dtype=torch.double)
        buffer['reward'] = torch.as_tensor(self.reward_buffer.sample(ids),dtype=torch.double)
        buffer['old_policy'] = torch.as_tensor(self.old_policy.sample(ids),dtype=torch.double)
        buffer['old_qvalue'] = torch.as_tensor(self.old_qvalue.sample(ids),dtype=torch.double)


        return buffer

    def how_many_batch(self):
        return self.buffer_length // self.batch_size

    def clear(self):
        self.view_buffer.clear()
        self.feature_buffer.clear()
        self.adv_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.old_qvalue.clear()
        self.old_policy.clear()


        self.buffer_length = 0

        self.agents = dict()

        # print("length after clear buffer: ", self.length_buffer)

    @property
    def length_buffer(self):
        return len(self.view_buffer)


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
# Independent
class Q_network(nn.Module):
    def __init__(self ,obs_dim,action_space):
        super(Q_network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, out_features=int(obs_dim/2)),
            nn.ReLU(),
            nn.Linear(int(obs_dim/2), out_features=int(obs_dim/4)),
            nn.ReLU(),
            nn.Linear(in_features=int(obs_dim/4), out_features=int(obs_dim/16)),
            nn.ReLU(),
            nn.Linear(in_features=int(obs_dim/16),out_features=action_space)
        )


    def forward(self ,obs):
        """
        obs: nbatch,obs_dim
        """
        res = self.net(obs.double())
        return res

class Policy_network(nn.Module):
    def __init__(self ,obs_dim,action_space):
        super(Policy_network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, out_features=int(obs_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(obs_dim / 2), out_features=int(obs_dim / 4)),
            nn.ReLU(),
            nn.Linear(in_features=int(obs_dim / 4), out_features=int(obs_dim / 16)),
            nn.ReLU(),
            nn.Linear(in_features=int(obs_dim / 16), out_features=action_space)
        )
    def forward(self ,obs):
        """
        obs: nbatch,obs_dim
        """
        res = self.net(obs.double())
        log_pi = F.log_softmax(res ,dim=-1)
        return log_pi

class ActorCritic(nn.Module):
    def __init__(self ,view_dim,feature_dim,action_dim):
        super(ActorCritic,self).__init__()
        self.view_dim = shape_calc(view_dim)
        self.feature_dim = shape_calc(feature_dim)
        self.action_dim = shape_calc(action_dim)
        self.input_dim = self.view_dim + self.feature_dim

        self.q = Q_network(self.input_dim,self.action_dim)
        self.pi = Policy_network(self.input_dim,self.action_dim)

    def step(self,view,feature):
        """
        input_latent: nbatch,input_dim
        """
        shape = view.shape
        nbatch = shape[0]
        view = view.view(nbatch, -1)
        feature = feature.view(nbatch, -1)
        input = torch.cat((view, feature), dim=-1)
        with torch.no_grad():
            log_pi = self.pi(input)
            log_pi_chosen = Categorical(logits=log_pi)
            a = log_pi_chosen.sample()
            q = self.q(input)


        return a.numpy().astype(np.int32),log_pi.numpy() ,q.numpy()

    def eval_policy(self, view,feature):
        shape = view.shape
        nbatch = shape[0]
        view = view.view(nbatch, -1)
        feature = feature.view(nbatch, -1)
        input = torch.cat((view, feature), dim=-1)

        log_pi = self.pi(input)
        return log_pi

    def eval_qvalue(self,view,feature):
        shape = view.shape
        nbatch = shape[0]
        view = view.view(nbatch, -1)
        feature = feature.view(nbatch, -1)
        input = torch.cat((view, feature), dim=-1)
        q = self.q(input)
        return q

    def act(self,view,feature):

        return self.step(view,feature)[0]

class Expert_PPO:
    def __init__(self,view_dim,feature_dim,action_dim,learning_rate = 0.0001,batch_size = 2048):


        self.name = "Expert_PPO"
        self.ac = ActorCritic(view_dim,feature_dim,action_dim).double()
        self.action_space = action_dim

        self.pi_optimizer = torch.optim.Adam(self.ac.q.parameters(), lr=learning_rate)
        self.q_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=learning_rate)

        self.buffer = GroupMemory(view_space = view_dim, feature_space=feature_dim, action_space = action_dim[0], max_buffer_size=2**12, batch_size = batch_size, sub__len = 2**11,
                 n_step=3, gamma=0.99)
        self.batch_size = batch_size

    def compute_loss_pi(self, data, clip_ratio=0.2):
        """
        You have to set data as tensor
        policy = logits
        """
        view,feature, action, reward, log_pi, q,adv = data['view'], data['feature'],data['action'], data['reward'], data['old_policy'], data['old_qvalue'],data['adv']

        a_indice = F.one_hot(action.long(), num_classes=int(self.action_space[0]))
        a_indice = a_indice.double()
        # adv = (torch.sum(a_indice * q,dim=1) - torch.sum(q * torch.exp(log_pi), dim=1))
        #  print("ADV",adv)
        # print("test",type(view),type(feature))
        now_log_pi = self.ac.eval_policy(view,feature)
        now_log_pi = torch.sum(a_indice * now_log_pi, dim=1)
        log_pi = torch.sum(a_indice * log_pi, dim=1)

        ratio = torch.exp(now_log_pi - log_pi)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_pi

    def compute_loss_q(self, data):
        """
        You have to set data as tensor
        """
        view,feature, action, reward, log_pi, q = data['view'], data['feature'],data['action'], data['reward'], data['old_policy'], data['old_qvalue']
        q = self.ac.eval_qvalue(view,feature)
        vals  = torch.sum(q * torch.exp(log_pi), dim=1)
        loss_q = torch.mean((reward - vals) ** 2)
        return loss_q

    def update(self):

        data = self.buffer.sample(self.batch_size)
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        self.buffer.clear()

        return loss_q.detach().numpy()

    def act(self,data):
        view = data['view']
        feature = data['feature']
        res = self.ac.act(view,feature).astype(np.int32)

        return res

    def step(self,data):
        view = data['view']
        feature = data['feature']
        view = torch.from_numpy(view)
        feature = torch.from_numpy(feature)
        a, log_pi, q = self.ac.step(view,feature)
        return a, log_pi, q

    def save(self, path, i=1):
        file_path = os.path.join(path, "Expert_{}".format(self.name) + '_%d') % i
        torch.save(self.ac.state_dict(), file_path)
        print("Saved")

    def load(self, path, i=1):
        file_path = os.path.join(path, "Expert_{}".format(self.name) + '_%d') % i
        self.ac.load_state_dict(torch.load(file_path))
        print("Loaded")