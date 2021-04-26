import torch
import torch.nn as nn
import torch.nn.functional as F
from python import magent 
from torch.distributions.categorical import Categorical
import numpy as np

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
        self.old_policy = MetaBuffer(shape=action_space, max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=action_space, max_len=max_buffer_size)


    def append(self, view, feature, action, reward, old_policy, old_qvalue):
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
        }

        return res

    def clear(self):
        self.view_buffer.clear()
        self.feature_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.old_qvalue.clear()
        self.old_policy.clear()
        

    def reshape(self):
        if self.n != -1:
            gamma = self.gamma
            n = self.n
            N = len(self.reward_buffer)
            reward = self.reward_buffer.pull()
            assert len(reward) == N
            action = self.action_buffer.pull()
            action_indice = np.eye(self.action_space[0])[action]

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
        self.reward_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)
        self.action_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size,dtype = 'int16')
        self.old_policy = MetaBuffer(shape=action_space, max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=action_space, max_len=max_buffer_size)
  

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
                                    old_qvalue=kwargs['old_qvalue'][i]
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
        action_indice = np.eye(self.action_space[0])[action]

        # old_policy = self.old_policy.pull()
        old_qvalue = self.old_qvalue.pull()
        value = np.sum(action_indice * old_qvalue, axis=-1)

        self.tree = SumTree(self.reward_buffer.pull(), value, 1)
        self.tree.build()

    def sample(self, batch_size=None, priority_replay=True):
        """

        :return: give the caller all kinds of training sample from the big container

        IMPORTANT!!! the return size is fixed !!! it's bathch size
        (s,a,s',r,mask)

        """
        if priority_replay == True:
            ids = self.tree.sample(batch_size)
            ids = np.array(ids, dtype=np.int32)

        else:
            ids = np.random.choice(self.length_buffer, size=self.batch_size if batch_size is None else batch_size)

        # prevent outflow
        # next_ids = (ids + 1) % self.length_buffer

        buffer = {}

        buffer['view'] = self.view_buffer.sample(ids)
        buffer['feature'] = self.feature_buffer.sample(ids)
        buffer['action'] = self.action_buffer.sample(ids)
        buffer['reward'] = self.reward_buffer.sample(ids)
        buffer['old_policy'] = self.old_policy.sample(ids)
        buffer['old_qvalue'] = self.old_qvalue.sample(ids)
            

        return buffer

    def how_many_batch(self):
        return self.buffer_length // self.batch_size

    def clear(self):
        self.view_buffer.clear()
        self.feature_buffer.clear()
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

class Q_network(nn.Module):
    def __init__(self,view_space,feature_space,action_space):
        super(Q_network, self).__init__()
        self.flatten = nn.Flatten()
        v_space = 1
        for i in view_space:
            v_space = v_space*i
        f_space = 1
        for i in feature_space:
            f_space = f_space*i
        self.linear1 = nn.Linear(v_space,out_features = 64)
        self.linear2 = nn.Linear(f_space,out_features = 64)

        self.linear3 = nn.Linear(in_features=64*2,out_features=64)
        self.linear4 = nn.Linear(in_features=64,out_features=32)
        self.final_linear = nn.Linear(in_features = 32,out_features = action_space)

    
    def forward(self,view,feature):
        x = self.flatten(view)
        y = self.flatten(feature)
        x = self.linear1(x)
        y = self.linear2(y)

        h = torch.cat((x,y),1)
        h = F.relu(h)
        h = F.relu(self.linear3(h))
        h = F.relu(self.linear4(h))
        res = self.final_linear(h)
        return res

class Policy_network(nn.Module):
    def __init__(self,view_space,feature_space,action_space):
        super(Policy_network,self).__init__()
        self.flatten = nn.Flatten()
        v_space = 1
        for i in view_space:
            v_space = v_space*i
        f_space = 1
        for i in feature_space:
            f_space = f_space*i
        self.linear1 = nn.Linear(v_space,out_features = 64)
        self.linear2 = nn.Linear(f_space,out_features = 64)

        self.linear3 = nn.Linear(in_features=64*2,out_features=64)
        self.linear4 = nn.Linear(in_features=64,out_features=32)
        self.final_linear = nn.Linear(in_features = 32,out_features = action_space)
    def forward(self,view,feature):
        
        x = self.flatten(view)
        y = self.flatten(feature)
        x = self.linear1(x)
        y = self.linear2(y)
        h = torch.cat((x,y),1)
        h = F.relu(h)
        h = F.relu(self.linear3(h))
        h = F.relu(self.linear4(h))
        res = self.final_linear(h)
        pi = F.log_softmax(res,dim=1)
        return pi

class ActorCritic(nn.Module):
    def __init__(self,view_space,feature_space,action_space):
        super(ActorCritic,self).__init__()
        self.q = Q_network(view_space,feature_space,action_space)
        self.pi = Policy_network(view_space,feature_space,action_space)
    
    def step(self,view,feaure):
        with torch.no_grad():
            log_pi = self.pi(view,feaure)
            pi = torch.exp(log_pi)
            log_pi = Categorical(logits=log_pi)
            a = log_pi.sample()
            q = self.q(view,feaure)
        return a.numpy(),pi.numpy(),q.numpy()
    
    def act(self,view,feature):
        
        return self.step(view,feature)[0]

    
        


class Agent:
    def __init__(self,env: magent.gridworld.GridWorld, handle, batch_size=1024,learning_rate=1e-4, discount_coef=0.9):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = discount_coef
        self.batch_size = batch_size
        """ Basic Environment Info """
        self.view_space = env.get_view_space(handle)
        self.feature_space = env.get_feature_space(handle)
        self.action_space = env.get_action_space(handle)[0]

        """ Network """
        self.replaybuffer = GroupMemory(view_space = self.view_space,feature_space=self.feature_space,action_space=env.get_action_space(handle),max_buffer_size=2**10,batch_size=batch_size,sub__len=2**9)
        self.agent = ActorCritic(self.view_space,self.feature_space,self.action_space)
        self.pi_optimizer = torch.optim.Adam(self.agent.pi.parameters(),lr=learning_rate)
        self.q_optimizer  = torch.optim.Adam(self.agent.q.parameters(),lr=learning_rate)

    
    def compute_loss_pi(self,data):
        view,feature,action,reward,pi,q = data['view'],data['feature'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward)
        q = torch.as_tensor(q)
        view = torch.as_tensor(view)
        feature = torch.as_tensor(feature)
        a_indice = F.one_hot(action.long(),num_classes=21)
        adv = (reward - torch.mean(q,dim=1))
        log_pi = torch.log(self.agent.pi(view,feature))
        log_pi = torch.sum(a_indice * log_pi,dim=1)
        loss_pi = -(torch.mean(adv * log_pi))
        return loss_pi


    def compute_loss_q(self,data):
        view,feature,action,reward,pi,q = data['view'],data['feature'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']
        reward = torch.as_tensor(reward)
        pi = torch.as_tensor(pi)
        q = self.agent.q(torch.as_tensor(view),torch.as_tensor(feature))
        q_value = torch.sum(q*pi,dim=1)
        loss_q = torch.mean((reward - q_value)**2)
        return loss_q

    def update(self):
        self.replaybuffer.tight()
        data = self.replaybuffer.sample(self.batch_size)
        pi_loss_old = self.compute_loss_pi(data)
        q_loss_old = self.compute_loss_q(data)
        train_iteration = 10
        for i in range(train_iteration):
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
        for i in range(train_iteration):
            self.q_optimizer.zero_grad()
            loss_q = self.compute_loss_q(data)
            loss_q.backward()
            self.q_optimizer.step()
        return pi_loss_old,q_loss_old

    def act(self,data):
        view = data['view']
        feature = data['feature']
        res = self.agent.act(torch.as_tensor(view),torch.as_tensor(feature)).astype(np.int32)
        
        return res
    def step(self,data):
        view = data['view']
        feature = data['feature']
        a,pi,q = self.agent.step(torch.as_tensor(view),torch.as_tensor(feature))
        return pi,q

    def save(self,path):
        torch.save(self.agent.state_dict(),path)
        print("Saved")
        
    def load(self,path):
        self.agent.load_state_dict(torch.load(path))
        print("Loaded")
    


    











