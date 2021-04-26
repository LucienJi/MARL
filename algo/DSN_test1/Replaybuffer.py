import numpy as np
import torch

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
        self.old_policy = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=(action_space,), max_len=max_buffer_size)

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
    def __len__(self):
        return len(self.reward_buffer)

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



    def sample(self,_id,k = 1):

        agent_buffer = self.agents[_id]
        agent_buffer.reshape()
        data = agent_buffer.pull()
        n = len(agent_buffer)
        source_data = {}
        target_data = {}

        for key in data:
            source_data[key] = torch.from_numpy(data[key][0:n-k])
            target_data[key] = torch.from_numpy(data[key][k:n])



        return source_data,target_data

    def how_many_batch(self):
        return self.buffer_length // self.batch_size

    def clear(self):
       self.agents = dict()

