import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from python import magent
from algo.DSN_test.Attention import Attention_Module
from algo.DSN_test.Expertise import ActorCritic
from algo.DSN_test.Encoder import SharedEncoder,Decoder
import scipy.signal

def discount_cumsum(x,discount):
    return scipy.signal.lfilter([1],[1,float(-discount)],x[::-1],axis=0)[::-1]


gamma = 0.99
lam = 0.95

rew = np.arange(50)
val = np.arange(50)

delta = rew[:-1] + gamma * rew[1:] - val[:-1]
adv = discount_cumsum(delta,gamma*lam)
rew_to_go = discount_cumsum(rew,gamma)[:-1]
print(rew_to_go)



"""

def battle(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})

    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 5, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(1.5),
         'damage': 2, 'step_recover': 0.1,

         'step_reward': -0.01, 'kill_reward': 3, 'dead_penalty': -0.1, 'attack_penalty': -0.05,
         })

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')
    b = gw.AgentSymbol(g1, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=1)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=b, value=1)
    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=b, value=-0.5)
    cfg.add_reward_rule(gw.Event(b, 'attack', a), receiver=a, value=-0.5)

    return cfg

env = magent.GridWorld(battle(map_size=10))
handles = env.get_handles()
view_dim = env.get_view_space(handles[0])
feature_dim = env.get_feature_space(handles[0])
action_dim = env.get_action_space(handles[0])
print("View space: ", env.get_view_space(handles[0]))
print("Feature space: ", env.get_feature_space(handles[0]))
print("Action space: ", env.get_action_space(handles[0]))
env.reset()
env.add_agents(handles[0],method = 'random',n = 3)
env.add_agents(handles[1],method='random',n=3)
obs1 = env.get_observation(handles[0])
obs2 = env.get_observation(handles[1])

view,feature = obs1

attn = Attention_Module(view_dim=view_dim,feature_dim=feature_dim,output_dim=32,max_agents=5,h=1)

#lat = attn(torch.from_numpy(view),torch.from_numpy(feature))

ac = ActorCritic(view_dim=view_dim,feature_dim=feature_dim,action_dim=action_dim).double()
encoder = SharedEncoder(max_agents=5,input_dim=32,latent_dim=16,action_dim=action_dim).double()
decoder = Decoder(view_dim=view_dim,feature_dim=feature_dim,latent_dim=16).double()

view = np.array(view)
feature = np.array(feature)
view = torch.from_numpy(view)
feature = torch.from_numpy(feature)

a,log_pi,q = ac.step(view,feature)
log_pi_1 = ac.eval_policy(view,feature)
q_1 = ac.eval_qvalue(view,feature)

action1 = env.set_action(handles[0],a)
action2 = env.set_action(handles[0],a)
reward1 = env.get_reward(handles[0])
reward2 = env.get_reward(handles[1])

reward1 = np.array([reward1]).reshape(1,3,1)
reward1 = torch.from_numpy(reward1)

last_lat = np.random.rand(1,16).reshape(1,16)
last_lat = torch.from_numpy(last_lat).double()

view = view.view(1,3,-1)
feature = feature.view(1,3,-1)

x = attn(view,feature)

latent = encoder(x,last_lat,log_pi_1.view(1,3,21),reward1)


recons_obs = decoder(latent,latent,view,feature)

"""

