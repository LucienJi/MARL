from python import magent
from algo.DSN_test.Expertise import Expert_PPO
from algo.DSN_test.DSN import DSN
import numpy as np
import math
import random as rd

def generate_map(env, map_size, handles, random=False):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 2

    leftID = rd.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 - gap - side, width // 2 - gap - side + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    if random is not False:
        env.add_agents(handles[leftID], n=len(pos), method="random")
    else:
        env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width // 2 + gap, width // 2 + gap + side, 2):
        for y in range((height - side) // 2, (height - side) // 2 + side, 2):
            pos.append([x, y, 0])
    if random is not False:
        env.add_agents(handles[rightID], n=len(pos), method="random")
    else:
        env.add_agents(handles[rightID], method="custom", pos=pos)

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


class Run:
    def __init__(self,env:magent.GridWorld,n_agents,map_size,max_steps,handles,models,print_every,render = False,train = True,DSN = None):

        self.env = env
        self.map_size = map_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.handles = handles
        self.models = models
        self.print_every = print_every
        self.render = render
        self.render_every = 100
        self.train = train
        self.DSN = DSN
    
    def play(self,round):
        self.env.reset()
        self.env.add_agents(self.handles[0],method = 'random',n = self.n_agents)
        self.env.add_agents(self.handles[1],method='random',n=self.n_agents)

        n_groups = len(self.handles)
        rewards = [[] for _ in range(len(self.handles))]
        nums = [self.env.get_num(handle) for handle in self.handles]
        max_nums = nums.copy()
        print("\n[*] ROUND #{0}, NUMBER: {1}".format(round, nums))
        mean_rewards = [[] for _ in range(n_groups)]
        total_rewards = [[] for _ in range(n_groups)]


        step_ct = 0
        done = False

        while not done and step_ct<self.max_steps:
            data = [{},{}]
            for i in range(n_groups):
                obs = self.env.get_observation(self.handles[i])
                data[i]['view'] = obs[0]
                data[i]['feature'] = obs[1]
                data[i]['id'] = self.env.get_agent_id(self.handles[i])
                data[i]['action'],data[i]['old_policy'],data[i]['old_qvalue'] = self.models[i].step(data[i])


            #print("Test Policy size",np.shape(data[0]['old_policy']))
            for i in range(n_groups):
                self.env.set_action(self.handles[i],data[i]['action'])
            done = self.env.step()

            for i in range(n_groups):
                rewards[i] = self.env.get_reward(self.handles[i])
                data[i]['reward'] = rewards[i]
                self.models[i].buffer.push(**data[i])
            if self.DSN is not None:
                self.DSN.push(data[0])


            for i in range(n_groups):
                sum_reward = sum(rewards[i])
                rewards[i] = sum_reward / max_nums[i]
                mean_rewards[i].append(rewards[i])
                total_rewards[i].append(sum_reward)

            
            if self.render:
                if round % self.render_every ==0:
                    self.env.render()
            # clear dead agents
            self.env.clear_dead()

            info = {"Total-Reward": np.round([sum(total_rewards[0]), sum(total_rewards[1])], decimals=6), "NUM": nums}

            step_ct += 1

            if step_ct % self.print_every == 0:
                print("> step #{}, info: {}".format(step_ct, info))
        for i in range(n_groups):
            mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
            total_rewards[i] = sum(total_rewards[i])

        play_result = {'Train_Reward(per step)': mean_rewards[0], 'Train_Reward(total)': total_rewards[0]}

        if self.train:
            q_loss = self.models[0].update()
            if self.DSN is not None:
                loss = self.DSN.update()

            print("Q_Loss: {0}".format(q_loss))
            with open('multibattle_test.csv','a+') as myfile:
                myfile.write('{0},{1},{2},{3}\n'.format(round,mean_rewards[0],total_rewards[0],q_loss))


if __name__ == '__main__':
    map_size = 10
    n_agents = 10
    n_rounds = 2000
    env = magent.GridWorld(battle(map_size = map_size))

    env.set_render_dir("build/render")
    handles = env.get_handles()
    save_dir = 'data/test_model'

    view_dim = env.get_view_space(handles[0])
    feature_dim = env.get_feature_space(handles[0])
    action_dim = env.get_action_space(handles[0])
    print("View space: ", view_dim)
    print("Feature space: ", feature_dim)
    print("Action space: ", action_dim)

    n = len(handles)
    models = []
    #DSN = DSN(view_dim=view_dim,feature_dim=feature_dim,action_dim=action_dim,latent_dim1=64,latent_dim2=32,max_agents=10,learning_rate=0.0001)
    for i in range(n):
        models.append(Expert_PPO(view_dim=view_dim,feature_dim=feature_dim,action_dim=action_dim,learning_rate=0.0001,batch_size=1024))

    runhandle = Run(env,map_size = map_size,n_agents = n_agents,max_steps=300,handles = handles,models = models,print_every = 100,render= True,train = True,DSN = None)

    start_from = 1


    with open('multibattle_test.csv', 'a+') as myfile:
        myfile.write('{0},{1},{2},{3}\n'.format("Episode", "Test Reward per step", "Test Total Reward","Q Loss"))

    for i in range(start_from,start_from + n_rounds):
        runhandle.play(i)

        if i%100 == 0:
            models[0].save(save_dir,i)






            
