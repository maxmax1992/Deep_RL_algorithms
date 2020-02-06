import gym
from collections import namedtuple
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd import Variable
from utils import LinearSchedule, moving_average

class Deep_Q_network(nn.Module):
    def __init__(self):
        super(Deep_Q_network, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
#         print(x.shape)
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        return action_scores
# DUELING
class Deep_Q_network_dueling(nn.Module):
    def __init__(self):
        super(Deep_Q_network_dueling, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        # self.action_head = nn.Linear(128, 2)
        
        self.value_head = nn.Linear(128, 128)
        self.value_end = nn.Linear(128, 1)

        self.advantages_head = nn.Linear(128, 128)
        self.advantages_end = nn.Linear(128, 2)
        
    def forward(self, x):
        x = F.relu(self.affine1(x))
        value = F.relu(self.value_head(x))
        value = self.value_end(value)
        advantages = F.relu(self.advantages_head(x))
        advantages = self.advantages_end(advantages)
        advantages = advantages - torch.mean(advantages)
        Q_values = value + advantages
        return Q_values

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

Q_network = Deep_Q_network()
Q_network_target = Deep_Q_network()
Q_network_target.load_state_dict(Q_network.state_dict())
Q_network_target.eval()
optimizer = optim.RMSprop(Q_network.parameters(), lr=3e-3)


def select_greedy(obs):
    with torch.no_grad():
        obs_ = torch.from_numpy(obs).float()
        values = Q_network(obs_)
        return torch.argmax(values.detach()).view(1, -1)
    
def train_on_batch(memory, batch_size, df):
    # TODO-in future: remove the casting to tensors all the time
    # Vectorized implementation
    batch = memory.sample(batch_size)
    # connect all batch Transitions to one tuple
    batch_n = Transition(*zip(*batch))
    # reshape actions so ve can collect the DQN(S_t, a_t) easily with gather
    actions = torch.cat(batch_n.action)
    # get batch states
#     print(batch_n.state)
    batch_states = torch.cat(batch_n.state).float()
    # 
    batch_rewards = torch.cat(batch_n.reward).float()
    # collect only needed Q-values with corresponding actions for loss computation
    inputs = Q_network(batch_states).gather(1, actions)
    targets = batch_rewards
    non_final_next_mask = [s is not None for s in batch_n.next_state]
    next_states = torch.tensor([s for s in batch_n.next_state if s is not None]).float()
    targets[non_final_next_mask] += df * Q_network_target(next_states).max(1)[0].detach()
    # update parameters here
    optimizer.zero_grad()
    loss = F.smooth_l1_loss(inputs, targets.view(inputs.shape))
    loss.backward()
    for param in Q_network.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
def learn_episodic_DQN(N_eps=500, max_ep_steps=500, use_dueling=False): 
    
    global Q_network
    global Q_network_target
    global optimizer
    
    if use_dueling:
        Q_network = Deep_Q_network_dueling()
        Q_network_target = Deep_Q_network_dueling()
    else:
        Q_network = Deep_Q_network()
        Q_network_target = Deep_Q_network()

    Q_network_target.load_state_dict(Q_network.state_dict())
    Q_network_target.eval()
    optimizer = optim.RMSprop(Q_network.parameters(), lr=3e-3)
    
    memory_len = 5000
    df = 0.99
    e_s = 1.0
    e_e = 0.05
    N_decay = 12000
    batch_size = 64
    train_freq = 3
    T = 0
    target_update_freq = 150

    scheduler = LinearSchedule(N_decay, e_e, e_s)
    memory = ReplayMemory(memory_len)
    rewards = []

    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_ep_steps
    n_actions = env.action_space.n
    
    for i_episode in range(N_eps):

        observation = env.reset()
        total_r = 0

        for t in range(100000):

            action = None
            curr_epsilon = scheduler.value(T)
            T += 1
            # select action based on e-greedy
            if random.random() < curr_epsilon:
                # random action
                # action = env.action_space.sample()
                action = torch.tensor([[random.randrange(n_actions)]])
            else:
                # argmax Q(s_t)
                action = select_greedy(observation)
            
            next_observation, reward, done, info = env.step(action.item())
            total_r += reward
            reward = torch.tensor([reward])
            
            # set next_state = None if finished ep
            if done:
                next_observation = None
            # add transition to the Memory
            memory.push(torch.from_numpy(observation).view(1, -1), action, reward, next_observation)
            
            # train the DQN
            if T % train_freq == 0:
                train_on_batch(memory, min(batch_size, T), df)
            
            observation = next_observation
            
            if T % target_update_freq == 0:
                Q_network_target.load_state_dict(Q_network.state_dict())
                
            if done:
                if (i_episode + 1) % 100 == 0:
                    print("Episode {} finished after {} timesteps, T: {}".format(i_episode, t + 1, T))
                break
                
        rewards.append(total_r)
    env.close()
    
    return rewards

N_EPS = 500
# rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
rewards_DQN = learn_episodic_DQN(N_EPS, 500)

plt.plot(moving_average(rewards_DQN, 100), label="DQN")
plt.legend()
plt.show()

