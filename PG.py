import gym
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torch.autograd import Variable
from utils import moving_average

N_EPS = 500

class Policy_PG(nn.Module):
    def __init__(self):
        super(Policy_PG, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        return F.softmax(action_scores, dim=-1)

SavedAction = namedtuple('SavedAction', ['log_prob'])

policy = Policy_PG()
optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(m.log_prob(action))
    return action.item()

"GIVEN rewards array from rollout return the returns with zero mean and unit std"        
def discount_rewards_on_rollout(rewards_arr, gamma):
    R = 0
    returns = []
    for r in rewards_arr[::-1]:
        R = r + R * gamma
        returns.insert(0, R)
#     print('rewards_arr', rewards_arr)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + eps)

def train_on_rollout(gamma=0.99):
    returns = discount_rewards_on_rollout(policy.rewards, gamma)
    actor_loss = []
    for log_prob, r in zip(policy.saved_actions, returns):
        actor_loss.append(-log_prob * r)
    optimizer.zero_grad()
    loss = torch.stack(actor_loss).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

def TRAIN_PG(N_eps=500, max_ep_steps=500): 
    df = 0.99
    rewards = []
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_ep_steps
    for i_episode in range(N_eps):
        observation = env.reset()
        total_r = 0
        for t in range(100000):
            action = select_action(observation)
            observation, reward, done, info = env.step(action)
            policy.rewards.append(reward)
            total_r += reward
            if done:
                train_on_rollout(df)
                if (i_episode + 1) % 100 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
        rewards.append(total_r)
    env.close()
    return rewards

rewards_PG = TRAIN_PG(N_EPS, 500)
plt.plot(moving_average(rewards_PG, 100), label="PG")
plt.legend()
plt.show()
