# import sys
# sys.path.append('.')
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from utils import moving_average
import argparse


class Policy(nn.Module):
    def __init__(self, n_hidden=128):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, n_hidden)
        # actions_head
        self.action_head = nn.Linear(n_hidden, 2)
        # value head
        self.value_head = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values.squeeze()

class PPO_Agent():
    def __init__(self, policy):
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_update_freq = 30

        self.policy = policy

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.actions.append(action.item())
        self.logprobs.append(m.log_prob(action).item())
        self.states.append(state)

        return action.item()

    "GIVEN rewards array from rollout return the returns with zero mean and unit std"        
    def discount_rewards(self, rewards_arr, dones, gamma, final_value=0):
        R = final_value
        returns = []
        zipped = list(zip(rewards_arr, dones))
        for (r, done) in zipped[::-1]:
            if done:
                R = 0
            R = r + R * gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return (returns - returns.mean()) / (returns.std() + self.eps)

    def get_experience(self, final_obs=None, done=True):
        state = torch.from_numpy(final_obs).float()
        _, state_value = self.policy(state)
        final_value = state_value.detach() if not done else 0.0

        # rewards
        returns = self.discount_rewards(self.rewards, \
                                        self.dones, self.gamma, final_value)
        
        states = torch.stack(self.states).float()
        old_actions = self.actions
        old_logprobs = torch.tensor(self.logprobs).float()
        # print('collected experience')
        return states, torch.tensor(old_actions), old_logprobs, returns

    def clear_experience(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        dist.barrier()

class PPO_Centralized_Trainer():
    def __init__(self, policy_shared, policy):
        self.shared_policy = shared_policy
        self.policy = policy
        self.clip_val = 0.1
        self.c2 = 0.001
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.n_epochs = 3

    def train(self, states, old_actions, old_logprobs, returns):

        # print('processing experience')
        for i in range(self.n_epochs): 
            p, v = policy.forward(states)
            m = Categorical(p)
            c = m.log_prob(old_actions)
            entr = m.entropy()

            # value fn loss
            loss_vf = F.mse_loss(v, returns)

            # surrogate loss
            advantage = returns - v.detach()
            r_ts = torch.exp(c - old_logprobs)
            loss_surr = - (torch.min(r_ts * advantage, \
                torch.clamp(r_ts, 1 - self.clip_val, 1 + self.clip_val) * advantage)).mean()
            
            # maximize entropy bonus
            loss_entropy = - self.c2 * entr.mean()

            # the total_loss
            loss_total = loss_vf + loss_surr + loss_entropy
            
            # step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

        self.shared_policy.load_state_dict(self.policy.state_dict())

        dist.barrier()

def run(shared_policy, policy, rank, size):
    N_eps = 300
    ep_steps = 500
    group = dist.new_group([i for i in range(size)])
    batch_update_freq = 30
    if rank != 0:
        if rank == 1: rewards = []
        env = gym.make('CartPole-v0')
        env.seed(rank); torch.manual_seed(rank)
        env._max_episode_steps = ep_steps
        T = 0
        agent = PPO_Agent(shared_policy)

        for i_episode in range(N_eps if rank == 1 else 2 * N_eps):
            observation = env.reset()
            if rank == 1: total_r = 0
            for t in range(100000):
                T += 1
                action = agent.select_action(observation)
                observation, reward, done, info = env.step(action)

                agent.rewards.append(reward)
                agent.dones.append(done)

                if rank == 1: total_r += reward

                if T % batch_update_freq == 0:
                    dist.gather
                    a, b, c, d = agent.get_experience(observation, done)
                    dist.gather(a, gather_list=[], dst=0, group=group)
                    dist.gather(b, gather_list=[], dst=0, group=group)
                    dist.gather(c, gather_list=[], dst=0, group=group)
                    dist.gather(d, gather_list=[], dst=0, group=group)
                    
                    agent.clear_experience()
                if done:
                    # print(f"rank: {rank}, episode: {i_episode}")
                    if (i_episode + 1) % 100 == 0:
                        if rank == 1: print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                    break
            if rank == 1: rewards.append(total_r)
        # exit since thread 1 ran for 300 episodes
        print("Finished running env", rank, "after", N_eps, "episodes.")
        env.close()
    else:
        trainer = PPO_Centralized_Trainer(shared_policy, policy)
        old_states = [torch.zeros((30, 4), dtype=torch.float32) for i in range(size)]
        old_actions = [torch.zeros((30), dtype=torch.int64) for i in range(size)]
        old_logprobs = [torch.zeros((30), dtype=torch.float32) for i in range(size)]
        old_returns = [torch.zeros((30), dtype=torch.float32) for i in range(size)]

        while(True): 
            dist.gather(old_states[0], gather_list=old_states, dst=0, group=group)
            dist.gather(old_actions[0], gather_list=old_actions, dst=0, group=group)
            dist.gather(old_logprobs[0], gather_list=old_logprobs, dst=0, group=group)
            dist.gather(old_returns[0], gather_list=old_returns, dst=0, group=group)
            states = torch.cat(old_states[1:])
            actions = torch.cat(old_actions[1:])
            logprobs = torch.cat(old_logprobs[1:])
            returns = torch.cat(old_returns[1:])
            trainer.train(states, actions, logprobs, returns)

def init_process(shared_policy, policy, rank, size, fn, port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = port
    os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(shared_policy, policy, rank, size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--port', default='3000', type=str, required=True)
    args = parser.parse_args()
    num_agents = 2
    # one thread 0 is reserved for centralized learner
    shared_policy = Policy()
    shared_policy.share_memory()
    policy = Policy()
    policy.load_state_dict(shared_policy.state_dict())

    size = num_agents + 1
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(shared_policy, policy, rank, size, run, args.port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()