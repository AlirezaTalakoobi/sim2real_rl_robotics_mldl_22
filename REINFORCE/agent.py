# from email import policy
from multiprocessing.managers import ValueProxy
from pickletools import optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        return normal_dist
class Agent(object):
    def __init__(self, policy, device='cpu',lr = 1e-3):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.policy.init_weights()
        self.optimizer = torch.optim.Adam(policy.parameters(), lr = lr)
        self.gamma = 0.98
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        total_rewards = torch.zeros_like(rewards)
        done = torch.Tensor(self.done).to(self.train_device)
        policy_loss = []
        numberOfSteps = len(done)
 
        for timestep in reversed(range(numberOfSteps)):
            discounted_reward = rewards[timestep]
            if timestep + 1 < numberOfSteps:
                discounted_reward += self.gamma * total_rewards[timestep + 1]
            total_rewards[timestep] = discounted_reward
        total_rewards = (total_rewards - total_rewards.mean())/total_rewards.std()

        for i in range(len(action_log_probs)):
            policy_loss.append(-1 * action_log_probs[i] * total_rewards[i])

        self.optimizer.zero_grad()

        policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        self.action_log_probs,self.states,self.next_states,self.rewards,self.done = [],[],[],[],[]


    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist = self.policy.forward(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)          

    def reset_outcomes(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []