import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from env.custom_hopper import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument(
        "--n-episodes", default=100000, type=int, help="Number of training episodes"
    )
parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )
parser.add_argument(
    "--print-every", default=1000, type=int, help="Print info every <> episodes"
)   
args = parser.parse_args()


lr = args.lr
episodes = args.n_episodes
env = gym.make("CustomHopper-source-v0")
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self,observation_space_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(observation_space_dim, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values

observation_space_dim = env.observation_space.shape[-1]
model = Policy(observation_space_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item(),model


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    reward_log = []
    best_reward = -1000

    # run inifinitely many episodes
    for i_episode in range(episodes):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        while True:

            # select action from policy
            
            action,model = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        reward_log.append(ep_reward)
        if(ep_reward > best_reward):
            best_reward = ep_reward
            checkpoint = {
                'model': model.state_dict(),
                'n_episode' : i_episode + 1
            }
            if(not os.path.exists('./models')):
                os.makedirs('./models')
            torch.save(checkpoint, f"./models/best_lr_{lr}_nEpisodes_{args.n_episodes}.pt")
        if (i_episode + 1) % args.print_every == 0:
            print("Training episode:", i_episode)
            print("Episode return:", ep_reward)


    if(not os.path.exists('./train_history')):
        os.makedirs('./train_history')
    with open(f"./train_history/lr_{lr}_nEpisodes_{args.n_episodes}.txt", "w+") as f:
        f.write(str(reward_log))

    torch.save(model.state_dict(), "./models/last_model.mdl")

        


if __name__ == '__main__':
    main()