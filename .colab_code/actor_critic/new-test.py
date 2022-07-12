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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="Model path")
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )
    parser.add_argument(
        "--render", default=False, action="store_true", help="Render the simulator"
    )
    parser.add_argument(
        "--episodes", default=100, type=int, help="Number of test episodes"
    )

    return parser.parse_args()


args = parse_args()


episodes = args.episodes
env = gym.make("CustomHopper-target-v0")


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
dict = torch.load(args.model)
model.load_state_dict(dict['model'], strict=False)

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



def main():
    reward_log = []
    best_reward = -1000
    total_test_reward = 0

    # run inifinitely many episodes
    for i_episode in range(episodes):

        # reset environment and episode reward
        test_reward = 0
        state = env.reset()

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        while True:

            # select action from policy
            
            action,model = select_action(state)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            test_reward += reward
            if done:
                with open("results2.txt", "a") as f:  
                    f.write(f"ACTOR_CRITIC,{test_reward}\n")
                break
        total_test_reward +=test_reward
        print(f"Episode: {i_episode} | Return: {total_test_reward/(i_episode+1)}")
    print(f"final average reward was:{total_test_reward/episodes}")


if __name__ == '__main__':
    main()