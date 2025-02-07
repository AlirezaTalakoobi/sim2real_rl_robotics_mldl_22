"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str,
                        help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False,
                        action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of test episodes')

    return parser.parse_args()


args = parse_args()


def main():

    env = gym.make('CustomHopper-target-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    dict = torch.load(args.model, map_location=args.device)
    print(f'best model was found at episode' + str(dict['n_episode']))
    policy.load_state_dict(dict['model'], strict=False)
    # baseline = Baseline(observation_space_dim)
    agent = Agent(policy, device=args.device)
    average = 0
    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:

            action, _ = agent.get_action(state, evaluation=True)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward
        average += test_reward
        print(f"Episode: {episode} | av_Return: {average/episode+1}")
    print(f"average return is: {average/args.episodes}")


if __name__ == '__main__':
    main()
