"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""
import matplotlib.pyplot as plt
from unittest.mock import Base
import torch
import gym
import argparse
import numpy as np
from env.custom_hopper import *
import os
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-episodes", default=100000, type=int, help="Number of training episodes"
    )
    parser.add_argument(
        "--print-every", default=1000, type=int, help="Print info every <> episodes"
    )
    parser.add_argument(
        "--device", default="cpu", type=str, help="network device [cpu, cuda]"
    )

    parser.add_argument(
        "--lr", default=1e-3, type=float, help="learning rate"
    )

    return parser.parse_args()


args = parse_args()


def main():

    # torch.cuda.set_per_process_memory_fraction(0.7, 0)

    env = gym.make("CustomHopper-source-v0")
    # env = gym.make('CustomHopper-target-v0')

    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())

    """
		Training
	"""
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    # baseline = Baseline(observation_space_dim)
    best_reward = -1000
    lr = args.lr
    agent = Agent(policy, device=args.device, lr = lr)
    reward_log = []

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over

            action, action_probabilities = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(
                previous_state, state, action_probabilities, reward, done
            )

            train_reward += reward
        reward_log.append(train_reward)
        if(train_reward > best_reward):
            best_reward = train_reward
            checkpoint = {
                'model': agent.policy.state_dict(),
                'n_episode' : episode + 1
            }
            if(not os.path.exists('./models')):
                os.makedirs('./models')
            torch.save(checkpoint, f"./models/best_lr_{lr}_nEpisodes_{args.n_episodes}.pt")

        if (episode + 1) % args.print_every == 0:
            print("Training episode:", episode)
            print("Episode return:", train_reward)
        

        agent.update_policy()
        agent.reset_outcomes()

    if(not os.path.exists('./train_history')):
        os.makedirs('./train_history')
    with open(f"./train_history/lr_{lr}_nEpisodes_{args.n_episodes}.txt", "w+") as f:
        f.write(str(reward_log))


    # plt.plot(reward_log,'-')
    # plt.xlabel('reward')
    # plt.title('reward')
    # plt.savefig("plot.png")

    torch.save(agent.policy.state_dict(), "./models/last_model.mdl")

    
if __name__ == "__main__":
    main()
