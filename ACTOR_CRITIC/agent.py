import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
from torch.distributions import Normal

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)

        """
            Critic network
        """
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_estimate = torch.nn.Linear(self.hidden, 1)
    
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        estimate = self.fc3_estimate(x_critic)

        return normal_dist, estimate

class Agent(object):
    def __init__(self, policy, device='cpu', lr=1e-3):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        
        self.policy = policy
       
        self.policy.init_weights()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = 0.99
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
        done = torch.Tensor(self.done).to(self.train_device)

        numberOfSteps = len(states)
        advantages = []
        current_state_values = []

        total_rewards = []
        D1 = 0
        for r in range(len(rewards)-1, -1, -1): 
            D1 = r + self.gamma * D1
            total_rewards.insert(0, D1)
        
        for timestep in range(len(states)):
            _, current_value = self.policy(states[timestep]) 
            _, onestep_value = self.policy(next_states[timestep])

            current_state_values.append(current_value)
            ad = rewards[timestep] + self.gamma * onestep_value - current_value
            advantages.append((ad).detach())

        advantages = torch.stack(advantages, dim = 0)

        actor_loss = []
        for i in range(len(action_log_probs)):
            actor_loss.append(action_log_probs[i] * advantages[i])

        actor_loss = torch.stack(actor_loss).sum()

        batch_loss = torch.nn.MSELoss()

        total_rewards = torch.tensor(total_rewards)
        current_state_values = torch.stack(current_state_values, dim = 0).squeeze(-1)
        
        critic_loss = batch_loss(current_state_values, total_rewards)

        loss = actor_loss + critic_loss
        loss = loss*(-1)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.action_log_probs,self.states,self.next_states,self.rewards,self.done = [],[],[],[],[]

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)
        normal_dist, _ = self.policy(x)

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