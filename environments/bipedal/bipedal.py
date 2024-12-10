import torch
from torch import nn
import gym
from torch.distributions import Normal
import numpy as np
from data_lodder.logger import Logger

class BipedalWalker(gym.ActionWrapper):
    def __init__(self) -> None:
        env = gym.make("BipedalWalker-v3")
        super().__init__(env)
        self.logger = Logger("logs/bipedal_episode_reward.csv")
        self.n_episodes = 0
        self.total_reward = 0

    def action(self, action):
        return action * 2 - 1
    
    def reset(self, **kwargs):
        if self.n_episodes > 0:
            self.logger.log("Episode", self.n_episodes)
            self.logger.log("Reward", self.total_reward)
            self.logger.write()
            self.logger.print("Episode Complete")
        
        self.n_episodes += 1
        self.total_reward = 0
        
        observation, info = super().reset(**kwargs)
        return observation
    
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.total_reward += reward
        return observation, reward, terminated or truncated, info

    def close(self):
        self.logger.close()
        return super().close()

class BipedalNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        self.backbone = nn.Sequential(nn.Linear(state_dim[0], 256), nn.ReLU(),)

        self.actor_fc = nn.Linear(256, 256)
        self.alpha_head = nn.Sequential(nn.Linear(256, action_dim[0]), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, action_dim[0]), nn.Softplus())

        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.backbone(x)
        value = self.critic(x)
        x = self.actor_fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return value, alpha, beta

class SACBipedalNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim[0]

        self.actor = nn.Sequential(
            nn.Linear(state_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(256, action_dim[0])
        self.log_std = nn.Linear(256, action_dim[0])

        self.q1 = nn.Sequential(
            nn.Linear(state_dim[0] + action_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim[0] + action_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        x = self.actor(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std, state

    def get_q_values(self, state, action, features=None):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def sample_action(self, state):
        mean, log_std, features = self(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, features