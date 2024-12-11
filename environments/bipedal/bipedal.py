import torch
from torch import nn
import gym
from torch.distributions import Normal
import numpy as np
from data_lodder.logger import Logger

class BipedalGymEnv(gym.ActionWrapper):
    def __init__(self) -> None:
        base_env = gym.make("BipedalWalker-v3")
        super().__init__(base_env)
        self.training_logger = Logger("logs/bipedal_episode_reward.csv")
        self.episode_count = 0
        self.episode_return = 0

    def action(self, action):
        # Transforming action from [0,1] to [-1,1]
        return action * 2 - 1
    
    def reset(self, **kwargs):
        if self.episode_count > 0:
            self.training_logger.log("Episode", self.episode_count)
            self.training_logger.log("Reward", self.episode_return)
            self.training_logger.write()
            self.training_logger.print("Episode Complete")
        
        self.episode_count += 1
        self.episode_return = 0
        
        initial_obs, info = super().reset(**kwargs)
        return initial_obs
    
    def step(self, action):
        # obs, rew, done, info = super().step(action)
        observation, rew, terminated, truncated, info = super().step(action)

        self.episode_return += rew
        return observation, rew, terminated or truncated, info

    def close(self):
        self.training_logger.close()
        return super().close()

class PPOBipedalPolicyNet(nn.Module):
    def __init__(self, state_shape, action_shape) -> None:
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
        )

        self.actor_hidden = nn.Linear(256, 256)
        self.alpha_out = nn.Sequential(nn.Linear(256, action_shape[0]), nn.Softplus())
        self.beta_out = nn.Sequential(nn.Linear(256, action_shape[0]), nn.Softplus())

        self.value_head = nn.Linear(256, 1)

    def forward(self, inputs):
        features = self.feature_extractor(inputs)
        state_value = self.value_head(features)
        actor_inp = self.actor_hidden(features)
        alpha_params = self.alpha_out(actor_inp) + 1
        beta_params = self.beta_out(actor_inp) + 1
        return state_value, alpha_params, beta_params

class SACBipedalModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.num_actions = act_dim[0]

        self.actor_layers = nn.Sequential(
            nn.Linear(obs_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_layer = nn.Linear(256, self.num_actions)

        self.q_network_1 = nn.Sequential(
            nn.Linear(obs_dim[0] + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q_network_2 = nn.Sequential(
            nn.Linear(obs_dim[0] + self.num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        actor_features = self.actor_layers(state)
        mean_val = self.mean_layer(actor_features)
        log_std_val = self.log_std_layer(actor_features)
        # Restricting log_std within a reasonable range
        log_std_val = torch.clamp(log_std_val, -20, 2)
        return mean_val, log_std_val, state

    def get_q_values(self, state, action):
        sa_concat = torch.cat([state, action], dim=1)
        return self.q_network_1(sa_concat), self.q_network_2(sa_concat)

    def sample_action(self, state):
        mean_val, log_std_val, features = self(state)
        std_dev = log_std_val.exp()
        
        dist = Normal(mean_val, std_dev)
        z_sample = dist.rsample()
        sampled_action = torch.tanh(z_sample)
        
        log_prob_term = dist.log_prob(z_sample) - torch.log(1 - sampled_action.pow(2) + 1e-6)
        log_prob_term = log_prob_term.sum(1, keepdim=True)
        
        return sampled_action, log_prob_term, features
