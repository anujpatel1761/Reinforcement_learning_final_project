# bipedal.py
import torch
from torch import nn
import gym
import numpy as np
from logger import Logger
from pathlib import Path
from torch.distributions import Normal

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512,512,256]):
        super(GaussianPolicy, self).__init__()
        layers = []
        input_dim = state_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(input_dim, hd))
            layers.append(nn.ReLU())
            input_dim = hd
        self.backbone = nn.Sequential(*layers)
        
        self.mean = nn.Linear(input_dim, action_dim)
        self.log_std = nn.Linear(input_dim, action_dim)

        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.bias, -3e-3, 3e-3)
        
    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(axis=-1)
        
        tanh_action = torch.tanh(raw_action)
        log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6).sum(axis=-1)
        
        return tanh_action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512,512,256]):
        super(QNetwork, self).__init__()
        input_dim = state_dim + action_dim
        layers = []
        for hd in hidden_dims:
            layers.append(nn.Linear(input_dim, hd))
            layers.append(nn.ReLU())
            input_dim = hd
        layers.append(nn.Linear(input_dim, 1))
        self.q = nn.Sequential(*layers)

        nn.init.uniform_(self.q[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q[-1].bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q(x)

class SACBipedalNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[512,512,256]):
        super(SACBipedalNet, self).__init__()
        self.actor = GaussianPolicy(state_dim[0], action_dim[0], hidden_dim)
        self.q1 = QNetwork(state_dim[0], action_dim[0], hidden_dim)
        self.q2 = QNetwork(state_dim[0], action_dim[0], hidden_dim)

    def sample_action(self, state):
        return self.actor.sample(state)

    def get_q_values(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

class BipedalWalker(gym.Wrapper):
    def __init__(self) -> None:
        env = gym.make("BipedalWalker-v3")
        super().__init__(env)
        
        current_dir = Path(__file__).parent
        project_dir = current_dir.parent
        log_dir = project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = Logger(str(log_dir / "bipedal_episode_reward.csv"))
        self.n_episodes = 0
        self.total_reward = 0
        
        self.running_mean = np.zeros(env.observation_space.shape[0])
        self.running_var = np.ones(env.observation_space.shape[0])
        self.running_count = 1.0
        
        self.prev_x_position = 0
        self.prev_velocity = 0
        self.steps_upright = 0
        self.max_steps_upright = 0
        
        self.success_threshold = 250
        self.successful_episodes = 0
        
        self.reward_history = []
        self.reward_scale = 1.0
        self.reward_update_freq = 10
        
    def update_reward_scaling(self):
        if len(self.reward_history) >= self.reward_update_freq:
            recent_mean = np.mean(self.reward_history[-self.reward_update_freq:])
            if abs(recent_mean) < 1e-3:
                self.reward_scale *= 2
            elif abs(recent_mean) > 1:
                self.reward_scale *= 0.5
            self.reward_scale = np.clip(self.reward_scale, 0.1, 10.0)
            self.reward_history = []

    def normalize_observation(self, obs):
        momentum = 0.99
        self.running_mean = momentum * self.running_mean + (1 - momentum) * obs
        self.running_var = momentum * self.running_var + (1 - momentum) * np.square(obs - self.running_mean)
        
        normalized_obs = (obs - self.running_mean) / np.sqrt(self.running_var + 1e-8)
        return np.clip(normalized_obs, -5, 5)

    def compute_shaped_reward(self, obs, reward, info):
        hull_angle = obs[0]
        x_position = info.get('x_pos', 0)
        
        forward_progress = x_position - self.prev_x_position
        velocity_change = forward_progress - self.prev_velocity
        forward_reward = forward_progress * 3.0 + np.clip(velocity_change, -1, 1)
        
        self.prev_x_position = x_position
        self.prev_velocity = forward_progress
        
        angle_penalty = -abs(hull_angle) * 0.5
        upright_reward = np.exp(-abs(hull_angle)) * 2.0
        
        if abs(hull_angle) < 0.3:
            self.steps_upright += 1
            upright_bonus = np.log1p(self.steps_upright) * 0.1
        else:
            self.steps_upright = max(0, self.steps_upright - 1)
            upright_bonus = 0
        
        energy_cost = info.get('energy_cost', 0)
        energy_penalty = -abs(energy_cost) * 0.2 * (1 - np.exp(-self.n_episodes / 50))
        
        shaped_reward = (
            forward_reward +
            angle_penalty +
            upright_reward +
            upright_bonus +
            energy_penalty +
            reward
        ) * self.reward_scale
        
        if shaped_reward > self.success_threshold:
            shaped_reward *= 1.2
            
        self.reward_history.append(shaped_reward)
        self.update_reward_scaling()
        
        return np.clip(shaped_reward, -10, 20)

    def reset(self, **kwargs):
        if self.n_episodes > 0:
            self.logger.log("Episode", self.n_episodes)
            self.logger.log("Reward", self.total_reward)
            self.logger.write()
            
            if self.total_reward > self.success_threshold:
                self.successful_episodes += 1
        
        self.n_episodes += 1
        self.total_reward = 0
        self.prev_x_position = 0
        self.prev_velocity = 0
        self.steps_upright = 0
        
        observation, info = self.env.reset(**kwargs)
        return self.normalize_observation(observation)

    def step(self, action):
        action = np.clip(action, -1, 1)
        observation, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = self.compute_shaped_reward(observation, reward, info)
        normalized_obs = self.normalize_observation(observation)
        self.total_reward += shaped_reward
        return normalized_obs, shaped_reward, terminated or truncated, info

    def close(self):
        self.logger.close()
        return super().close()
