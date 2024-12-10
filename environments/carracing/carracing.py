import torch
from torch import nn
import gym
from gym.spaces import Box
import numpy as np
from collections import deque

from data_lodder.logger import Logger


class RacingNet(nn.Module):
    def __init__(self, state_dim, action_dim) -> None:
        super().__init__()

        n_actions = action_dim[0]

        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        conv_out_size = self._get_conv_out(state_dim)

        # Estimates the parameters of a Beta distribution over actions
        self.actor_fc = nn.Sequential(nn.Linear(conv_out_size, 256), nn.ReLU(),)

        self.alpha_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(256, n_actions), nn.Softplus())

        # Estimates the value of the state
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv(x)

        # Estimate value of the state
        value = self.critic(x)

        # Estimate the parameters of a Beta distribution over actions
        x = self.actor_fc(x)

        # add 1 to alpha & beta to ensure the distribution is "concave and unimodal" (https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return value, alpha, beta

    def _get_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv(x)

        return int(np.prod(x.size()))


class CarRacing(gym.Wrapper):
    def __init__(self, frame_skip=0, frame_stack=4):
        self.env = gym.make("CarRacing-v2",render_mode="rgb_array")
        super().__init__(self.env)

        self.frame_skip = frame_skip
        self.frame_stack = frame_stack

        self.action_space = Box(low=0, high=1, shape=(2,))
        self.observation_space = Box(low=0, high=1, shape=(frame_stack, 96, 96))

        self.frame_buf = deque(maxlen=frame_stack)

        self.t = 0
        self.last_reward_step = 0
        self.total_reward = 0
        self.n_episodes = 0

        self.logger = Logger("C:/Users/anujp/OneDrive/Desktop/deep-racing - Copy/deep-racing/logs/episode_reward.csv")

    def preprocess(self, original_action):
        original_action = original_action * 2 - 1  # map from [0, 1] to [-1, 1]

        action = np.zeros(3)

        action[0] = original_action[0]

        # Separate acceleration and braking
        action[1] = max(0, original_action[1])
        action[2] = max(0, -original_action[1])

        return action

    def postprocess(self, original_observation):
        # Ensure observation is the correct shape and type
        if isinstance(original_observation, tuple):
            original_observation = original_observation[0]
        
        # Convert to float32 and normalize
        observation = original_observation.astype(np.float32)
        
        # Convert to grayscale using weighted sum
        grayscale = np.array([0.299, 0.587, 0.114])
        observation = np.dot(observation, grayscale) / 255.0
        
        return observation

    def shape_reward(self, reward):
        return np.clip(reward, -1, 1)

    def get_observation(self):
        return np.array(self.frame_buf)

    def reset(self):
        self.logger.log("Episode", self.n_episodes)
        self.logger.log("Reward", self.total_reward)
        self.logger.write()
        self.logger.print()
        
        self.t = 0
        self.last_reward_step = 0
        self.n_episodes += 1
        self.total_reward = 0
        
        observation = self.env.reset()
        first_frame = self.postprocess(observation)
        
        # Clear and refill the frame buffer
        self.frame_buf.clear()
        for _ in range(self.frame_stack):
            self.frame_buf.append(first_frame)
        
        return self.get_observation()

    def step(self, action):
        self.t += 1
        action = self.preprocess(action)

        total_reward = 0
        for _ in range(self.frame_skip + 1):
            new_frame, reward, terminated, truncated, info = self.env.step(action)
            self.total_reward += reward
            reward = self.shape_reward(reward)
            total_reward += reward

            if reward > 0:
                self.last_reward_step = self.t
                       
        done = bool(terminated or truncated)

        if self.t - self.last_reward_step > 30:
            done = True

        reward = total_reward / (self.frame_skip + 1)

        new_frame = self.postprocess(new_frame)
        self.frame_buf.append(new_frame)

        return self.get_observation(), reward, done, info

