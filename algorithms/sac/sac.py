# sac.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from torch.utils.data import DataLoader
from data_lodder.logger import Logger
import random
from collections import deque
import os
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e5):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((self.max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, *action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, *state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.bool_)
        
    def add(self, state, action, reward, next_state, done):
        reward = np.clip(reward, -1, 1)  # Normalize rewards
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[ind]).to(device),
            torch.FloatTensor(self.actions[ind]).to(device),
            torch.FloatTensor(self.rewards[ind]).to(device),
            torch.FloatTensor(self.next_states[ind]).to(device),
            torch.FloatTensor(self.dones[ind]).to(device)
        )

class SACRacingNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim[0]

        self.encoder = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        test_input = torch.zeros(1, *state_dim)
        conv_out_size = self.encoder(test_input).shape[1]
        
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(512, action_dim[0])
        self.log_std = nn.Linear(512, action_dim[0])

        self.q1 = nn.Sequential(
            nn.Linear(conv_out_size + action_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(conv_out_size + action_dim[0], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def forward(self, state):
        features = self.encoder(state)
        actor_features = self.actor(features)
        
        mean = self.mean(actor_features)
        log_std = self.log_std(actor_features)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std, features
    
    def get_q_values(self, state, action, features=None):
        if features is None:
            features = self.encoder(state)
        sa = torch.cat([features, action], dim=1)
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

class SAC:
    def __init__(
        self,
        env,
        actor_critic,
        lr=1e-4,
        batch_size=512,
        gamma=0.99,
        tau=0.001,
        alpha=0.3,
        buffer_size=1e5,
        target_update_interval=2,
        num_steps=800000,
        updates_per_step=2,
        start_steps=10000,
        save_interval=10000,
        log_interval=500,
        save_dir='ckpt'
    ):
        self.env = env
        self.actor_critic = actor_critic.to(device)
        # self.target_critic = SACRacingNet(env.observation_space.shape, env.action_space.shape).to(device)
        self.target_critic = type(actor_critic)(env.observation_space.shape, env.action_space.shape).to(device)
        self.target_critic.load_state_dict(self.actor_critic.state_dict())
        
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.actor_critic.q1.parameters()) + 
            list(self.actor_critic.q2.parameters()), 
            lr=lr
        )
        
        self.buffer = ReplayBuffer(env.observation_space.shape, env.action_space.shape, buffer_size)
        self.logger = Logger("C:/Users/anujp/OneDrive - Northeastern University/Desktop/deep-racing - Copy/deep-racing/logs/sac_training.csv")
        self.episode_logger = Logger("C:/Users/anujp/OneDrive - Northeastern University/Desktop/deep-racing - Copy/deep-racing/logs/sac_episode_reward.csv")
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_update_interval = target_update_interval
        self.num_steps = num_steps
        self.updates_per_step = updates_per_step
        self.start_steps = start_steps
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.save_dir = save_dir
        
        self.total_steps = 0
        self.updates = 0
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            if self.total_steps < self.start_steps:
                action = np.random.random(self.env.action_space.shape)
            else:
                action, _, _ = self.actor_critic.sample_action(state)
                action = action.squeeze().cpu().numpy()
            return action
    
# Update the update_parameters method in the SAC class:

    def update_parameters(self):
        if self.buffer.size < self.batch_size:
            return None, None
            
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        with torch.no_grad():
            _, _, features = self.actor_critic(states)
        current_q1, current_q2 = self.actor_critic.get_q_values(states, actions, features)
        
        with torch.no_grad():
            next_actions, next_log_probs, next_features = self.actor_critic.sample_action(next_states)
            next_q1, next_q2 = self.target_critic.get_q_values(next_states, next_actions, next_features)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)  # Add gradient clipping
        self.critic_optimizer.step()
        
        actions_new, log_probs, features = self.actor_critic.sample_action(states)
        q1, q2 = self.actor_critic.get_q_values(states, actions_new, features.detach())
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)  # Add gradient clipping
        self.actor_optimizer.step()
        
        if self.updates % self.target_update_interval == 0:
            for param, target_param in zip(self.actor_critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        self.updates += 1
        
        return critic_loss.item(), actor_loss.item()

    # Update the ReplayBuffer's add method for reward scaling:
    def add(self, state, action, reward, next_state, done):
        reward = np.clip(reward * 0.1, -1, 1)  # Scale rewards by 0.1 before clipping
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def train(self):
        episode_reward = 0
        episode_steps = 0
        episodes = 0
        state = self.env.reset()
        
        print(f"Starting training for {self.num_steps} steps...")
        
        for step in range(self.num_steps):
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            if self.total_steps >= self.start_steps:
                for _ in range(self.updates_per_step):
                    critic_loss, actor_loss = self.update_parameters()
                    if critic_loss is not None and step % self.log_interval == 0:
                        self.logger.log("Step", step)
                        self.logger.log("Critic Loss", critic_loss)
                        self.logger.log("Actor Loss", actor_loss)
                        self.logger.log("Total Steps", self.total_steps)
                        self.logger.write()
                        print(f"Step {step}, Critic Loss: {critic_loss:.3f}, Actor Loss: {actor_loss:.3f}")
            
            if done:
                episodes += 1
                self.episode_logger.log("Episode", episodes)
                self.episode_logger.log("Total Reward", episode_reward)
                self.episode_logger.log("Episode Steps", episode_steps)
                self.episode_logger.log("Total Steps", self.total_steps)
                self.episode_logger.write()
                
                print(f"/nEpisode {episodes}:")
                print(f"Total Reward: {episode_reward:.2f}")
                print(f"Episode Steps: {episode_steps}")
                print(f"Total Steps: {self.total_steps}")
                
                state = self.env.reset()
                episode_reward = 0
                episode_steps = 0
            
            if step % self.save_interval == 0:
                save_path = os.path.join(self.save_dir, f"sac_model_{step}.pth")
                torch.save(self.actor_critic.state_dict(), save_path)
                print(f"/nSaved model checkpoint to {save_path}")
        
        print("/nTraining finished. Saving final model...")
        torch.save(
            self.actor_critic.state_dict(),
            os.path.join(self.save_dir, "sac_model_final.pth")
        )
        
        self.logger.close()
        self.episode_logger.close()
