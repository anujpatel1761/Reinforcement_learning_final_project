import torch
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import time
from collections import deque
from tqdm import tqdm
from games.carracing import RacingNet
from wrapper import SACWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        state_dim = state_dim if isinstance(state_dim, tuple) else (state_dim,)
        action_dim = action_dim if isinstance(action_dim, tuple) else (action_dim,)

        self.states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, *action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
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

class RunningNormalizer:
    def __init__(self):
        self.mean = 0
        self.std = 1
        self.count = 0
        
    def __call__(self, x):
        if torch.is_tensor(x):
            batch_mean = x.mean().item()
            batch_std = x.std().item()
        else:
            batch_mean = float(np.mean(x))
            batch_std = float(np.std(x))
            
        self.count += 1
        delta = batch_mean - self.mean
        self.mean += delta / self.count
        self.std = max(1e-3, np.sqrt(self.std**2 + delta * (batch_mean - self.mean)))
        
        return x / (self.std + 1e-8)

class SAC:
    def __init__(self, env, actor_critic, logger):
        self.env = env
        self.actor_critic = actor_critic.to(device)
        self.logger = logger
        
        # Modified hyperparameters for stability
        self.max_episodes = 850
        self.batch_size = 256  # Increased for more stable updates
        self.gamma = 0.95  # Reduced for more immediate rewards
        self.tau = 0.005  # Increased for faster target network updates
        self.initial_alpha = 0.01  # Much lower temperature
        self.actor_lr = 3e-5  # Reduced learning rate
        self.critic_lr = 3e-5  # Reduced learning rate
        self.buffer_size = 500000
        self.reward_scale = 0.1  # Reduced reward scale
        self.start_steps = 10000  # More initial exploration
        self.gradient_steps = 1
        self.target_update_interval = 2
        self.max_grad_norm = 0.1  # Reduced gradient clipping
        
        # Initialize networks
        base_net = RacingNet(
            state_dim=env.observation_space.shape,
            action_dim=(2,)
        )
        self.target_critic = SACWrapper(base_net).to(device)
        self.target_critic.load_state_dict(self.actor_critic.state_dict())
        
        # Modified optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_critic.q1_net.parameters(),
            lr=self.actor_lr,
            weight_decay=0
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                {'params': self.target_critic.q1_net.parameters()},
                {'params': self.target_critic.q2_net.parameters()}
            ],
            lr=self.critic_lr,
            weight_decay=0
        )
        
        # Alpha tuning
        self.target_entropy = -np.prod(env.action_space.shape).item() * 0.5
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.initial_alpha
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-5)

        # Separate buffers for positive and regular experiences
        self.positive_buffer = ReplayBuffer(
            env.observation_space.shape,
            env.action_space.shape[0],
            50000  # Smaller buffer for positive experiences
        )
        self.buffer = ReplayBuffer(
            env.observation_space.shape,
            env.action_space.shape[0],
            self.buffer_size
        )

        # Training stability improvements
        self.reward_normalizer = RunningNormalizer()
        self.recent_rewards = deque(maxlen=100)
        self.total_steps = 0
        self.episodes = 0
        self.best_reward = float('-inf')

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            if not evaluate and self.total_steps < self.start_steps:
                # Conservative exploration
                action = np.array([
                    np.random.uniform(0.45, 0.55),  # Limited steering
                    np.random.uniform(0.8, 1.0)     # High forward bias
                ])
                return action
            
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _, _ = self.actor_critic.sample_action(state)
            action = action.cpu().numpy().squeeze()
            
            if not evaluate:
                # Gentle noise
                noise_scale = 0.1 * (0.9995 ** self.episodes)
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action + noise, 0, 1)
                
                # Strong forward bias
                action[1] = np.clip(action[1] * 1.5, 0.7, 1.0)
            
            return action

    def update_parameters(self):
        if self.buffer.size < self.batch_size:
            return None
            
        total_critic_loss = 0
        total_actor_loss = 0

        # Mix positive and regular experiences
        if self.positive_buffer.size > 0:
            pos_ratio = 0.3  # 30% positive experiences
            pos_batch_size = int(self.batch_size * pos_ratio)
            reg_batch_size = self.batch_size - pos_batch_size
            
            # Sample from both buffers
            pos_states, pos_actions, pos_rewards, pos_next_states, pos_dones = \
                self.positive_buffer.sample(max(1, pos_batch_size))
            reg_states, reg_actions, reg_rewards, reg_next_states, reg_dones = \
                self.buffer.sample(reg_batch_size)
            
            # Combine batches
            states = torch.cat([pos_states, reg_states], dim=0)
            actions = torch.cat([pos_actions, reg_actions], dim=0)
            rewards = torch.cat([pos_rewards, reg_rewards], dim=0)
            next_states = torch.cat([pos_next_states, reg_next_states], dim=0)
            dones = torch.cat([pos_dones, reg_dones], dim=0)
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Compute current Q values
        current_q1, current_q2 = self.actor_critic.get_q_values(states, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor_critic.sample_action(next_states)
            next_q1, next_q2 = self.target_critic.get_q_values(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards * self.reward_scale + (1 - dones) * self.gamma * next_q

        # Use Huber loss for more stable gradients
        critic_loss = F.huber_loss(current_q1, target_q) + F.huber_loss(current_q2, target_q)
        
        # Only update if loss is reasonable
        if critic_loss.item() < 10.0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.target_critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_steps % self.target_update_interval == 0:
            # Actor update
            actions_pred, log_probs_pred, _ = self.actor_critic.sample_action(states)
            q1_pred, q2_pred = self.actor_critic.get_q_values(states, actions_pred)
            min_q_pred = torch.min(q1_pred, q2_pred)
            
            actor_loss = (self.alpha * log_probs_pred - min_q_pred).mean()
            
            # Only update if loss is reasonable
            if abs(actor_loss.item()) < 10.0:
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Soft target update
                for target_param, param in zip(self.target_critic.parameters(), 
                                            self.actor_critic.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

        return critic_loss.item(), actor_loss.item() if self.total_steps % self.target_update_interval == 0 else 0

    def train(self):
        progress = tqdm(total=self.max_episodes, desc="Training Progress")
    
        # Warm-up phase with conservative actions
        warmup_episodes = 20  # Increased warm-up period
        for episode in range(warmup_episodes):
            state = self.env.reset()
            for _ in range(100):
                action = np.array([0.5, 0.9])  # Straight forward
                next_state, reward, done, info = self.env.step(action)
                if reward > 0:  # Only store positive experiences
                    self.positive_buffer.add(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break

        # Main training loop
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            actor_loss = critic_loss = 0
            episode_steps = 0
        
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
            
                # Store experience
                if reward > 0:
                    self.positive_buffer.add(state, action, reward, next_state, done)
                self.buffer.add(state, action, reward, next_state, done)
            
                if self.total_steps >= self.start_steps:
                    # Update parameters
                    update_result = self.update_parameters()
                    if update_result:
                        critic_loss, actor_loss = update_result
            
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                episode_steps += 1
            
                # Early termination if episode is too long or reward is too negative
                if episode_steps > 1000 or episode_reward < -20:
                    done = True
        
            # Update training statistics
            self.episodes += 1
            self.recent_rewards.append(episode_reward)
            avg_reward = np.mean(self.recent_rewards)
            self.best_reward = max(self.best_reward, episode_reward)
        
            # Log progress
            self.logger.log(
                episode=episode+1,
                total_steps=self.total_steps,
                episode_reward=episode_reward,
                avg_reward=avg_reward,
                best_reward=self.best_reward,
                actor_loss=actor_loss,
                critic_loss=critic_loss,
                actor_lr=self.actor_lr,
                buffer_size=self.buffer.size
            )
        
            progress.update(1)
            progress.set_description(
                f"Episode {episode+1} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Best: {self.best_reward:.1f}"
            )
        
            # Early stopping if we reach good performance
            if avg_reward > 15.0 and episode > 100:
                print("\nReached target performance! Stopping training.")
                break
    
        progress.close()