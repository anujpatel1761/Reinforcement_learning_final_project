# sac.py - Part 1: Imports and ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from logger import Logger
from pathlib import Path
import traceback
from tqdm import tqdm
from collections import deque
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=2000000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.float32)
        
        self.priorities = np.zeros(self.max_size, dtype=np.float32)
        self.alpha = 0.6
        self.epsilon = 1e-6
        self.beta = 0.4
        self.beta_increment = 0.001
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, prioritized=True):
        if prioritized and self.size > 0:
            probs = self.priorities[:self.size] ** self.alpha
            probs = probs / np.sum(probs)
            indices = np.random.choice(self.size, batch_size, p=probs)
            weights = (self.size * probs[indices]) ** -self.beta
            weights = weights / weights.max()
            self.beta = min(1.0, self.beta + self.beta_increment)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)
            weights = np.ones_like(indices, dtype=np.float32)

        samples = (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.FloatTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device)
        )
        
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # Ensure priorities is 1D
        priorities = np.squeeze(priorities)  # This will convert (512,1) to (512,)
        priorities = np.abs(priorities) + self.epsilon
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, np.max(priorities))

        # sac.py - Part 2: SAC Class Initialization

class SAC:
    def __init__(
        self,
        env,
        actor_critic,
        max_episodes=1500,
        lr=3e-4,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        alpha="auto",
        buffer_size=2000000,
        target_update_interval=2,
        num_steps=3000000,
        updates_per_step=1,
        start_steps=20000,
        save_interval=100000,
        reward_scale=5.0,
        gradient_steps=2,
        learning_starts=10000,
        save_dir='ckpt',
        log_interval=1000
    ):
        try:
            # Store environment and actor-critic network
            self.env = env
            self.actor_critic = actor_critic.to(device)
            
            # Initialize dimensions
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Initialize target network
            self.target_critic = type(actor_critic)(
                env.observation_space.shape,
                env.action_space.shape
            ).to(device)
            self.target_critic.load_state_dict(self.actor_critic.state_dict())
            
            # Setup directories
            self.current_dir = Path(__file__).parent.resolve()
            self.log_dir = self.current_dir / "logs"
            self.save_dir = self.current_dir / save_dir
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize optimizers
            self.actor_optimizer = torch.optim.AdamW(
                self.actor_critic.actor.parameters(),
                lr=lr,
                weight_decay=1e-4,
                amsgrad=True
            )
            
            self.critic_optimizer = torch.optim.AdamW(
                list(self.actor_critic.q1.parameters()) +
                list(self.actor_critic.q2.parameters()),
                lr=lr * 2.0,
                weight_decay=1e-4,
                amsgrad=True
            )
            
            # Setup alpha (entropy temperature)
            if alpha == "auto":
                self.target_entropy = -np.prod(env.action_space.shape).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer = torch.optim.AdamW(
                    [self.log_alpha],
                    lr=lr,
                    amsgrad=True
                )
            else:
                self.alpha = alpha
            
            # Initialize replay buffer and logging
            self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
            self.logger = Logger(str(self.log_dir / "sac_training.csv"))
            self.episode_logger = Logger(str(self.log_dir / "sac_episode_reward.csv"))
            
            # Store hyperparameters
            self.batch_size = batch_size
            self.gamma = gamma
            self.tau = tau
            self.max_episodes = max_episodes
            self.target_update_interval = target_update_interval
            self.num_steps = num_steps
            self.updates_per_step = updates_per_step
            self.start_steps = start_steps
            self.save_interval = save_interval
            self.reward_scale = reward_scale
            self.gradient_steps = gradient_steps
            self.learning_starts = learning_starts
            self.log_interval = log_interval
            
            # Initialize tracking variables
            self.total_steps = 0
            self.updates = 0
            self.episodes = 0
            self.max_grad_norm = 0.5
            self.episode_rewards = deque(maxlen=100)
            self.best_reward = float('-inf')
            self.best_average = float('-inf')
            
            # Initialize learning rate scheduler
            self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer,
                mode='max',
                factor=0.5,
                patience=25,
                min_lr=1e-6,
                verbose=True
            )
            
            print("SAC initialization successful")
            
        except Exception as e:
            print(f"Error in SAC initialization: {str(e)}")
            print(traceback.format_exc())
            raise

        # sac.py - Part 3: Action Selection and Parameter Updates

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            if not evaluate and self.total_steps < self.start_steps:
                # Random actions for initial exploration
                action = np.random.uniform(-1, 1, self.env.action_space.shape)
            else:
                # Use policy network for action selection
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                action, _, _ = self.actor_critic.sample_action(state_t)
                action = action.squeeze().cpu().numpy()
                
                if not evaluate:
                    # Add noise for exploration during training
                    noise = np.random.normal(0, 0.1, size=action.shape)
                    noise *= max(0, 1 - self.total_steps / 1e6)  # Decay noise over time
                    action = np.clip(action + noise, -1, 1)
                    
            return action

    def update_parameters(self):
        if self.buffer.size < self.batch_size:
            return None, None

        (states, actions, rewards, next_states, dones), indices, weights = self.buffer.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(device)
        rewards = rewards * self.reward_scale

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor_critic.sample_action(next_states)
            next_log_probs = next_log_probs.unsqueeze(-1)
            next_q1, next_q2 = self.target_critic.get_q_values(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q

        current_q1, current_q2 = self.actor_critic.get_q_values(states, actions)
        critic_loss1 = F.huber_loss(current_q1, target_q, reduction='none')
        critic_loss2 = F.huber_loss(current_q2, target_q, reduction='none')
        
        critic_loss1 = (critic_loss1 * weights).mean()
        critic_loss2 = (critic_loss2 * weights).mean()
        critic_loss = critic_loss1 + critic_loss2

        # Calculate TD errors and ensure proper shape
        td_errors = torch.max(
            torch.abs(target_q - current_q1),
            torch.abs(target_q - current_q2)
        ).detach().cpu().numpy()
        td_errors = np.squeeze(td_errors)  # Ensure 1D shape

        # Update priorities in buffer
        self.buffer.update_priorities(indices, td_errors)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor_critic.q1.parameters()) + 
            list(self.actor_critic.q2.parameters()),
            self.max_grad_norm
        )
        self.critic_optimizer.step()

        actor_loss = None
        if self.updates % 2 == 0:
            actions_new, log_probs, _ = self.actor_critic.sample_action(states)
            log_probs = log_probs.unsqueeze(-1)
            q1, q2 = self.actor_critic.get_q_values(states, actions_new)
            min_q = torch.min(q1, q2)
            actor_loss = (self.alpha * log_probs - min_q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.actor.parameters(),
                self.max_grad_norm
            )
            self.actor_optimizer.step()

            if isinstance(self.alpha, torch.Tensor):
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp()

        if self.updates % self.target_update_interval == 0:
            for param, target_param in zip(
                self.actor_critic.parameters(),
                self.target_critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

        self.updates += 1
        return (
            critic_loss.item(),
            actor_loss.item() if actor_loss is not None else None
        )
    
    # sac.py - Part 4: Checkpointing and Training Loop

    def save_checkpoint(self, step):
        try:
            checkpoint_path = self.save_dir / f"sac_model_{step}.pth"
            checkpoint = {
                'actor_critic_state_dict': self.actor_critic.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'total_steps': self.total_steps,
                'episodes': self.episodes,
                'buffer_state': {
                    'size': self.buffer.size,
                    'ptr': self.buffer.ptr,
                    'priorities': self.buffer.priorities,
                },
                'best_reward': self.best_reward,
                'best_average': self.best_average
            }
            
            if isinstance(self.alpha, torch.Tensor):
                checkpoint['log_alpha'] = self.log_alpha
                checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
                
            torch.save(checkpoint, str(checkpoint_path))
            print(f"\nSaved checkpoint to {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"\nWarning: Could not save checkpoint: {str(e)}")
            return False

    def train(self):
        try:
            # Initialize training variables
            episode_reward = 0
            episode_steps = 0
            obs = self.env.reset()
            state = obs
            
            training_start_time = time.time()
            self.last_actor_loss = None
            self.last_critic_loss = None
            critic_loss = None  # Initialize critic_loss
            actor_loss = None   # Initialize actor_loss

            # Setup progress bars
            episode_pbar = tqdm(total=self.max_episodes, position=0, desc="Episodes")
            step_pbar = tqdm(total=self.num_steps, position=1, desc="Steps")
            status_bar = tqdm(total=0, position=2, bar_format='{desc}')

            print(f"\nStarting training for {self.num_steps} steps...")

            for step in range(self.num_steps):
                if self.episodes >= self.max_episodes:
                    print(f"\nReached target of {self.max_episodes} episodes. Training complete!")
                    break

                step_pbar.update(1)

                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # Train agent after collecting sufficient data
                if self.total_steps >= self.learning_starts:
                    for _ in range(self.gradient_steps):
                        critic_loss, actor_loss = self.update_parameters()
                        if critic_loss is not None:
                            self.last_critic_loss = critic_loss
                            self.last_actor_loss = actor_loss

                # Update progress bar with losses if available
                if self.total_steps % self.log_interval == 0:
                    status = "Training..."
                    if self.last_critic_loss is not None:
                        status = f"Loss → Critic: {self.last_critic_loss:.2f}"
                        if self.last_actor_loss is not None:
                            status += f", Actor: {self.last_actor_loss:.2f}"
                    step_pbar.set_description(status)

                # Episode completion
                if done:
                    self.episodes += 1
                    self.episode_rewards.append(episode_reward)
                    self.best_reward = max(self.best_reward, episode_reward)
                    
                    # Calculate average reward
                    current_avg = np.mean(list(self.episode_rewards))
                    self.best_average = max(self.best_average, current_avg)
                    
                    # Update learning rate scheduler
                    self.actor_scheduler.step(current_avg)
                    
                    # Log progress
                    tqdm.write(f"Episode {self.episodes} ended with Reward: {episode_reward:.2f}")
                    episode_pbar.update(1)

                    # Calculate remaining time
                    elapsed_time = time.time() - training_start_time
                    avg_time_per_episode = elapsed_time / max(1, self.episodes)
                    remaining_episodes = self.max_episodes - self.episodes
                    estimated_remaining = avg_time_per_episode * remaining_episodes

                    # Update status display
                    status_text = (
                        f"Episode {self.episodes}/{self.max_episodes} | "
                        f"Reward: {episode_reward:.1f} | "
                        f"Best: {self.best_reward:.1f} | "
                        f"Avg(100): {current_avg:.1f} | "
                        f"Steps: {episode_steps} | "
                        f"Time Left: {estimated_remaining/60:.1f}m"
                    )
                    status_bar.set_description(status_text)

                    # Log episode data
                    self.episode_logger.log("Episode", self.episodes)
                    self.episode_logger.log("Total Steps", self.total_steps)
                    self.episode_logger.log("Episode Reward", episode_reward)
                    self.episode_logger.log("Actor Loss", self.last_actor_loss if self.last_actor_loss is not None else float('nan'))
                    self.episode_logger.log("Critic Loss", self.last_critic_loss if self.last_critic_loss is not None else float('nan'))
                    self.episode_logger.log("Actor LR", self.actor_optimizer.param_groups[0]['lr'])
                    self.episode_logger.log("Buffer Size", self.buffer.size)
                    self.episode_logger.write()

                    # Reset episode state
                    state = self.env.reset()
                    episode_reward = 0
                    episode_steps = 0

                # Save periodic checkpoints
                if step > 0 and step % self.save_interval == 0:
                    self.save_checkpoint(step)

            # Clean up progress bars
            episode_pbar.close()
            step_pbar.close()
            status_bar.close()

            # Final training summary
            print("\nTraining Summary:")
            print(f"Total Episodes Completed: {self.episodes}")
            print(f"Best Episode Reward: {self.best_reward:.1f}")
            print(f"Best Average Reward: {self.best_average:.1f}")
            print(f"Total Training Steps: {self.total_steps}")
            print(f"Total Training Time: {(time.time() - training_start_time)/60:.1f} minutes")

            # Close loggers
            self.logger.close()
            self.episode_logger.close()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
        except Exception as e:
            print(f"\nError in training: {str(e)}")
            print(traceback.format_exc())
            raise