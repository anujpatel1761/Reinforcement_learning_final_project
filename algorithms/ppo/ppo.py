
# #ppo.py
# from typing import Tuple
# from os.path import join
# import gym
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.distributions import Beta
# from torch.utils.data import DataLoader
# from os import path
# from time import sleep

# from data_lodder.memory import Memory
# from data_lodder.logger import Logger
# import os
# from os.path import join

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


# class PPO:
#     def __init__(
#         self,
#         env: gym.Env,
#         net: nn.Module,
#         lr: float = 1e-4,
#         batch_size: int = 128,
#         gamma: float = 0.99,
#         gae_lambda: float = 0.95,
#         horizon: int = 1024,
#         epochs_per_step: int = 5,
#         num_steps: int = 1000,
#         clip: float = 0.2,
#         value_coef: float = 0.5,
#         entropy_coef: float = 0.01,
#         save_dir: str = "C:/Users/anujp/OneDrive/Desktop/deep-racing - Copy/deep-racing/ckpt",
#         save_interval: int = 100,
#     ) -> None:
#         self.env = env
#         self.net = net.to(device)

#         self.lr = lr
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.horizon = horizon
#         self.epochs_per_step = epochs_per_step
#         self.num_steps = num_steps
#         self.gae_lambda = gae_lambda
#         self.clip = clip
#         self.value_coef = value_coef
#         self.entropy_coef = entropy_coef
#         self.save_dir = save_dir
#         self.save_interval = save_interval

#         self.optim = optim.Adam(self.net.parameters(), lr=self.lr)
#         self.logger = Logger("C:/Users/anujp/OneDrive/Desktop/deep-racing - Copy/deep-racing/logs/training_bipedal.csv")

#         self.state = self._to_tensor(env.reset())
#         self.alpha = 1.0

#     def train(self):
#         os.makedirs(self.save_dir, exist_ok=True)

#         for step in range(self.num_steps):
#             self._set_step_params(step)
#             # Collect episode trajectory for the horizon length
#             with torch.no_grad():
#                 memory = self.collect_trajectory(self.horizon)

#             self.logger.log("Total Reward", memory.rewards.sum().item())

#             memory_loader = DataLoader(
#                 memory, batch_size=self.batch_size, shuffle=True,
#             )

#             avg_loss = 0.0

#             for epoch in range(self.epochs_per_step):
#                 for (
#                     states,
#                     actions,
#                     log_probs,
#                     rewards,
#                     advantages,
#                     values,
#                 ) in memory_loader:
#                     loss, _, _, _ = self.train_batch(
#                         states, actions, log_probs, rewards, advantages, values
#                     )

#                     avg_loss += loss

#             self.logger.log("Loss", avg_loss / len(memory_loader))
#             self.logger.print(f"Step {step}")
#             self.logger.write()

#             if step % self.save_interval == 0:
#                 self.save(join(self.save_dir, f"net_{step}.pth"))

#         # save final model
#         self.save(join(self.save_dir, f"net_final.pth"))
#         self.logger.close()

#     def train_batch(
#         self,
#         states: torch.Tensor,
#         old_actions: torch.Tensor,
#         old_log_probs: torch.Tensor,
#         rewards: torch.Tensor,
#         advantages: torch.Tensor,
#         old_values: torch.Tensor,
#     ):
#         self.optim.zero_grad()

#         values, alpha, beta = self.net(states)
#         values = values.squeeze(1)

#         policy = Beta(alpha, beta)
#         entropy = policy.entropy().mean()
#         log_probs = policy.log_prob(old_actions).sum(dim=1)

#         ratio = (log_probs - old_log_probs).exp()  # same as policy / policy_old
#         policy_loss_raw = ratio * advantages
#         policy_loss_clip = (
#             ratio.clamp(min=1 - self.clip, max=1 + self.clip) * advantages
#         )
#         policy_loss = -torch.min(policy_loss_raw, policy_loss_clip).mean()

#         with torch.no_grad():
#             value_target = advantages + old_values  # V_t = (Q_t - V_t) + V_t

#         value_loss = nn.MSELoss()(values, value_target)

#         entropy_loss = -entropy

#         loss = (
#             policy_loss
#             + self.value_coef * value_loss
#             + self.entropy_coef * entropy_loss
#         )

#         loss.backward()

#         self.optim.step()

#         return loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()

#     def collect_trajectory(self, num_steps: int, delay_ms: int = 0) -> Memory:
#         states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

#         for t in range(num_steps):
#             # Run one step of the environment based on the current policy
#             value, alpha, beta = self.net(self.state)
#             value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)

#             policy = Beta(alpha, beta)
#             action = policy.sample()
#             log_prob = policy.log_prob(action).sum()

#             next_state, reward, done, _ = self.env.step(action.cpu().numpy())

#             if done:
#                 next_state = self.env.reset()

#             next_state = self._to_tensor(next_state)

#             # Store the transition
#             states.append(self.state)
#             actions.append(action)
#             rewards.append(reward)
#             log_probs.append(log_prob)
#             values.append(value)
#             dones.append(done)

#             self.state = next_state

#             self.env.render()

#             if delay_ms > 0:
#                 sleep(delay_ms / 1000)

#         # Get value of last state (used in GAE)
#         final_value, _, _ = self.net(self.state)
#         final_value = final_value.squeeze(0)

#         # Compute generalized advantage estimates
#         advantages = self._compute_gae(rewards, values, dones, final_value)

#         # Convert to tensors
#         states = torch.cat(states)
#         actions = torch.stack(actions)
#         log_probs = torch.stack(log_probs)
#         advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
#         rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
#         values = torch.cat(values)

#         return Memory(states, actions, log_probs, rewards, advantages, values)

#     def save(self, filepath: str):
#         torch.save(self.net.state_dict(), filepath)

#     def load(self, filepath: str):
#         self.net.load_state_dict(torch.load(filepath))

#     def predict(
#         self, state: np.ndarray
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         state = self._to_tensor(state)
#         value, alpha, beta = self.net(state)
#         return value, alpha, beta

#     def _compute_gae(self, rewards, values, dones, last_value):
#         advantages = [0] * len(rewards)

#         last_advantage = 0

#         for i in reversed(range(len(rewards))):
#             delta = rewards[i] + (1 - dones[i]) * self.gamma * last_value - values[i]
#             advantages[i] = (
#                 delta + (1 - dones[i]) * self.gamma * self.gae_lambda * last_advantage
#             )

#             last_value = values[i]
#             last_advantage = advantages[i]

#         return advantages

#     def _to_tensor(self, x):
#         return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

#     def _set_step_params(self, step):
#         # interpolate self.alpha between 1.0 and 0.0
#         self.alpha = 1.0 - step / self.num_steps

#         for param_group in self.optim.param_groups:
#             param_group["lr"] = self.lr * self.alpha

#         self.logger.log("Learning Rate", self.optim.param_groups[0]["lr"])

from typing import Tuple
from os.path import join
import gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Beta
from torch.utils.data import DataLoader
from os import path
from time import sleep

from data_lodder.memory import Memory
from data_lodder.logger import Logger
import os
from os.path import join

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class ProximalPolicyTrainer:
    def __init__(
        self,
        environment: gym.Env,
        model: nn.Module,
        initial_lr: float = 1e-4,
        minibatch_size: int = 128,
        discount_factor: float = 0.99,
        gae_factor: float = 0.95,
        rollout_length: int = 1024,
        optimization_epochs: int = 5,
        total_iterations: int = 1000,
        clip_range: float = 0.2,
        val_loss_coef: float = 0.5,
        ent_loss_coef: float = 0.01,
        checkpoint_path: str = "ckpt",
        checkpoint_interval: int = 100,
    ) -> None:
        self.env = environment
        self.policy_net = model.to(device)

        self.initial_lr = initial_lr
        self.minibatch_size = minibatch_size
        self.gamma = discount_factor
        self.rollout_length = rollout_length
        self.optimization_epochs = optimization_epochs
        self.total_iterations = total_iterations
        self.gae_lambda = gae_factor
        self.clip_range = clip_range
        self.val_loss_coef = val_loss_coef
        self.ent_loss_coef = ent_loss_coef
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.initial_lr)
        self.logger = Logger("training_bipedal.csv")

        start_obs = self.env.reset()
        self.current_state = self._to_device_tensor(start_obs)
        self.lr_decay_factor = 1.0

    def run_training(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)

        for iteration in range(self.total_iterations):
            self._update_learning_rate(iteration)
            # Gather data for one rollout
            with torch.no_grad():
                memory_data = self._collect_experience(self.rollout_length)

            self.logger.log("Total Reward", memory_data.rewards.sum().item())

            data_loader = DataLoader(
                memory_data, batch_size=self.minibatch_size, shuffle=True,
            )

            cumulative_loss = 0.0

            for _ in range(self.optimization_epochs):
                for (
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_rewards,
                    batch_advantages,
                    batch_values,
                ) in data_loader:
                    loss_val, _, _, _ = self._update_model(
                        batch_states,
                        batch_actions,
                        batch_old_log_probs,
                        batch_rewards,
                        batch_advantages,
                        batch_values,
                    )
                    cumulative_loss += loss_val

            avg_loss = cumulative_loss / len(data_loader)
            self.logger.log("Loss", avg_loss)
            self.logger.print(f"Iteration {iteration}")
            self.logger.write()

            if iteration % self.checkpoint_interval == 0:
                self.save_model(join(self.checkpoint_path, f"net_{iteration}.pth"))

        # Save final model
        self.save_model(join(self.checkpoint_path, f"net_final.pth"))
        self.logger.close()

    def _update_model(
        self,
        sampled_states: torch.Tensor,
        sampled_old_actions: torch.Tensor,
        sampled_old_log_probs: torch.Tensor,
        sampled_rewards: torch.Tensor,
        sampled_advantages: torch.Tensor,
        sampled_old_values: torch.Tensor,
    ):
        self.optimizer.zero_grad()

        current_values, current_alpha, current_beta = self.policy_net(sampled_states)
        current_values = current_values.squeeze(1)

        new_policy = Beta(current_alpha, current_beta)
        new_entropies = new_policy.entropy().mean()
        new_log_probs = new_policy.log_prob(sampled_old_actions).sum(dim=1)

        probability_ratio = (new_log_probs - sampled_old_log_probs).exp()

        # Unclipped and clipped policy objectives
        unclip_obj = probability_ratio * sampled_advantages
        clip_obj = (
            probability_ratio.clamp(min=1 - self.clip_range, max=1 + self.clip_range)
            * sampled_advantages
        )

        policy_loss = -torch.min(unclip_obj, clip_obj).mean()

        with torch.no_grad():
            target_values = sampled_advantages + sampled_old_values

        value_loss = nn.MSELoss()(current_values, target_values)
        entropy_penalty = -new_entropies

        total_loss = policy_loss + self.val_loss_coef * value_loss + self.ent_loss_coef * entropy_penalty
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item(), entropy_penalty.item()

    def _collect_experience(self, steps: int, pause_ms: int = 0) -> Memory:
        state_buffer, action_buffer, reward_buffer, logprob_buffer, value_buffer, done_buffer = [], [], [], [], [], []

        for _ in range(steps):
            val_estimate, alpha_val, beta_val = self.policy_net(self.current_state)
            val_estimate = val_estimate.squeeze(0)
            alpha_val = alpha_val.squeeze(0)
            beta_val = beta_val.squeeze(0)

            action_dist = Beta(alpha_val, beta_val)
            chosen_action = action_dist.sample()
            chosen_action_logprob = action_dist.log_prob(chosen_action).sum()

            next_state, obtained_reward, terminal, _ = self.env.step(chosen_action.cpu().numpy())
            if terminal:
                next_state = self.env.reset()

            next_state_tensor = self._to_device_tensor(next_state)

            state_buffer.append(self.current_state)
            action_buffer.append(chosen_action)
            reward_buffer.append(obtained_reward)
            logprob_buffer.append(chosen_action_logprob)
            value_buffer.append(val_estimate)
            done_buffer.append(terminal)

            self.current_state = next_state_tensor
            self.env.render()

            if pause_ms > 0:
                sleep(pause_ms / 1000)

        # Compute advantages
        final_val, _, _ = self.policy_net(self.current_state)
        final_val = final_val.squeeze(0)
        computed_advantages = self._calculate_gae(reward_buffer, value_buffer, done_buffer, final_val)

        state_tensor = torch.cat(state_buffer)
        action_tensor = torch.stack(action_buffer)
        logprob_tensor = torch.stack(logprob_buffer)
        advantage_tensor = torch.tensor(computed_advantages, dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(reward_buffer, dtype=torch.float32, device=device)
        value_tensor = torch.cat(value_buffer)

        return Memory(state_tensor, action_tensor, logprob_tensor, reward_tensor, advantage_tensor, value_tensor)

    def save_model(self, file_name: str):
        torch.save(self.policy_net.state_dict(), file_name)

    def load_model(self, file_name: str):
        self.policy_net.load_state_dict(torch.load(file_name))

    def predict_action_values(
        self, input_state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tensor_state = self._to_device_tensor(input_state)
        val, a, b = self.policy_net(tensor_state)
        return val, a, b

    def _calculate_gae(self, rewards, values, dones, last_val):
        advantages = [0] * len(rewards)
        running_adv = 0

        for idx in reversed(range(len(rewards))):
            delta = rewards[idx] + (1 - dones[idx]) * self.gamma * last_val - values[idx]
            advantages[idx] = delta + (1 - dones[idx]) * self.gamma * self.gae_lambda * running_adv
            running_adv = advantages[idx]
            last_val = values[idx]

        return advantages

    def _to_device_tensor(self, input_data):
        return torch.tensor(input_data, dtype=torch.float32, device=device).unsqueeze(0)

    def _update_learning_rate(self, iteration):
        self.lr_decay_factor = 1.0 - iteration / self.total_iterations
        for g in self.optimizer.param_groups:
            g["lr"] = self.initial_lr * self.lr_decay_factor
        self.logger.log("Learning Rate", self.optimizer.param_groups[0]["lr"])

