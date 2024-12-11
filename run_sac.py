# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
from pathlib import Path
import toml

from environments.bipedal import BipedalWalker, SACBipedalNet
from environments.carracing import CarRacing
from sac.sac import SAC
from sac.logger import Logger  
from sac.wrapper import SACWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG_FILE = "config.toml"

def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            content = content.encode("ascii", "ignore").decode()
            config = toml.loads(content)
        return config
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        exit(1)

def seed(seed_value=42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def select_environment():
    while True:
        print("\nSelect environment to train:")
        print("1. CarRacing-v2")
        print("2. BipedalWalker-v3")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            env = CarRacing(frame_skip=1, frame_stack=4)
            print("\nSelected: CarRacing-v2")
            return env, "carracing"
        elif choice == "2":
            env = BipedalWalker()
            print("\nSelected: BipedalWalker-v3")
            return env, "bipedal"
        else:
            print("\nInvalid choice! Please enter 1 or 2.")

def create_network_and_wrapper(env_type, env):
    if env_type == "carracing":
        # CarRacing specific network initialization
        base_net = RacingNet(
            state_dim=(env.observation_space.shape[0], 96, 96),  # Specific to CarRacing
            action_dim=(2,)  # CarRacing uses 2D actions
        )
        wrapped_net = SACWrapper(base_net)
    else:
        # BipedalWalker network initialization
        base_net = BipedalNet(
            state_dim=env.observation_space.shape,
            action_dim=env.action_space.shape
        )
        wrapped_net = SACWrapper(base_net)
    
    return wrapped_net

def main():
    # Load configuration
    cfg = load_config()
    
    # Set up directories
    current_dir = Path.cwd()
    log_dir = current_dir / "logs"
    log_dir.mkdir(mode=0o777, exist_ok=True, parents=True)
    
    # Set seed for reproducibility
    seed(42)
    
    # Get environment choice
    env, env_type = select_environment()
    
    # Initialize network with wrapper
    wrapped_net = create_network_and_wrapper(env_type, env)
    
    # Create loggers with the maximum episodes from config
    episode_logger = Logger(
        str(log_dir / f"{env_type}_episode_reward.csv"),
        total_episodes=cfg['sac']['max_episodes'],
        name=f"{env_type.capitalize()} Training"
    )
    training_logger = Logger(
        str(log_dir / f"{env_type}_training_metrics.csv"),
        total_episodes=cfg['sac']['max_episodes'],
        name=f"{env_type.capitalize()} Metrics"
    )

    # Initialize SAC agent
    agent = SAC(
        env=env,
        actor_critic=wrapped_net,
        logger=training_logger
    )
    
    # Override default parameters with config values
    agent.max_episodes = cfg['sac']['max_episodes']
    agent.batch_size = cfg['sac']['batch_size']
    agent.gamma = cfg['sac']['gamma']
    agent.actor_lr = cfg['sac']['actor_lr']
    agent.critic_lr = cfg['sac']['critic_lr']
    agent.buffer_size = cfg['sac']['buffer_size']
    agent.tau = cfg['sac']['tau']
    agent.reward_scale = cfg['sac']['reward_scale']
    agent.start_steps = cfg['sac']['start_steps']
    agent.gradient_steps = cfg['sac']['gradient_steps']
    agent.target_update_interval = cfg['sac']['target_update_interval']
    agent.max_grad_norm = cfg['sac']['max_grad_norm']
    
    # Print training configuration
    print("\nTraining Configuration:")
    print("----------------------")
    print(f"Environment: {env_type}")
    print(f"Episodes: {agent.max_episodes}")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Actor LR: {agent.actor_lr}")
    print(f"Critic LR: {agent.critic_lr}")
    print(f"Buffer Size: {agent.buffer_size}")
    print(f"Reward Scale: {agent.reward_scale}")
    print(f"Start Steps: {agent.start_steps}")
    print(f"Target Update Interval: {agent.target_update_interval}")
    print(f"Max Grad Norm: {agent.max_grad_norm}")
    print(f"Device: {device}")
    if env_type == "carracing":
        print(f"Frame Stack: 4")
        print(f"Frame Skip: 1")
    print("----------------------\n")
    
    try:
        print("Starting training...")
        agent.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        print("\nClosing environment...")
        env.close()
        episode_logger.close()
        training_logger.close()

if __name__ == "__main__":
    main()