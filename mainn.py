# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import toml

# from games.carracing import RacingNet, CarRacing
# from games.bipedal import BipedalNet, BipedalWalker, SACBipedalNet
# from ppo import PPO
# from sac import SAC, SACRacingNet
from environments.carracing.carracing import RacingNet, CarRacing
from environments.bipedal.bipedal import BipedalNet, BipedalWalker, SACBipedalNet
from algorithms.ppo.ppo import PPO
from algorithms.sac.sac import SAC, SACRacingNet

CONFIG_FILE = r"C:/Users/anujp/Desktop/Reinforcement_learning_final_project/parameters/ppo/ppo.toml"

def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            # Remove any BOM or hidden characters
            content = content.encode("ascii", "ignore").decode()
            config = toml.loads(content)
        return config
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        exit(1)

def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def select_algorithm():
    while True:
        print("\nSelect algorithm to use:")
        print("1. PPO (Proximal Policy Optimization)")
        print("2. SAC (Soft Actor-Critic)")
        
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice in ["1", "2"]:
            return choice
        else:
            print("\nInvalid choice! Please enter 1 or 2.")

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

def main():
    cfg = load_config()
    seed(cfg["seed"])

    # Get algorithm choice
    algo_choice = select_algorithm()
    
    # Get environment choice
    env, env_type = select_environment()

    # Initialize appropriate network and agent based on choices
    if algo_choice == "1":  # PPO
        if env_type == "carracing":
            net = RacingNet(env.observation_space.shape, env.action_space.shape)
        else:
            net = BipedalNet(env.observation_space.shape, env.action_space.shape)
            
        agent = PPO(
            env,
            net,
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            gae_lambda=cfg["gae_lambda"],
            clip=cfg["clip"],
            value_coef=cfg["value_coef"],
            entropy_coef=cfg["entropy_coef"],
            epochs_per_step=cfg["epochs_per_step"],
            num_steps=cfg["num_steps"],
            horizon=cfg["horizon"],
            save_dir=cfg["save_dir"],
            save_interval=cfg["save_interval"],
        )
        print("\nUsing PPO algorithm")
        
    else:  
        if env_type == "carracing":
            net = SACRacingNet(env.observation_space.shape, env.action_space.shape)
        else:
            net = SACBipedalNet(env.observation_space.shape, env.action_space.shape)
            
        agent = SAC(
            env=env,
            actor_critic=net,
            lr=cfg["lr"],
            batch_size=cfg["batch_size"],
            gamma=cfg["gamma"],
            tau=cfg.get("tau", 0.001),
            alpha=cfg.get("alpha", 0.3),
            buffer_size=cfg.get("buffer_size", 1e5),
            target_update_interval=cfg.get("target_update_interval", 2),
            num_steps=cfg["num_steps"],
            updates_per_step=cfg.get("updates_per_step", 4),
            start_steps=cfg.get("start_steps", 10000),
            save_interval=cfg["save_interval"],
            save_dir=cfg["save_dir"]
        )
        print("\nUsing SAC algorithm")

    print("\nStarting training...")
    agent.train()

    env.close()

if __name__ == "__main__":
    main()