# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import toml

from environments.carracing.carracing import RacingNet, CarRacing
from environments.bipedal.bipedal import PPOBipedalPolicyNet, BipedalGymEnv
from algorithms.ppo.ppo import ProximalPolicyTrainer

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
            env = BipedalGymEnv()
            print("\nSelected: BipedalWalker-v3")
            return env, "bipedal"
        else:
            print("\nInvalid choice! Please enter 1 or 2.")

def main():
    cfg = load_config()
    seed(cfg["seed"])
    
    # Get environment choice
    env, env_type = select_environment()

    # Initialize appropriate network and agent
    if env_type == "carracing":
        net = RacingNet(env.observation_space.shape, env.action_space.shape)
    else:
        net = PPOBipedalPolicyNet(env.observation_space.shape, env.action_space.shape)
            
    agent = ProximalPolicyTrainer(
        environment=env,
        model=net,
        initial_lr=cfg["lr"],
        discount_factor=cfg["gamma"],
        minibatch_size=cfg["batch_size"],
        gae_factor=cfg["gae_lambda"],
        clip_range=cfg["clip"],
        val_loss_coef=cfg["value_coef"],
        ent_loss_coef=cfg["entropy_coef"],
        optimization_epochs=cfg["epochs_per_step"],
        total_iterations=cfg["num_steps"],
        rollout_length=cfg["horizon"],
        checkpoint_path=cfg["save_dir"],
        checkpoint_interval=cfg["save_interval"],
    )
    print("\nUsing PPO algorithm")


    print("\nStarting training...")
    agent.run_training()

    env.close()

if __name__ == "__main__":
    main()