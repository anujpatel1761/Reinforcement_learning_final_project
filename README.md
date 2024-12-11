# Reinforcement Learning Final Project

This repository contains the implementation of a final project focused on reinforcement learning (RL). The project explores various RL algorithms, including Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC), applied to custom and standard environments. The modular structure of the repository enables easy experimentation and extension.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Algorithms Implemented](#algorithms-implemented)
- [Installation](#installation)
- [Usage](#usage)
  - [Running PPO](#running-ppo)
  - [Running SAC](#running-sac)
- [Directory and File Overview](#directory-and-file-overview)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

The goal of this project is to develop and evaluate reinforcement learning agents using popular RL algorithms. These agents are trained to perform specific tasks in simulated environments. The repository is structured to allow for easy experimentation with different algorithms, environments, and hyperparameters.

---

## Project Structure

The repository is organized into the following directories and files:

### Directories

- **algorithms/**: Contains implementations of reinforcement learning algorithms, such as PPO and SAC.
- **ckpt/**: Stores checkpoint files for saving and resuming model states during training.
- **data_loader/**: Includes scripts for loading and preprocessing datasets used for training and evaluation.
- **environments/**: Defines custom environments and wrappers for standard environments like those in OpenAI Gym.
- **logs/**: Holds logs for tracking training progress, metrics, and results.
- **models/**: Contains neural network architectures used for the agents.
- **parameters/**: Stores configuration files with hyperparameters and other settings.

### Files

- **requirements.txt**: Lists Python dependencies needed to set up the project.
- **run_ppo.py**: Script to execute training using the Proximal Policy Optimization algorithm.
- **run_sac.py**: Script to execute training using the Soft Actor-Critic algorithm.

---

## Algorithms Implemented

### Proximal Policy Optimization (PPO)
PPO is a policy gradient method designed to stabilize training and improve performance. It works by minimizing a clipped surrogate objective function, ensuring that policy updates remain within a safe range.

### Soft Actor-Critic (SAC)
SAC is an off-policy actor-critic method that incorporates entropy maximization to encourage exploration. It is particularly effective for environments with continuous action spaces.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anujpatel1761/Reinforcement_learning_final_project.git
   ```

2. Navigate to the repository:
   ```bash
   cd Reinforcement_learning_final_project
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running PPO
To train an agent using the PPO algorithm:
```bash
python run_ppo.py
```

### Running SAC
To train an agent using the SAC algorithm:
```bash
python run_sac.py
```

---

## Directory and File Overview

### **algorithms/**
Contains Python files implementing the core logic of various reinforcement learning algorithms. These implementations are modular, enabling easy integration and comparison.

### **ckpt/**
Stores saved models and training checkpoints. Use these to resume training or evaluate previously trained models.

### **data_loader/**
Includes utilities for loading datasets and preprocessing input data to suit the environments or algorithms used.

### **environments/**
Defines custom environments and wrappers for integrating standard environments, making them compatible with implemented algorithms.

### **logs/**
Holds training logs for visualizing and analyzing metrics like rewards, losses, and policy performance over time.

### **models/**
Contains the neural network architectures designed for the agents, including actor and critic networks.

### **parameters/**
Houses configuration files specifying hyperparameters like learning rate, discount factor, and batch size.

### **requirements.txt**
Lists all dependencies required for the project, ensuring compatibility and reproducibility.

### **run_ppo.py**
The main script to train agents using PPO. It initializes the environment, configures the agent, and manages the training loop.

### **run_sac.py**
The main script to train agents using SAC. Similar to `run_ppo.py`, it handles initialization, configuration, and training.

---

## Future Work

- Add support for additional algorithms, such as DDPG or A3C.
- Integrate advanced visualization tools for better insight into training progress.
- Extend the project to include multi-agent environments.
- Experiment with hyperparameter optimization techniques.

---

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code adheres to the project's style guidelines and is well-documented.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
