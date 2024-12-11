import csv
import os
import time
from collections import defaultdict
import numpy as np

class Logger:
    def __init__(self, logfile, total_episodes, name="Training"):
        print("Logger instance created.")

        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.file = open(logfile, "w", newline="")
        self.writer = csv.writer(self.file)
        
        # Comprehensive metrics list
        self.metrics_list = [
            'Episode', 
            'Total Steps', 
            'Episode Reward', 
            'Average Reward', 
            'Best Reward', 
            'Actor Loss', 
            'Critic Loss', 
            'Actor LR', 
            'Buffer Size'
        ]
        self.writer.writerow(self.metrics_list)
        
        self.total_episodes = total_episodes
        self.start_time = time.time()
        
        # Add write and print methods
        self.write = self._write_placeholder
        self.print = self._print_placeholder

    def _write_placeholder(self, *args, **kwargs):
        """
        Placeholder write method to prevent AttributeError
        This can be overridden or implemented as needed
        """
        pass

    def _print_placeholder(self, *args, **kwargs):
        """
        Placeholder print method to prevent AttributeError
        This can be printed to console or logged as needed
        """
        pass

    def log(self, episode=0, total_steps=0, episode_reward=0, avg_reward=0, 
            best_reward=0, actor_loss=0, critic_loss=0, actor_lr=0, buffer_size=0):
        """
        Log comprehensive training metrics
        """
        row = [
            episode, 
            total_steps, 
            episode_reward, 
            avg_reward, 
            best_reward, 
            actor_loss or 0, 
            critic_loss or 0, 
            actor_lr, 
            buffer_size
        ]
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        """
        Close the CSV file when done.
        """
        self.file.close()