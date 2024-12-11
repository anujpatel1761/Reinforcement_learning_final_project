# logger.py
import csv
from rich.console import Console
import os

console = Console()

class Logger:
    def __init__(self, logfile):
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        self.tracks = {}
        self.file = open(logfile, "w")
        self.writer = csv.writer(self.file)
        
        # Initialize metrics
        self.metrics = {
            'episode_length': [],
            'average_reward': [],
            'success_rate': [],
            'learning_rate': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'exploration_rate': []
        }
    
    def close(self):
        self.file.close()
    
    def log(self, name, value):
        if name not in self.tracks:
            self.tracks[name] = []
        self.tracks[name].append(value)
    
    def print(self, title=None):
        if title is not None:
            console.rule(title)
        for name, track in self.tracks.items():
            console.print(f"{name}: {track[-1]}")
    
    def write(self):
        if self.file.tell() == 0:
            self.writer.writerow(list(self.tracks.keys()))
        row = [track[-1] for track in self.tracks.values()]
        self.writer.writerow(row)
        self.file.flush()
        
    def log_metrics(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.tracks:
                self.tracks[key] = []
            self.tracks[key].append(value)
