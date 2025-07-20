#!/usr/bin/python3.10

import json
from pathlib import Path
import numpy as np
import torch
import yaml


def save_trajectory_ep_to_yaml(episode_count, training_runname, training_dict):
        save_path = f'{training_runname}/{training_runname}_{episode_count}_training_data.yaml'
        # Create directory if it doesn't exist
        save_dir = Path(training_runname)
        save_dir.mkdir(exist_ok=True)

        with open(save_path, 'w') as file:
            yaml.dump(training_dict, file)
        return

def save_model(self, filepath: str):
    """Save the trained model."""
    checkpoint = {
        'policy_net_state_dict': self.policy_net.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epsilon': self.epsilon,
        'step_count': self.step_count,
        'episode_count': self.episode_count,
        'current_pepper_state': self.current_pepper_state,
        'config': self.config,
        'training_data_summary': {
            'total_episodes': self.episode_count,
            'final_success_rates': {g: self.training_data['success_rates'][g][-1] 
                                    if self.training_data['success_rates'][g] else 0 
                                    for g in range(3)}
        }
    }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(self, filepath: str):
    """Load a trained model."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.epsilon = checkpoint['epsilon']
    self.step_count = checkpoint['step_count']
    self.episode_count = checkpoint.get('episode_count', 0)
    
    if 'current_pepper_state' in checkpoint:
        self.current_pepper_state = checkpoint['current_pepper_state']
    
    print(f"Model loaded from {filepath}")

def save_training_data(self, filepath: str):
    """Save training data and metrics."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_data = {}
    for key, value in self.training_data.items():
        if isinstance(value, dict):
            serializable_data[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist() 
                                    for k, v in value.items()}
        elif isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        else:
            serializable_data[key] = value
    
    # Save as JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    print(f"Training data saved to {filepath}")

def load_training_data(self, filepath: str):
    """Load training data from file."""
    with open(filepath, 'r') as f:
        self.training_data = json.load(f)
    print(f"Training data loaded from {filepath}")