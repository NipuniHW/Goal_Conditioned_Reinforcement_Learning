#!/usr/bin/python3.10

# Class constants
from collections import deque
from typing import Dict, Optional
import numpy as np
from torch import nn
import torch


GAZE_RANGES = {
    0: (0, 33),    # Low gaze
    1: (34, 66),   # Medium gaze
    2: (67, 100)   # High gaze
}

GOAL_NAMES = ["Low", "Medium", "High"]

def _load_config( config: Optional[Dict]) -> Dict:
        """Load and validate configuration parameters."""
        default_config = {
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.1,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100,
            'network_hidden_sizes': [128, 128, 64],
            'max_episode_steps': 50,
            'camera_id': 2,
            'goal_names': GOAL_NAMES,
            'gaze_ranges': GAZE_RANGES,
        }
        
        if config:
            default_config.update(config)
        
        return default_config

def _init_spaces(self):
    """Initialize state and action spaces."""
    # Current Pepper state
    self.current_pepper_state = {
        'gaze': 0,           # Current gaze level (0-2)
        'head_position': 0,  # 0-4 (0=down, 1=left, 2=center, 3=right, 4=up)
        'navigation': 0,     # 0-4 (0=back more, 1=back, 2=still, 3=closer, 4=much closer)
        'gesture_level': 0,  # 0-10 (gesture intensity)
        'volume_level': 0    # 0-10 (volume level)
    }
    
    # Generate delta actions: each parameter can change by -1, 0, or +1
    self.delta_actions = [
        (dh, dn, dg, dv) 
        for dh in [-1, 0, 1] 
        for dn in [-1, 0, 1] 
        for dg in [-1, 0, 1] 
        for dv in [-1, 0, 1]
    ]
    
    # Define dimensions
    self.state_dim = 5  # [gaze, head_pos, nav_pos, gesture_level, volume_level]
    self.action_dim = len(self.delta_actions)  # 3^4 = 81 actions
    self.goal_dim = 3   # One-hot encoding for 3 gaze levels

def _init_networks(self):
    """Initialize policy and target networks."""
    input_dim = self.state_dim + self.goal_dim
    hidden_sizes = self.config['network_hidden_sizes']
    
    # Build network architecture
    layers = []
    prev_size = input_dim
    
    for hidden_size in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        prev_size = hidden_size
    
    layers.append(nn.Linear(prev_size, self.action_dim))
    
    self.policy_net = nn.Sequential(*layers)
    self.target_net = nn.Sequential(*layers)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    
    # Initialize optimizer
    self.optimizer = torch.optim.Adam(
        self.policy_net.parameters(), 
        lr=self.config['learning_rate']
    )

def _init_training(self):
    """Initialize training-related variables."""
    self.memory = deque(maxlen=self.config['memory_size'])
    self.step_count = 0
    self.episode_count = 0
    self.epsilon = self.config['epsilon']

def _init_data_tracking(self):
    """Initialize data tracking for training metrics."""
    self.training_data = {
        'episode_rewards': {0: [], 1: [], 2: []},
        'success_rates': {0: [], 1: [], 2: []},
        'episode_lengths': {0: [], 1: [], 2: []},
        'loss_history': [],
        'epsilon_history': [],
        'q_value_stats': [],
        'gaze_trajectories': [],
        'action_distributions': {i: 0 for i in range(self.action_dim)},
        'training_config': self.config.copy(),
        'timestamps': []
    }

def _normalize_gaze_to_category( gaze_value: float) -> int:
    """Convert continuous gaze value to categorical (0, 1, 2)."""
    if gaze_value <= 33:
        return 0
    elif gaze_value <= 66:
        return 1
    else:
        return 2

def _state_to_vector(state: Dict) -> np.ndarray:
    """Convert state dictionary to vector."""
    return np.array([
        state['gaze'],
        state['head_position'],
        state['navigation'],
        state['gesture_level'],
        state['volume_level']
    ], dtype=np.float32)

class GazeFormulationBaseClass:
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)
            
        if 'states_generator' not in config_dict:
            self.states_generator = None
        else:
            self.states_generator = config_dict['states_generator']
            self.states = self.states_generator()

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = GazeFormulationBaseClass(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, GazeFormulationBaseClass):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)
    
    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)
