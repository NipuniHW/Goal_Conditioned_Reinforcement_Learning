#!/usr/bin/python3.10

# import argparse
# from copy import deepcopy
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random
# import json
# from datetime import datetime
# import time
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional

# # Import custom modules (assuming these exist)
# # from configurations import _load_config, _init_spaces, _init_networks, _init_training, _init_data_tracking, _normalize_gaze_to_category, _state_to_vector
# # from reward_function import calculate_reward 
# # from train_saving_functions import *

# # Constants
# GAZE_RANGES = {
#     0: (0, 33),    # Low gaze
#     1: (34, 66),   # Medium gaze
#     2: (67, 100)   # High gaze
# }

# GOAL_NAMES = ["Low", "Medium", "High"]

# class SyntheticEnvironment:
#     """
#     Synthetic environment that simulates gaze responses based on pepper actions.
#     Models realistic human gaze behavior with noise and dynamics.
#     """
    
#     def __init__(self, noise_level=0.1, dynamics_factor=0.3):
#         self.noise_level = noise_level
#         self.dynamics_factor = dynamics_factor
        
#         # Current pepper state (tracking actual robot parameters)
#         self.pepper_state = {
#             'head_pos': 2,      # 0-4 scale
#             'nav_pos': 4,       # 0-8 scale 
#             'gesture_level': 5, # 0-10 scale
#             'volume_level': 5   # 0-10 scale
#         }
        
#         # Current gaze level
#         self.current_gaze = 50.0  # Start at medium
        
#         # Define action effects on gaze based on observed patterns
#         self.action_effects = {
#             'head_pos': {
#                 -1: {'low': +2, 'medium': -3, 'high': -1},  # Move left/down
#                 0:  {'low': 0, 'medium': 0, 'high': 0},     # No change  
#                 1:  {'low': -2, 'medium': +3, 'high': +1}   # Move right/up
#             },
#             'nav_pos': {
#                 -1: {'low': +1, 'medium': -2, 'high': -4},  # Move back
#                 0:  {'low': 0, 'medium': 0, 'high': 0},     # No change
#                 1:  {'low': -1, 'medium': +2, 'high': +4}   # Move closer
#             },
#             'gesture_level': {
#                 -1: {'low': +1, 'medium': -2, 'high': -3},  # Less gestures
#                 0:  {'low': 0, 'medium': 0, 'high': 0},     # No change
#                 1:  {'low': -1, 'medium': +2, 'high': +3}   # More gestures
#             },
#             'volume_level': {
#                 -1: {'low': +1, 'medium': -1, 'high': -2},  # Quieter
#                 0:  {'low': 0, 'medium': 0, 'high': 0},     # No change
#                 1:  {'low': -1, 'medium': +1, 'high': +2}   # Louder
#             }
#         }
    
#     def _get_gaze_category(self, gaze_value):
#         """Convert gaze value to category string."""
#         if gaze_value <= 33:
#             return 'low'
#         elif gaze_value <= 66:
#             return 'medium'
#         else:
#             return 'high'
    
#     def step(self, delta_head, delta_nav, delta_gesture, delta_volume):
#         """
#         Execute action and return new gaze value.
#         Simulates realistic gaze response based on pepper behavior changes.
#         """
#         # Update pepper state (with bounds checking)
#         self.pepper_state['head_pos'] = np.clip(
#             self.pepper_state['head_pos'] + delta_head, 0, 4
#         )
#         self.pepper_state['nav_pos'] = np.clip(
#             self.pepper_state['nav_pos'] + delta_nav, 0, 8
#         )
#         self.pepper_state['gesture_level'] = np.clip(
#             self.pepper_state['gesture_level'] + delta_gesture, 0, 10
#         )
#         self.pepper_state['volume_level'] = np.clip(
#             self.pepper_state['volume_level'] + delta_volume, 0, 10
#         )
        
#         # Calculate gaze change based on current gaze level and actions
#         current_category = self._get_gaze_category(self.current_gaze)
        
#         gaze_change = 0
#         gaze_change += self.action_effects['head_pos'][delta_head][current_category]
#         gaze_change += self.action_effects['nav_pos'][delta_nav][current_category]
#         gaze_change += self.action_effects['gesture_level'][delta_gesture][current_category]
#         gaze_change += self.action_effects['volume_level'][delta_volume][current_category]
        
#         # Add dynamics (gaze tends to drift toward baseline)
#         baseline_gaze = 50.0
#         drift = (baseline_gaze - self.current_gaze) * self.dynamics_factor * 0.1
        
#         # Add noise
#         noise = np.random.normal(0, self.noise_level * 10)
        
#         # Update gaze
#         self.current_gaze += gaze_change + drift + noise
#         self.current_gaze = np.clip(self.current_gaze, 0, 100)
        
#         return self.current_gaze
    
#     def reset(self, initial_gaze=None):
#         """Reset environment to initial state."""
#         if initial_gaze is None:
#             initial_gaze = np.random.uniform(20, 80)  # Random start
        
#         self.current_gaze = initial_gaze
        
#         # Reset pepper to neutral state
#         self.pepper_state = {
#             'head_pos': 2,
#             'nav_pos': 4, 
#             'gesture_level': 5,
#             'volume_level': 5
#         }
        
#         return self.current_gaze

# def calculate_reward(next_gaze, goal_id, previous_gaze):
#     """Calculate reward based on gaze improvement toward goal."""
#     target_min, target_max = GAZE_RANGES[goal_id]
#     target_center = (target_min + target_max) / 2
    
#     # Distance from target center
#     prev_distance = abs(previous_gaze - target_center)
#     new_distance = abs(next_gaze - target_center)
    
#     # Reward for getting closer to target
#     improvement = prev_distance - new_distance
    
#     # Bonus for being in target range
#     in_range_bonus = 5.0 if target_min <= next_gaze <= target_max else 0.0
    
#     # Penalty for being very far from target
#     distance_penalty = -0.1 * new_distance
    
#     return improvement + in_range_bonus + distance_penalty

# class SyntheticGazeGCRL:
#     """
#     GCRL system adapted for synthetic training data.
#     Same architecture as real-time version but uses synthetic environment.
#     """
    
#     def __init__(self, config: Optional[Dict] = None):
#         # Load configuration
#         if config is None:
#             config = {}
#         self.config = self._load_config(config)
        
#         # Initialize synthetic environment
#         self.env = SyntheticEnvironment(
#             noise_level=self.config.get('env_noise', 0.1),
#             dynamics_factor=self.config.get('env_dynamics', 0.3)
#         )
        
#         # Initialize state and action spaces
#         self._init_spaces()
        
#         # Initialize neural networks
#         self._init_networks()
        
#         # Initialize training components
#         self._init_training()
        
#         # Initialize data tracking
#         self._init_data_tracking()
        
#         print(f"Initialized Synthetic GazeGCRL with {self.action_dim} delta actions")
#         print(f"State space: Gaze level only (3 categories)")

#     def _load_config(self, config: Optional[Dict]) -> Dict:
#         """Load and validate configuration parameters."""
#         default_config = {
#             'learning_rate': 0.001,
#             'discount_factor': 0.95,
#             'epsilon': 1.0,
#             'epsilon_decay': 0.995,
#             'epsilon_min': 0.1,
#             'memory_size': 10000,
#             'batch_size': 32,
#             'target_update_freq': 100,
#             'network_hidden_sizes': [128, 128, 64],
#             'max_episode_steps': 50,
#             'goal_names': GOAL_NAMES,
#             'gaze_ranges': GAZE_RANGES,
#             'env_noise': 0.1,
#             'env_dynamics': 0.3,
#         }
        
#         if config:
#             default_config.update(config)
        
#         return default_config

#     def _init_spaces(self):
#         """Initialize state and action spaces."""
#         # Current state - only gaze level
#         self.current_state = {
#             'gaze': 0  # Current gaze level (0=low, 1=medium, 2=high)
#         }
        
#         # Generate delta actions: each parameter can change by -1, 0, or +1
#         self.delta_actions = [
#             (dh, dn, dg, dv) 
#             for dh in [-1, 0, 1] 
#             for dn in [-1, 0, 1] 
#             for dg in [-1, 0, 1] 
#             for dv in [-1, 0, 1]
#         ]
        
#         # Define dimensions
#         self.state_dim = 1  # Only gaze level
#         self.action_dim = len(self.delta_actions)  # 3^4 = 81 actions
#         self.goal_dim = 3   # One-hot encoding for 3 gaze levels

#     def _init_networks(self):
#         """Initialize policy and target networks."""
#         input_dim = self.state_dim + self.goal_dim
#         hidden_sizes = self.config['network_hidden_sizes']
        
#         # Build network architecture
#         layers = []
#         prev_size = input_dim
        
#         for hidden_size in hidden_sizes:
#             layers.extend([
#                 nn.Linear(prev_size, hidden_size),
#                 nn.ReLU(),
#                 nn.Dropout(0.1)
#             ])
#             prev_size = hidden_size
        
#         layers.append(nn.Linear(prev_size, self.action_dim))
        
#         self.policy_net = nn.Sequential(*layers)
#         self.target_net = nn.Sequential(*layers)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
        
#         # Initialize optimizer
#         self.optimizer = torch.optim.Adam(
#             self.policy_net.parameters(), 
#             lr=self.config['learning_rate']
#         )

#     def _init_training(self):
#         """Initialize training-related variables."""
#         self.memory = deque(maxlen=self.config['memory_size'])
#         self.step_count = 0
#         self.episode_count = 0
#         self.epsilon = self.config['epsilon']

#     def _init_data_tracking(self):
#         """Initialize data tracking for training metrics."""
#         self.training_data = {
#             'episode_rewards': {0: [], 1: [], 2: []},
#             'success_rates': {0: [], 1: [], 2: []},
#             'episode_lengths': {0: [], 1: [], 2: []},
#             'loss_history': [],
#             'epsilon_history': [],
#             'q_value_stats': [],
#             'gaze_trajectories': [],
#             'action_distributions': {i: 0 for i in range(self.action_dim)},
#             'training_config': self.config.copy(),
#             'timestamps': []
#         }

#     def _normalize_gaze_to_category(self, gaze_value: float) -> int:
#         """Convert continuous gaze value to categorical (0, 1, 2)."""
#         if gaze_value <= 33:
#             return 0
#         elif gaze_value <= 66:
#             return 1
#         else:
#             return 2

#     def _state_to_vector(self, state: Dict) -> np.ndarray:
#         """Convert state dictionary to vector."""
#         return np.array([state['gaze']], dtype=np.float32)

#     def _create_network_input(self, state: Dict, goal_id: int) -> torch.Tensor:
#         """Create neural network input from state and goal."""
#         # Convert state to vector
#         state_vector = self._state_to_vector(state)
        
#         # Create one-hot goal vector
#         goal_vector = np.zeros(self.goal_dim, dtype=np.float32)
#         goal_vector[goal_id] = 1.0
        
#         # Concatenate and convert to tensor
#         input_vector = np.concatenate([state_vector, goal_vector])
#         return torch.FloatTensor(input_vector).unsqueeze(0)

#     def update_state(self, current_gaze: float) -> Dict:
#         """Update state based on current gaze value."""
#         new_state = {
#             'gaze': self._normalize_gaze_to_category(current_gaze)
#         }
        
#         # Update current state
#         self.current_state = new_state
#         return new_state

#     def select_action(self, state: Dict, goal_id: int, training: bool = True) -> Tuple[int, int, int, int, int]:
#         """Select action using epsilon-greedy policy."""
#         if training and random.random() < self.epsilon:
#             action_id = random.randint(0, self.action_dim - 1)
#         else:
#             with torch.no_grad():
#                 input_tensor = self._create_network_input(state, goal_id)
#                 q_values = self.policy_net(input_tensor)
#                 action_id = q_values.argmax().item()
        
#         delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[action_id]

#         # Track action distribution
#         self.training_data['action_distributions'][action_id] += 1
#         return action_id, delta_head, delta_nav, delta_gesture, delta_volume

#     def store_experience(self, state: Dict, action: int, next_state: Dict, 
#                         reward: float, done: bool, goal_id: int):
#         """Store experience in replay buffer."""
#         experience = (
#             self._state_to_vector(state),
#             action,
#             self._state_to_vector(next_state),
#             reward,
#             done,
#             goal_id
#         )
#         self.memory.append(experience)

#     def train_step(self) -> Optional[float]:
#         """Perform one training step and return loss."""
#         if len(self.memory) < self.config['batch_size']:
#             return None
        
#         # Sample batch
#         batch = random.sample(self.memory, self.config['batch_size'])
#         states, actions, next_states, rewards, dones, goals = zip(*batch)
        
#         # Convert to tensors
#         batch_states = []
#         batch_next_states = []
        
#         for i in range(self.config['batch_size']):
#             # Convert arrays back to state dicts for network input
#             state_dict = {
#                 'gaze': int(states[i][0])
#             }
            
#             next_state_dict = {
#                 'gaze': int(next_states[i][0])
#             }
            
#             batch_states.append(self._create_network_input(state_dict, goals[i]))
#             batch_next_states.append(self._create_network_input(next_state_dict, goals[i]))
        
#         # Stack into batch tensors
#         batch_states = torch.cat(batch_states, dim=0)
#         batch_next_states = torch.cat(batch_next_states, dim=0)
#         batch_actions = torch.LongTensor(actions)
#         batch_rewards = torch.FloatTensor(rewards)
#         batch_dones = torch.BoolTensor(dones)
        
#         # Get current Q-values
#         current_q_values = self.policy_net(batch_states).gather(1, batch_actions.unsqueeze(1))
        
#         # Get target Q-values
#         with torch.no_grad():
#             next_q_values = self.target_net(batch_next_states).max(1)[0]
#             target_q_values = batch_rewards + (self.config['discount_factor'] * next_q_values * (~batch_dones))
        
#         # Compute loss
#         loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
#         # Optimize
#         self.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
#         self.optimizer.step()
        
#         # Update epsilon
#         if self.epsilon > self.config['epsilon_min']:
#             self.epsilon *= self.config['epsilon_decay']
        
#         # Update target network
#         self.step_count += 1
#         if self.step_count % self.config['target_update_freq'] == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())
        
#         return loss.item()

#     def run_synthetic_episode(self, goal_id: int, episode_count: int) -> Tuple[float, float, int, List[float]]:
#         """Run one synthetic training episode."""
        
#         # Reset environment
#         initial_gaze = self.env.reset()
#         current_gaze = initial_gaze
        
#         # Initialize tracking
#         episode_reward = 0
#         gaze_trajectory = [current_gaze]
#         step = 0
#         done = False
        
#         # Initialize state
#         self.current_state = {
#             'gaze': self._normalize_gaze_to_category(current_gaze)
#         }
        
#         print(f"Episode {episode_count} - Goal: {GOAL_NAMES[goal_id]} - Initial gaze: {current_gaze:.2f}")
        
#         while step < self.config['max_episode_steps'] and not done:
#             # Current state
#             state = self.current_state.copy()
            
#             # Select and execute action
#             action, delta_head, delta_nav, delta_gesture, delta_volume = self.select_action(state, goal_id)
            
#             # Execute action in synthetic environment
#             next_gaze = self.env.step(delta_head, delta_nav, delta_gesture, delta_volume)
#             next_state = self.update_state(next_gaze)
            
#             # Calculate reward
#             reward = calculate_reward(next_gaze, goal_id, current_gaze)
#             episode_reward += reward
            
#             # Check if goal achieved
#             target_min, target_max = GAZE_RANGES[goal_id]
#             if target_min <= next_gaze <= target_max:
#                 done = True
#                 reward += 10  # Bonus for achieving goal
            
#             # Store experience
#             self.store_experience(state, action, next_state, reward, done, goal_id)
            
#             # Train network
#             loss = self.train_step()
#             if loss is not None:
#                 self.training_data['loss_history'].append(loss)
            
#             # Update for next step
#             self.current_state = next_state
#             current_gaze = next_gaze
#             gaze_trajectory.append(current_gaze)
#             step += 1
            
#             # Print progress every 10 steps
#             if step % 10 == 0:
#                 print(f"  Step {step}: Gaze {current_gaze:.1f} -> {next_gaze:.1f}, Reward: {reward:.2f}")
        
#         # Record episode metrics
#         self.training_data['episode_rewards'][goal_id].append(episode_reward)
#         self.training_data['episode_lengths'][goal_id].append(step)
#         self.training_data['gaze_trajectories'].append(gaze_trajectory)
#         self.training_data['success_rates'][goal_id].append(1.0 if done else 0.0)
#         self.training_data['epsilon_history'].append(self.epsilon)
        
#         print(f"Episode {episode_count} completed: Reward={episode_reward:.2f}, Steps={step}, Final gaze={current_gaze:.2f}")
        
#         return episode_reward, current_gaze, step, gaze_trajectory

#     def save_model(self, filepath: str):
#         """Save the trained model."""
#         torch.save({
#             'policy_net_state_dict': self.policy_net.state_dict(),
#             'target_net_state_dict': self.target_net.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'epsilon': self.epsilon,
#             'step_count': self.step_count,
#             'episode_count': self.episode_count,
#             'config': self.config
#         }, filepath)

#     def load_model(self, filepath: str):
#         """Load a trained model."""
#         checkpoint = torch.load(filepath)
#         self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
#         self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.epsilon = checkpoint['epsilon']
#         self.step_count = checkpoint['step_count']
#         self.episode_count = checkpoint['episode_count']
#         self.config = checkpoint['config']

#     def save_training_data(self, filepath: str):
#         """Save training data to JSON."""
#         # Convert numpy arrays to lists for JSON serialization
#         data_to_save = deepcopy(self.training_data)
#         with open(filepath, 'w') as f:
#             json.dump(data_to_save, f, indent=2)

#     def print_policy_summary(self, episode_num: int, save_dir: str = "policy_summary_logs"):
#         """Print and save a concise summary of the best action for each gaze state across all goals."""
        
#         # Ensure the output directory exists
#         os.makedirs(save_dir, exist_ok=True)
#         filename = os.path.join(save_dir, f"policy_summary_episode_{episode_num}.txt")
        
#         gaze_names = ['L', 'M', 'H']  # Low, Medium, High
#         goal_names = ['Low', 'Medium', 'High']
        
#         with open(filename, 'w') as f:
#             header = f"=== Policy Summary - Episode {episode_num} ===\n"
#             header += f"Epsilon: {self.epsilon:.4f}\n\n"
#             print(header.strip())
#             f.write(header)
            
#             # For each goal
#             for goal_id in range(3):
#                 goal_line = f"Goal = {goal_names[goal_id]} ({goal_id})\n"
#                 print(goal_line.strip())
#                 f.write(goal_line)
                
#                 # For each gaze state
#                 for gaze in range(3):
#                     state = {'gaze': gaze}
                    
#                     # Get Q-values for this state-goal combination
#                     input_tensor = self._create_network_input(state, goal_id)
#                     with torch.no_grad():
#                         q_values = self.policy_net(input_tensor).squeeze().numpy()
                    
#                     # Find best action
#                     best_action_id = np.argmax(q_values)
#                     delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[best_action_id]
                    
#                     # Format the line: "L +1 +1 +1 +1"
#                     action_line = f"{gaze_names[gaze]} {delta_head:+d} {delta_nav:+d} {delta_gesture:+d} {delta_volume:+d}\n"
#                     print(action_line.strip())
#                     f.write(action_line)
                
#                 # Add blank line between goals
#                 print()
#                 f.write("\n")

#     def analyze_convergence(self):
#         """Analyze training convergence across all goals."""
#         print("\n=== Training Convergence Analysis ===")
        
#         for goal_id in range(3):
#             rewards = self.training_data['episode_rewards'][goal_id]
#             success_rates = self.training_data['success_rates'][goal_id]
            
#             if len(rewards) > 0:
#                 recent_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
#                 recent_success = success_rates[-10:] if len(success_rates) >= 10 else success_rates
                
#                 print(f"\nGoal {goal_id} ({GOAL_NAMES[goal_id]}):")
#                 print(f"  Episodes trained: {len(rewards)}")
#                 print(f"  Average reward (last 10): {np.mean(recent_rewards):.2f}")
#                 print(f"  Success rate (last 10): {np.mean(recent_success)*100:.1f}%")
#                 print(f"  Total experiences: {len([x for x in self.memory if x[5] == goal_id])}")

# def save_training_state_after_episode(gcrl_model, episode_count: int, training_runname: str):
#     """Save training state after episode."""
#     if not os.path.exists(training_runname):
#         os.makedirs(training_runname)
    
#     # Save model
#     model_path = os.path.join(training_runname, f"model_episode_{episode_count}.pt")
#     gcrl_model.save_model(model_path)
    
#     # Save training data
#     data_path = os.path.join(training_runname, f"training_data_episode_{episode_count}.json")
#     gcrl_model.save_training_data(data_path)
    
#     print(f"Saved training state after episode {episode_count}")

# if __name__ == "__main__":
#     # Training configuration
#     training_config = {
#         'learning_rate': 0.001,
#         'epsilon_decay': 0.995,
#         'max_episode_steps': 30,
#         'batch_size': 64,
#         'memory_size': 20000,
#         'target_update_freq': 50,
#         'env_noise': 0.15,  # Slightly higher noise for robustness
#         'env_dynamics': 0.2,
#     }
    
#     # Initialize synthetic GCRL system
#     gcrl_model = SyntheticGazeGCRL(config=training_config)
    
#     # Training parameters
#     training_runname = 'gcrl_training'
#     episodes_per_goal = 100  # Train each goal extensively
    
#     # Create training directory
#     if not os.path.exists(training_runname):
#         os.makedirs(training_runname)
    
#     print("Starting synthetic GCRL training...")
#     print(f"Training {episodes_per_goal} episodes per goal")
    
#     episode_count = 0
    
#     # Train each goal sequentially
#     for goal_id in range(3):
#         print(f"\n{'='*50}")
#         print(f"Training Goal {goal_id}: {GOAL_NAMES[goal_id]}")
#         print(f"{'='*50}")
        
#         for episode in range(episodes_per_goal):
#             episode_count += 1
            
#             # Run episode
#             episode_reward, final_gaze, steps, gaze_trajectory = gcrl_model.run_synthetic_episode(
#                 goal_id, episode_count
#             )
            
#             # Save periodically and at end of goal training
#             if episode % 20 == 0 or episode == episodes_per_goal - 1:
#                 save_training_state_after_episode(gcrl_model, episode_count, training_runname)
#                 gcrl_model.print_policy_summary(episode_count)
#                 gcrl_model.analyze_convergence()
    
#     print("\n" + "="*60)
#     print("SYNTHETIC TRAINING COMPLETED")
#     print("="*60)
    
#     # Final analysis
#     gcrl_model.analyze_convergence()
#     gcrl_model.print_policy_summary(episode_count, save_dir=training_runname)
    
#     # Save final model
#     final_model_path = os.path.join(training_runname, "final_model.pt")
#     gcrl_model.save_model(final_model_path)
    
#     print(f"\nFinal model saved to: {final_model_path}")
#     print(f"Training data saved in: {training_runname}")
#     print("\nTraining complete! The model is ready for deployment.")

# import torch
# import torch.nn.functional as F
# import csv
# import random
# import numpy as np
# from collections import deque
# from model import DQN  # Your model definition
# from entropy_controller import encode_state  # You already have this
# import os

# # Configuration
# DATA_PATH = "synthetic_dataset.csv"
# BATCH_SIZE = 32
# EPOCHS = 20
# GAMMA = 0.99
# LEARNING_RATE = 1e-3
# REPLAY_BUFFER_SIZE = 1000
# TARGET_UPDATE_FREQ = 5

# # Model & Optimizer setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# policy_net = DQN().to(device)
# target_net = DQN().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
# optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

# # Load dataset
# def load_dataset(filepath):
#     data = []
#     with open(filepath, mode="r") as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             transition = {
#                 "goal_id": int(row["goal_id"]),
#                 "current_gaze": int(row["current_gaze"]),
#                 "action_id": int(row["action_id"]),
#                 "next_gaze": float(row["next_gaze"]),
#                 "reward": float(row["reward"])
#             }
#             data.append(transition)
#     return data

# dataset = load_dataset(DATA_PATH)

# # Replay buffer
# replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

# for sample in dataset:
#     state = encode_state(sample["goal_id"], sample["current_gaze"])
#     next_state = encode_state(sample["goal_id"], sample["next_gaze"])
#     transition = (state, sample["action_id"], sample["reward"], next_state)
#     replay_buffer.append(transition)

# # Training loop
# def train():
#     for epoch in range(EPOCHS):
#         random.shuffle(replay_buffer)
#         batch_losses = []

#         for i in range(0, len(replay_buffer), BATCH_SIZE):
#             batch = list(replay_buffer)[i:i + BATCH_SIZE]

#             states, actions, rewards, next_states = zip(*batch)

#             states_tensor = torch.FloatTensor(states).to(device)
#             actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
#             rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
#             next_states_tensor = torch.FloatTensor(next_states).to(device)

#             # Q(s, a)
#             q_values = policy_net(states_tensor).gather(1, actions_tensor)

#             # max_a' Q_target(s', a')
#             with torch.no_grad():
#                 next_q_values = target_net(next_states_tensor).max(1)[0].unsqueeze(1)

#             target = rewards_tensor + GAMMA * next_q_values
#             loss = F.mse_loss(q_values, target)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             batch_losses.append(loss.item())

#         # Sync target network
#         if (epoch + 1) % TARGET_UPDATE_FREQ == 0:
#             target_net.load_state_dict(policy_net.state_dict())

#         print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {np.mean(batch_losses):.4f}")

#     print("Training completed.")

# # Save model
# def save_model(filepath):
#     torch.save({
#         'policy_net_state_dict': policy_net.state_dict(),
#         'target_net_state_dict': target_net.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }, filepath)
#     print(f"Model saved to {filepath}")

# if __name__ == "__main__":
#     train()
#     save_model("trained_gcrl_synthetic.pt")

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os

# # Config
# BATCH_SIZE = 64
# EPOCHS = 30
# LEARNING_RATE = 1e-3
# SEED = 42
# MODEL_SAVE_PATH = "new_trained_action_model.pth"

# torch.manual_seed(SEED)

# # Dataset class
# class GazeDataset(Dataset):
#     def __init__(self, dataframe):
#         self.features = dataframe[['Goal Id', 'Current Gaze', 'dh', 'dn', 'dg', 'dv']].values.astype(np.float32)
#         self.labels = dataframe['action_id'].values.astype(np.int64)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

# # Model
# class ActionPredictor(nn.Module):
#     def __init__(self):
#         super(ActionPredictor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(6, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 81)  # 81 possible action_ids
#         )

#     def forward(self, x):
#         return self.net(x)

# def save_model(model, optimizer, epoch, path):
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'config': {
#             'batch_size': BATCH_SIZE,
#             'learning_rate': LEARNING_RATE
#         }
#     }, path)
#     print(f"✅ Model saved to {path}")

# # Training loop
# def train():
#     # Load data
#     df = pd.read_csv("/home/vscode/gaze_ws/L2CS-Net/dataset.csv")
#     train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)

#     train_dataset = GazeDataset(train_df)
#     val_dataset = GazeDataset(val_df)

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

#     model = ActionPredictor()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         total_loss = 0
#         correct = 0
#         total = 0

#         for inputs, targets in train_loader:
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == targets).sum().item()
#             total += targets.size(0)

#         train_acc = 100 * correct / total
#         print(f"Epoch [{epoch}/{EPOCHS}] Loss: {total_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

#         # Validation
#         model.eval()
#         val_correct = 0
#         val_total = 0
#         with torch.no_grad():
#             for inputs, targets in val_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 val_correct += (predicted == targets).sum().item()
#                 val_total += targets.size(0)

#         val_acc = 100 * val_correct / val_total
#         print(f"Validation Accuracy: {val_acc:.2f}%\n")

#         # Save checkpoint at the end of training
#         if epoch == EPOCHS:
#             save_model(model, optimizer, epoch, MODEL_SAVE_PATH)

# if __name__ == "__main__":
#     train()

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import numpy as np
import os
from collections import deque

# GCRL Config
config = {
    'gamma': 0.99,
    'lr': 1e-3,
    'batch_size': 64,
    'target_update': 10,
    'memory_size': 10000,
    'epochs': 100,
    'model_path': './gcrl_offline_model.pth'
}

# State: gaze (0=low, 1=medium, 2=high)
# Action: tuple (dh, dn, dg, dv)
action_space = [(dh, dn, dg, dv) for dh in [-1, 0, 1] for dn in [-1, 0, 1] for dg in [-1, 0, 1] for dv in [-1, 0, 1]]

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class GCRLAgent:
    def __init__(self, config):
        self.config = config
        self.policy_net = QNetwork(input_dim=4, output_dim=len(action_space))
        self.target_net = QNetwork(input_dim=4, output_dim=len(action_space))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr'])
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=config['memory_size'])
        self.epsilon = 1.0
        self.step_count = 0
        self.episode_count = 0

    def save_model(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)

    def train(self):
        if len(self.memory) < self.config['batch_size']:
            return

        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states).detach()
        target_q = q_values.clone()

        for i in range(self.config['batch_size']):
            target = rewards[i] + self.config['gamma'] * torch.max(next_q_values[i])
            target_q[i][actions[i]] = target

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(action_space)-1)
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

# Load dataset
csv_path = '/home/vscode/gaze_ws/L2CS-Net/synthetic_gaze_dataset.csv'
df = pd.read_csv(csv_path)

def encode_state(gaze, goal):
    return [gaze, goal, abs(gaze - goal), int(gaze == goal)]

# Initialize agent
agent = GCRLAgent(config)

# Populate memory
for _, row in df.iterrows():
    state = encode_state(row['state'], row['goal'])
    next_state = encode_state(row['next_state'], row['goal'])
    action_idx = action_space.index(eval(row['action']))
    agent.memory.append((state, action_idx, row['reward'], next_state))

# Offline training
for epoch in range(config['epochs']):
    agent.train()
    if epoch % config['target_update'] == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
    agent.epsilon = max(0.1, agent.epsilon * 0.95)
    agent.episode_count += 1
    print(f"Epoch {epoch+1}/{config['epochs']}, Epsilon: {agent.epsilon:.2f}")

# Save model
agent.save_model(config['model_path'])
print(f"Model saved to {config['model_path']}")
