#!/usr/bin/python3.10

import argparse
from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
from datetime import datetime
import time
import os
from collections import deque
from typing import Dict, List, Tuple, Optional

# Import custom modules
from configurations import _load_config, _init_spaces, _init_networks, _init_training, _init_data_tracking, _normalize_gaze_to_category, _state_to_vector
from pepper import Pepper
from entropy_interfacing import GazeInterfaceController
from configurations import * 
from reward_function import calculate_reward 
from train_saving_functions import *
from visualizations import *

class GazeGCRL:
    # """
    # Goal-Conditioned Reinforcement Learning for Pepper Robot Gaze Control
    
    # This system uses delta-based actions to control Pepper's behavior and 
    # optimize human gaze levels for different target ranges:
    # - Low gaze: 0-33 (goal_id=0)
    # - Medium gaze: 34-66 (goal_id=1) 
    # - High gaze: 67-100 (goal_id=2)
    
    # State space: Only gaze level (0=low, 1=medium, 2=high)
    # Action space: Delta changes for head, navigation, gesture, volume
    # """
    
    def __init__(self, config: Optional[Dict] = None):
        # """Initialize the GCRL system with configurable parameters."""

        # Load configuration - provide default config if none given
        if config is None:
            config = {}
        self.config = self._load_config(config)
        
        # Initialize state and action spaces
        self._init_spaces()
        
        # Initialize neural networks
        self._init_networks()
        
        # Initialize training components
        self._init_training()
        
        # Initialize data tracking
        self._init_data_tracking()
        
        print(f"Initialized GazeGCRL with {self.action_dim} delta actions")
        print(f"State space: Gaze level only (3 categories)")

    def _load_config(self, config: Optional[Dict]) -> Dict:
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
        # Current state - only gaze level
        self.current_state = {
            'gaze': 0  # Current gaze level (0=low, 1=medium, 2=high)
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
        self.state_dim = 1  # Only gaze level
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

    def _normalize_gaze_to_category(self, gaze_value: float) -> int:
        """Convert continuous gaze value to categorical (0, 1, 2)."""
        if gaze_value <= 33:
            return 0
        elif gaze_value <= 66:
            return 1
        else:
            return 2

    def _state_to_vector(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to vector."""
        return np.array([state['gaze']], dtype=np.float32)

    def _create_network_input(self, state: Dict, goal_id: int) -> torch.Tensor:
        """Create neural network input from state and goal."""
        # Convert state to vector
        state_vector = self._state_to_vector(state)
        
        # Create one-hot goal vector
        goal_vector = np.zeros(self.goal_dim, dtype=np.float32)
        goal_vector[goal_id] = 1.0
        
        # Concatenate and convert to tensor
        input_vector = np.concatenate([state_vector, goal_vector])
        return torch.FloatTensor(input_vector).unsqueeze(0)

    def update_state(self, current_gaze: float) -> Dict:
        """Update state based on current gaze value."""
        new_state = {
            'gaze': self._normalize_gaze_to_category(current_gaze)
        }
        
        # Update current state
        self.current_state = new_state
        return new_state

    def select_action(self, state: Dict, goal_id: int, training: bool = False) -> Tuple[int, int, int, int, int]:
        """Select action using epsilon-greedy policy."""
        # if training and random.random() < self.epsilon:
        #     # action_id = 40
        #     action_id = random.randint(0, self.action_dim - 1)
        # else:
        with torch.no_grad():
            input_tensor = self._create_network_input(state, goal_id)
            q_values = self.policy_net(input_tensor)
            action_id = q_values.argmax().item()
        
        delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[action_id]

        # Track action distribution
        self.training_data['action_distributions'][action_id] += 1
        return action_id, delta_head, delta_nav, delta_gesture, delta_volume

    def store_experience(self, state: Dict, action: int, next_state: Dict, 
                        reward: float, done: bool, goal_id: int):
        """Store experience in replay buffer."""
        experience = (
            self._state_to_vector(state),
            action,
            self._state_to_vector(next_state),
            reward,
            done,
            goal_id
        )
        self.memory.append(experience)

    def train_step(self) -> Optional[float]:
        """Perform one training step and return loss."""
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, next_states, rewards, dones, goals = zip(*batch)
        
        # Convert to tensors
        batch_states = []
        batch_next_states = []
        
        for i in range(self.config['batch_size']):
            # Convert arrays back to state dicts for network input
            state_dict = {
                'gaze': int(states[i][0])
            }
            
            next_state_dict = {
                'gaze': int(next_states[i][0])
            }
            
            batch_states.append(self._create_network_input(state_dict, goals[i]))
            batch_next_states.append(self._create_network_input(next_state_dict, goals[i]))
        
        # Stack into batch tensors
        batch_states = torch.cat(batch_states, dim=0)
        batch_next_states = torch.cat(batch_next_states, dim=0)
        batch_actions = torch.LongTensor(actions)
        batch_rewards = torch.FloatTensor(rewards)
        batch_dones = torch.BoolTensor(dones)
        
        # Get current Q-values
        current_q_values = self.policy_net(batch_states).gather(1, batch_actions.unsqueeze(1))
        
        # Get target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(batch_next_states).max(1)[0]
            target_q_values = batch_rewards + (self.config['discount_factor'] * next_q_values * (~batch_dones))
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon *= self.config['epsilon_decay']
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def run_training_episode(self, goal_id: int, episode_count: int, online_episode_duration: int, training_runname: str) -> Tuple[float, float, int, List[float]]:
        """Train one episode and return metrics"""
        print('connecting a session to pepper')
        
        pepper = Pepper()
        # pepper.connect("172.20.10.4", 9559) # physical pepper
        # # pepper.connect("localhost", 44889)    # virtual pepper

        # pepper.execute_animation("action_gcrl/neutral")

        # Change the camera ID to 2 if using external usb webcam, 0 if using the laptop webcam
        controller = GazeInterfaceController(camera_id=0)
        time.sleep(1)
        # ask the user to press enter to start a calibration
        print('Press Enter to start the calibration')
        input()
        controller.calibration_exe()
        controller.start_detecting_attention()

        time.sleep(1)
        print('Press Enter to start the training')
        input()
        # start the training
        current_time = time.time()

        #Convert to minutes
        online_episode_duration_seconds = online_episode_duration
        online_episodes_duration_minutes = online_episode_duration*60
        time_step_count = 0
        save_dictionary = {}

        # Initialize variables
        episode_reward = 0
        gaze_trajectory = []
        step = 0
        done = False
        
        # Initialize Pepper control variables (these track the actual robot state)
        current_navigation = 4
        current_head_mov = 2
        current_gesture = 5
        current_volume = 5
        
        # Get initial gaze and initialize state
        initial_gaze = controller.get_gaze_score()
        current_gaze = initial_gaze
        gaze_trajectory = [current_gaze]
        
        # Initialize state (only gaze level)
        self.current_state = {
            'gaze': self._normalize_gaze_to_category(current_gaze)
        }
        
        # Reset Pepper to neutral state
        # pepper.execute_animation("action_gcrl/neutral")
        
        start_time_inner_loop = time.time()

        while time.time() - current_time < online_episodes_duration_minutes:
            frame = controller.get_visualisation_frame()
            if frame is not None:
                f = deepcopy(frame)
                cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            # Similar timing structure to Q-learning (5 second intervals)
            if time.time() - start_time_inner_loop >= 5:
                start_time_inner_loop = time.time()
                            
                # Training step
                if step < self.config['max_episode_steps']:

                    # Get the current gaze score
                    gaze_score = controller.get_gaze_score()
                    print(f"Current gaze score: {gaze_score:.2f} -- goal_id: {goal_id}")
                    
                    # Update gaze trajectory
                    gaze_trajectory.append(gaze_score)
                
                    # Current state is just the gaze level
                    state = self.current_state.copy()
                    print(f"Current state: {state}")

                    # Select and execute action
                    action, delta_head, delta_nav, delta_gesture, delta_volume = self.select_action(state, goal_id)
                    print(f"Selected action {action}: delta_head={delta_head}, delta_nav={delta_nav}, delta_gesture={delta_gesture}, delta_volume={delta_volume}")

                    # Execute action on Pepper
                    current_head_mov, current_navigation, current_gesture, current_volume = pepper.update_behavior(
                        delta_head, delta_nav, delta_gesture, delta_volume, 
                        current_head_mov, current_navigation, current_gesture, current_volume
                    )
                    
                    # Debug print to verify updates
                    print(f"Updated Pepper state: head={current_head_mov}, nav={current_navigation}, gesture={current_gesture}, volume={current_volume}")
                    
                    #Wait for 3s (same as Q-learning)
                    time.sleep(3)

                    # Get the current gaze score after action
                    next_gaze = controller.get_gaze_score()
                    next_state = self.update_state(next_gaze)
                    print(f"Next state: {next_state}")

                    # Calculate reward
                    reward = calculate_reward(next_gaze, goal_id, gaze_score)
                    episode_reward += reward
                    
                    # Store experience
                    self.store_experience(state, action, next_state, reward, done, goal_id)
                    
                    # Train network
                    loss = self.train_step()
                    if loss is not None:
                        self.training_data['loss_history'].append(loss)
                    
                    # Save data similar to Q-learning format
                    save_dictionary[f'state_episode_{episode_count}_timestep_{time_step_count}'] = state
                    save_dictionary[f'nextstate_episode_{episode_count}_timestep_{time_step_count}'] = next_state
                    save_dictionary[f'action_episode_{episode_count}_timestep_{time_step_count}'] = action
                    save_dictionary[f'action_deltas_episode_{episode_count}_timestep_{time_step_count}'] = (delta_head, delta_nav, delta_gesture, delta_volume)
                    save_dictionary[f'pepper_state_episode_{episode_count}_timestep_{time_step_count}'] = (current_navigation, current_gesture, current_head_mov, current_volume)
                    save_dictionary[f'reward_episode_{episode_count}_timestep_{time_step_count}'] = reward
                    save_dictionary[f'goal_id_episode_{episode_count}_timestep_{time_step_count}'] = goal_id
                    save_dictionary[f'gaze_score_episode_{episode_count}_timestep_{time_step_count}'] = gaze_score
                    save_dictionary[f'next_gaze_score_episode_{episode_count}_timestep_{time_step_count}'] = next_gaze

                    # Update for next step
                    self.current_state = next_state
                    gaze_score = next_gaze
                    gaze_trajectory.append(gaze_score)
                    
                    # Check goal achievement
                    target_min, target_max = GAZE_RANGES[goal_id]
                    if (target_min <= next_gaze <= target_max) and step >= self.config['max_episode_steps']:
                        done = True
                    
                    step += 1
                    time_step_count += 1
                    
                    print(f"Next gaze score: {next_gaze:.2f} -- reward: {reward:.2f}")
                    
                    if done:
                        break

        # Save episode data
        save_trajectory_ep_to_yaml(episode_count, training_runname, save_dictionary)
        
        print('Training episode complete')
        controller.kill_attention_thread()
        
        del pepper
        return episode_reward, current_gaze, step, gaze_trajectory

    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'config': self.config
        }, filepath)

    # def load_model(self, filepath: str):
    #     """Load a trained model."""
    #     self.policy_net = torch.load(filepath)
    #     self.target_net = deepcopy(self.policy_net)
    #     self.epsilon = 0.0  # default to exploitation during testing
    #     print("Model loaded successfully.")


    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        self.config = checkpoint['config']
    
    # def load_model(self, filepath: str):
    #     """Load a trained model."""
    #     checkpoint = torch.load(filepath)
        
    #     self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    #     self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.epsilon = checkpoint['epsilon']
    #     self.step_count = checkpoint['step_count']
    #     self.episode_count = checkpoint['episode_count']
    #     self.config = checkpoint['config']
        
    #     print(f"Model loaded from {filepath}")

    def save_training_data(self, filepath: str):
        """Save training data to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.training_data, f, indent=2)

    def print_q_summary(self, goal_id: int, episode_num: int, save_dir: str = "q_summary_logs"):
        """Print and save a summary of Q-values for all possible gaze states and actions for the given goal."""

        # Ensure the output directory exists
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"q_summary_episode_{episode_num}.txt")

        with open(filename, 'w') as f:
            header = f"=== Q-Table Approximation Summary ===\nGoal ID: {goal_id} | Epsilon: {self.epsilon:.4f}\n\n"
            # print(header)
            f.write(header)

            # Only iterate through gaze states (0, 1, 2)
            for gaze in range(3):  # 0: Low, 1: Medium, 2: High
                state = {'gaze': gaze}
                
                input_tensor = self._create_network_input(state, goal_id)
                with torch.no_grad():
                    q_values = self.policy_net(input_tensor).squeeze().numpy()

                # Get top 5 actions for this state
                top_actions = np.argsort(q_values)[-5:][::-1]  # Top 5 in descending order
                
                gaze_names = ['Low', 'Medium', 'High']
                state_line = f"\n--- Gaze State: {gaze_names[gaze]} (value={gaze}) ---\n"
                # print(state_line)
                f.write(state_line)
                
                for rank, action_id in enumerate(top_actions, 1):
                    q_val = float(q_values[action_id])
                    delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[action_id]
                    
                    action_line = (f"  Rank {rank}: Action {action_id} | "
                                  f"Deltas(head={delta_head:+d}, nav={delta_nav:+d}, "
                                  f"gesture={delta_gesture:+d}, volume={delta_volume:+d}) | "
                                  f"Q-value: {q_val:.3f}\n")
                    
                    # print(action _line.strip())
                    f.write(action_line)

            footer = "\n=====================================\n"
            # print(footer)
            f.write(footer)

    def print_policy_summary(self, episode_num: int, save_dir: str = "policy_summary_logs"):
        """Print and save a concise summary of the best action for each gaze state across all goals."""
        
        # Ensure the output directory exists
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"policy_summary_episode_{episode_num}.txt")
        
        gaze_names = ['L', 'M', 'H']  # Low, Medium, High
        goal_names = ['Low', 'Medium', 'High']
        
        with open(filename, 'w') as f:
            header = f"=== Policy Summary - Episode {episode_num} ===\n"
            header += f"Epsilon: {self.epsilon:.4f}\n\n"
            # print(header.strip())
            f.write(header)
            
            # For each goal
            for goal_id in range(3):
                goal_line = f"Goal = {goal_names[goal_id]} ({goal_id})\n"
                # print(goal_line.strip())
                f.write(goal_line)
                
                # For each gaze state
                for gaze in range(3):
                    state = {'gaze': gaze}
                    
                    # Get Q-values for this state-goal combination
                    input_tensor = self._create_network_input(state, goal_id)
                    with torch.no_grad():
                        q_values = self.policy_net(input_tensor).squeeze().numpy()
                    
                    # Find best action
                    best_action_id = np.argmax(q_values)
                    delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[best_action_id]
                    
                    # Format the line: "L +1 +1 +1 +1"
                    action_line = f"{gaze_names[gaze]} {delta_head:+d} {delta_nav:+d} {delta_gesture:+d} {delta_volume:+d}\n"
                    # print(action_line.strip())
                    f.write(action_line)
                
                # Add blank line between goals
                # print()
                f.write("\n")
            
            footer = "=" * 40 + "\n"
            f.write(footer)

    def analyze_learned_policy(self, goal_id: int):
        """Analyze what the agent has learned for each gaze state."""
        # print(f"\n=== Learned Policy Analysis for Goal {goal_id} ===")
        
        gaze_names = ['Low', 'Medium', 'High']
        
        for gaze in range(3):
            state = {'gaze': gaze}
            
            input_tensor = self._create_network_input(state, goal_id)
            with torch.no_grad():
                q_values = self.policy_net(input_tensor).squeeze().numpy()
            
            best_action_id = np.argmax(q_values)
            best_q_value = q_values[best_action_id]
            delta_head, delta_nav, delta_gesture, delta_volume = self.delta_actions[best_action_id]
            
            # print(f"\n{gaze_names[gaze]} Gaze State:")
            # print(f"  Best Action: {best_action_id}")
            # print(f"  Delta Values: head={delta_head:+d}, nav={delta_nav:+d}, gesture={delta_gesture:+d}, volume={delta_volume:+d}")
            # print(f"  Q-value: {best_q_value:.3f}")
            
            # Show interpretation
            # actions_meaning = {
            #     'head': {-1: 'move left/down', 0: 'no change', 1: 'move right/up'},
            #     'nav': {-1: 'move back', 0: 'no change', 1: 'move closer'},
            #     'gesture': {-1: 'less gestures', 0: 'no change', 1: 'more gestures'},
            #     'volume': {-1: 'quieter', 0: 'no change', 1: 'louder'}
            # }
            
            # print(f"  Interpretation: {actions_meaning['head'][delta_head]}, {actions_meaning['nav'][delta_nav]}, {actions_meaning['gesture'][delta_gesture]}, {actions_meaning['volume'][delta_volume]}")

def save_training_state_after_episode(gcrl_model, episode_count: int, training_runname: str):
    """Save training state after episode (similar to Q-learning)."""
    if not os.path.exists(training_runname):
        os.makedirs(training_runname)
    
    # Save model
    model_path = os.path.join(training_runname, f"model_episode_{episode_count}.pt")
    gcrl_model.save_model(model_path)
    
    # Save training data
    data_path = os.path.join(training_runname, f"training_data_episode_{episode_count}.json")
    gcrl_model.save_training_data(data_path)
    
    print(f"Saved training state after episode {episode_count}")

if __name__ == "__main__":
    goal_id = 1
    training_runname = 'low_training_data'
    online_episode_duration = 5  # in minutes
    load_training_dir = True  # Set to True if you want to load existing training data

    # Initialize GCRL system
    gcrl_model = GazeGCRL()
    
    # Set training information
    episode_count = 0
    goal_id = goal_id
    online_episode_duration = online_episode_duration
    
    # Check if the training folder exists and handle loading
    if not os.path.exists(training_runname) and not load_training_dir:
        print('We are executing a new training session and do not need to load training data')
        os.makedirs(training_runname)
        print('Made directory:' + training_runname + ' for training data')
        print('New GCRL model initialized for training')
    elif load_training_dir and os.path.exists(training_runname):
        # Load the training data
        try:
            # Load the most recent model from the directory
            model_files = [f for f in os.listdir(training_runname) if f.startswith('model_episode_')]
            if model_files:
                # Get the latest episode number
                episode_numbers = [int(f.split('_')[-1].split('.')[0]) for f in model_files]
                latest_episode = max(episode_numbers)
                episode_count = latest_episode
                
                model_path = os.path.join(training_runname, f"model_episode_{latest_episode}.pt")
                gcrl_model.load_model(model_path)
                print(f'Loaded training data from {training_runname}, continuing from episode {episode_count}')
            else:
                print('No model files found in directory, starting new training')
        except Exception as e:
            print(f'Failed to load training data from {training_runname}: {e}')
            raise Exception(f'Failed to load training data from {training_runname}. Please check the file path and try again')
    else:
        raise Exception('An invalid arrangement of configurations and training data was provided. Please check the configurations/Arguments and try again')
    
    print('Starting training loop')
    
    # Training loop similar to Q-learning
    while True:
        # Increment the episode count
        episode_count += 1
        
        # Get the user input asking if they want to continue training
        user_input = input('Would you like to continue training for another episode? (Y/N): ')
        if user_input.lower() == 'y' or user_input.lower() == 'Y':
            # Run the next episode
            episode_reward, final_gaze, steps, gaze_trajectory = gcrl_model.run_training_episode(
                goal_id, episode_count, online_episode_duration, training_runname
            )
            
            # Print episode results
            print(f"Episode {episode_count} completed:")
            print(f"  Goal ID: {goal_id}")
            print(f"  Episode reward: {episode_reward:.2f}")
            print(f"  Final gaze: {final_gaze:.2f}")
            print(f"  Steps taken: {steps}")
            print(f"  Epsilon: {gcrl_model.epsilon:.4f}")
            
            # After episode, save the training state
            save_training_state_after_episode(gcrl_model, episode_count, training_runname)
            
            # Print Q-table summary for goal_id
            gcrl_model.print_q_summary(goal_id=goal_id, episode_num=episode_count)
            
            # Analyze learned policy
            gcrl_model.analyze_learned_policy(goal_id=goal_id)

            # Print concise policy summary
            gcrl_model.print_policy_summary(episode_num=episode_count)

        else:
            print('Your input was not Y/y. Exiting training')
            break
    
    print('Training completed successfully!')
