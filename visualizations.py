#!/usr/bin/python3.10
# 
from copy import deepcopy
import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
import yaml
# Import your custom modules
from configurations import _load_config, _init_spaces, _init_networks, _init_training, _init_data_tracking, _normalize_gaze_to_category, _state_to_vector
from pepper import Pepper
from entropy_interfacing import GazeInterfaceController
from configurations import *
from train_saving_functions import save_trajectory_ep_to_yaml # type: ignore
from reward_function import calculate_reward 

class PepperGazeGCRL:
    """
    Goal-Conditioned Reinforcement Learning for Pepper Robot Gaze Control
    
    This system uses delta-based actions to control Pepper's behavior and 
    optimize human gaze levels for different target ranges:
    - Low gaze: 0-33 (goal_id=0)
    - Medium gaze: 34-66 (goal_id=1) 
    - High gaze: 67-100 (goal_id=2)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the GCRL system with configurable parameters."""

        # Load configuration
        self.config = _load_config(config)
        
        # Initialize state and action spaces, neural networks, training components, data tracking
        _init_spaces()
        _init_networks()
        _init_training()
        _init_data_tracking()
        
        print(f"Initialized PepperGazeGCRL with {_init_spaces.action_dim} delta actions")

    def next_state(self, action_id: int, current_gaze: float) -> Dict:
        """Apply delta action to current Pepper state."""
        delta_head, delta_nav, delta_gesture, delta_volume = _init_spaces.delta_actions[action_id]
        
        # Update gaze category based on current gaze value
        self.current_pepper_state['gaze'] = _normalize_gaze_to_category(current_gaze)
        
        # Apply deltas with bounds checking
        new_state = {
            'gaze': self.current_pepper_state['gaze'],
            'head_position': np.clip(self.current_pepper_state['head_position'] + delta_head, 0, 4),
            'navigation': np.clip(self.current_pepper_state['navigation'] + delta_nav, 0, 4),
            'gesture_level': np.clip(self.current_pepper_state['gesture_level'] + delta_gesture, 0, 10),
            'volume_level': np.clip(self.current_pepper_state['volume_level'] + delta_volume, 0, 10)
        }
        
        # Update current state
        self.current_pepper_state = new_state
        return new_state

    def select_action(self, state: Dict, goal_id: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            action_id = random.randint(0, _init_spaces.action_dim - 1)
        else:
            with torch.no_grad():
                input_tensor = _create_network_input(state, goal_id)
                q_values = self.policy_net(input_tensor)
                action_id = q_values.argmax().item()
        
        delta_head, delta_nav, delta_gesture, delta_volume = _init_spaces.delta_actions[action_id]

        # Track action distribution
        self.training_data['action_distributions'][action_id] += 1
        return action_id, delta_head, delta_nav, delta_gesture, delta_volume

    def store_experience(self, state: Dict, action: int, next_state: Dict, 
                        reward: float, done: bool, goal_id: int):
        """Store experience in replay buffer."""
        experience = (
            _state_to_vector(state),
            action,
            _state_to_vector(next_state),
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
        batch_goals = []
        
        for i in range(self.config['batch_size']):
            # Convert arrays back to state dicts for network input
            state_dict = {
                'gaze': int(states[i][0]),
                'head_position': int(states[i][1]),
                'navigation': int(states[i][2]),
                'gesture_level': int(states[i][3]),
                'volume_level': int(states[i][4])
            }
            
            next_state_dict = {
                'gaze': int(next_states[i][0]),
                'head_position': int(next_states[i][1]),
                'navigation': int(next_states[i][2]),
                'gesture_level': int(next_states[i][3]),
                'volume_level': int(next_states[i][4])
            }
            
            batch_states.append(_create_network_input(state_dict, goals[i]))
            batch_next_states.append(_create_network_input(next_state_dict, goals[i]))
        
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
            

    def train_episode(self, initial_gaze: float, goal_id: int, online_episode_duration, training_runname) -> Tuple[float, float, int, List[float]]:
        """Train one episode and return metrics."""
        print('connecting a session to pepper')
        
        pepper = Pepper()
        # pepper.connect("pepper.local", 9559)
        pepper.connect("localhost", 43197)

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

        # Initialize variables outside the loop - FIX: Initialize these variables at the start
        episode_reward = 0
        gaze_trajectory = []
        current_gaze = initial_gaze
        step = 0
        done = False
        
        # Initialize Pepper control variables at the start
        navigation = 2
        gesture = 5
        head_mov = 2
        volume = 5
        
        # Initialize episode data
        gaze_trajectory = [current_gaze]
        
        # Reset Pepper to neutral state
        self.current_pepper_state = {
            'gaze': _normalize_gaze_to_category(current_gaze),
            'head_position': 2,
            'navigation': 2,
            'gesture_level': 5,
            'volume_level': 5
        }
        
        start_time_inner_loop = time.time()

        while time.time() - current_time < online_episodes_duration_minutes:
            frame = controller.get_visualisation_frame()
            if frame is not None:
                f = deepcopy(frame)
                # print("the type of frame is ", type(f))
                cv2.imshow('Calibrated HRI Attention Detection', f)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            
            if time.time() - start_time_inner_loop >= 5:
                start_time_inner_loop = time.time()
                # Get the current gaze score
                gaze_score = controller.get_gaze_score()
                print(f"Current gaze score: {gaze_score:.2f}")
                current_gaze = gaze_score
                
                # Update gaze trajectory
                gaze_trajectory.append(current_gaze)
            
            state = self.current_pepper_state.copy()
            
            # Training step
            if step < self.config['max_episode_steps']:
                # Select and execute action
                action, delta_head, delta_nav, delta_gesture, delta_volume = self.select_action(state, goal_id)
                prev_gaze = current_gaze

                # Execute action on Pepper
                navigation, gesture, head_mov, volume = pepper.update_behavior(delta_head, delta_nav, delta_gesture, delta_volume, navigation, gesture, head_mov, volume)
                
                # Get the current gaze score after action
                next_gaze = controller.get_gaze_score()
                next_state = self.next_state(action, next_gaze)

                # Calculate reward
                reward = calculate_reward(next_gaze, goal_id, prev_gaze)
                episode_reward += reward
                
                # Store experience
                self.store_experience(state, action, next_state, reward, done, goal_id)
                
                # Train network
                loss = self.train_step()
                if loss is not None:
                    self.training_data['loss_history'].append(loss)
                
                # Update for next step
                state = next_state
                current_gaze = next_gaze
                gaze_trajectory.append(current_gaze)
                
                # Check goal achievement
                target_min, target_max = GAZE_RANGES[goal_id]
                if target_min <= next_gaze <= target_max:
                    done = True
                
                # Prepare data for saving
                step_data = {
                    'step': step,
                    'action': action,
                    'state': state,
                    'next_state': next_state,
                    'reward': reward,
                    'current_gaze': current_gaze,
                    'goal_id': goal_id,
                    'pepper_params': {
                        'navigation': navigation,
                        'gesture': gesture,
                        'head_mov': head_mov,
                        'volume': volume
                    },
                    'timestamp': time.time()
                }
                
                save_dictionary[f'step_{step}'] = step_data
                
                step += 1
                
                if done or step >= self.config['max_episode_steps']:
                    break

        # Save episode data
        episode_summary = {
            'episode_reward': episode_reward,
            'final_gaze': current_gaze,
            'initial_gaze': initial_gaze,
            'gaze_trajectory': gaze_trajectory,
            'steps_taken': step,
            'goal_id': goal_id,
            'success': done,
            'episode_duration': time.time() - current_time
        }
        
        save_dictionary['episode_summary'] = episode_summary
        
        episode_count += 1
        # Save the episode data
        save_trajectory_ep_to_yaml(
            episode_count=self.episode_count,
            training_runname=training_runname,
            training_dict=save_dictionary
        )
        
        # Clean up
        cv2.destroyAllWindows()
        controller.stop_detecting_attention()
    
        return episode_reward, current_gaze, step, gaze_trajectory

    def train(self, episodes: int = 1000, save_interval: int = 100, training_runname: str = None) -> Dict:
        """Train the model and save progress."""
        if training_runname is None:
            training_runname = f"pepper_gcrl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"Starting training for {episodes} episodes...")
        time_step_count = 0
        save_dictionary = {}

        for episode in range(episodes):
            goal_id = episode % 3
            initial_gaze = np.random.uniform(0, 100)
            
            # Pass training_runname to train_episode
            reward, final_gaze, length, trajectory = self.train_episode(
                initial_gaze, goal_id, online_episode_duration=10, training_runname=training_runname
            )
            
            # Record metrics
            self.training_data['episode_rewards'][goal_id].append(reward)
            self.training_data['episode_lengths'][goal_id].append(length)
            self.training_data['epsilon_history'].append(self.epsilon)
            self.training_data['gaze_trajectories'].append({
                'episode': episode,
                'goal_id': goal_id,
                'initial_gaze': initial_gaze,
                'final_gaze': final_gaze,
                'trajectory': trajectory
            })
            self.training_data['timestamps'].append(datetime.now().isoformat())

            # Calculate success rate
            target_min, target_max = GAZE_RANGES[goal_id]
            success = target_min <= final_gaze <= target_max
            
            # Update rolling success rate
            recent_rewards = self.training_data['episode_rewards'][goal_id][-100:]
            recent_successes = [r > 5 for r in recent_rewards]  # Threshold for success
            success_rate = sum(recent_successes) / len(recent_successes) if recent_successes else 0
            self.training_data['success_rates'][goal_id].append(success_rate)
            
            # Log progress
            if episode % 100 == 0:
                print(f"\nEpisode {episode}")
                for g in range(3):
                    if self.training_data['success_rates'][g]:
                        rate = self.training_data['success_rates'][g][-1]
                        avg_reward = np.mean(self.training_data['episode_rewards'][g][-10:]) if self.training_data['episode_rewards'][g] else 0
                        print(f"  Goal {g} ({self.GOAL_NAMES[g]}): Success rate: {rate:.2f}, Avg reward: {avg_reward:.2f}")
                print(f"  Epsilon: {self.epsilon:.3f}")
                
                # Save intermediate results
                if episode > 0 and episode % save_interval == 0:
                    self.save_training_data(f"training_checkpoint_{episode}.json")
            
            self.episode_count += 1

    
        print("\nTraining completed!")
        return self.training_data



    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress and metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot success rates
        axes[0, 0].set_title('Success Rates by Goal')
        for goal_id in range(3):
            if self.training_data['success_rates'][goal_id]:
                axes[0, 0].plot(self.training_data['success_rates'][goal_id], 
                              label=f'Goal {goal_id} ({self.GOAL_NAMES[goal_id]})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot episode rewards
        axes[0, 1].set_title('Episode Rewards')
        for goal_id in range(3):
            if self.training_data['episode_rewards'][goal_id]:
                # Plot moving average
                rewards = self.training_data['episode_rewards'][goal_id]
                window = min(50, len(rewards) // 10)
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(moving_avg, label=f'Goal {goal_id} ({self.GOAL_NAMES[goal_id]})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward (Moving Average)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot epsilon decay
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].plot(self.training_data['epsilon_history'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Plot loss history
        axes[1, 1].set_title('Training Loss')
        if self.training_data['loss_history']:
            # Plot moving average of loss
            losses = self.training_data['loss_history']
            window = min(100, len(losses) // 20)
            if window > 1:
                moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(moving_avg)
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss (Moving Average)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()


class TrainingDataAnalyzer:
    """Utility class for analyzing saved training data."""
    
    def __init__(self, data_path: str):
        """Load training data from file."""
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def print_summary(self):
        """Print a summary of the training data."""
        print("Training Data Summary:")
        print("="*50)
        
        # Basic info
        total_episodes = sum(len(rewards) for rewards in self.data['episode_rewards'].values())
        print(f"Total Episodes: {total_episodes}")
        
        if self.data['timestamps']:
            print(f"Training Duration: {self.data['timestamps'][0]} to {self.data['timestamps'][-1]}")
        
        # Success rates
        print("\nFinal Success Rates:")
        for goal_id in range(3):
            if self.data['success_rates'][str(goal_id)]:
                final_rate = self.data['success_rates'][str(goal_id)][-1]
                print(f"  Goal {goal_id} ({GOAL_NAMES[goal_id]}): {final_rate:.3f}")
        
        # Reward statistics
        print("\nReward Statistics:")
        for goal_id in range(3):
            rewards = self.data['episode_rewards'][str(goal_id)]
            if rewards:
                print(f"  Goal {goal_id}: Mean={np.mean(rewards):.2f}, "
                      f"Std={np.std(rewards):.2f}, Max={np.max(rewards):.2f}")
        
        # Action distribution
        if self.data['action_distributions']:
            total_actions = sum(self.data['action_distributions'].values())
            most_used = max(self.data['action_distributions'].items(), key=lambda x: x[1])
            print(f"\nMost used action: {most_used[0]} ({most_used[1]}/{total_actions} = "
                  f"{most_used[1]/total_actions:.3f})")
        
        # Training loss
        if self.data['loss_history']:
            losses = self.data['loss_history']
            print(f"\nTraining Loss: Mean={np.mean(losses):.4f}, "
                  f"Final={losses[-1]:.4f}, Min={np.min(losses):.4f}")
    
    def plot_detailed_analysis(self, save_path: Optional[str] = None):
        """Create detailed analysis plots."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        goal_names = ["Low", "Medium", "High"]
        
        # Success rates over time
        axes[0, 0].set_title('Success Rates Over Time')
        for goal_id in range(3):
            success_rates = self.data['success_rates'][str(goal_id)]
            if success_rates:
                axes[0, 0].plot(success_rates, label=f'Goal {goal_id} ({goal_names[goal_id]})')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].set_title('Episode Lengths')
        for goal_id in range(3):
            lengths = self.data['episode_lengths'][str(goal_id)]
            if lengths:
                axes[0, 1].plot(lengths, label=f'Goal {goal_id} ({goal_names[goal_id]})', alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Reward distributions
        axes[1, 0].set_title('Reward Distributions')
        for goal_id in range(3):
            rewards = self.data['episode_rewards'][str(goal_id)]
            if rewards:
                axes[1, 0].hist(rewards, alpha=0.7, bins=30, 
                               label=f'Goal {goal_id} ({goal_names[goal_id]})')
        axes[1, 0].set_xlabel('Episode Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Action distribution
        axes[1, 1].set_title('Action Usage Distribution')
        if self.data['action_distributions']:
            actions = list(self.data['action_distributions'].keys())
            counts = list(self.data['action_distributions'].values())
            # Show top 20 most used actions
            sorted_data = sorted(zip(actions, counts), key=lambda x: x[1], reverse=True)[:20]
            actions_top, counts_top = zip(*sorted_data)
            axes[1, 1].bar(range(len(actions_top)), counts_top)
            axes[1, 1].set_xlabel('Action ID (Top 20)')
            axes[1, 1].set_ylabel('Usage Count')
            axes[1, 1].set_xticks(range(len(actions_top)))
            axes[1, 1].set_xticklabels([str(a) for a in actions_top], rotation=45)
        
        # Sample gaze trajectories
        axes[2, 0].set_title('Sample Gaze Trajectories')
        if self.data['gaze_trajectories']:
            # Show a few sample trajectories for each goal
            for goal_id in range(3):
                goal_trajectories = [t for t in self.data['gaze_trajectories'] 
                                   if t['goal_id'] == goal_id]
                if goal_trajectories:
                    # Show up to 3 trajectories per goal
                    for i, traj in enumerate(goal_trajectories[-3:]):
                        axes[2, 0].plot(traj['trajectory'], 
                                       label=f'Goal {goal_id} ({goal_names[goal_id]}) #{i+1}',
                                       alpha=0.7)
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Gaze Level')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Learning curves (smoothed)
        axes[2, 1].set_title('Learning Curves (Smoothed)')
        for goal_id in range(3):
            rewards = self.data['episode_rewards'][str(goal_id)]
            if rewards and len(rewards) > 10:
                # Apply smoothing
                window = min(50, len(rewards) // 5)
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    axes[2, 1].plot(smoothed, label=f'Goal {goal_id} ({goal_names[goal_id]})')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('Smoothed Reward')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Detailed analysis plots saved to {save_path}")
        
        plt.show()


# Example usage functions
def load_and_analyze_training(data_path: str):
    """Load and analyze saved training data."""
    analyzer = TrainingDataAnalyzer(data_path)
    analyzer.print_summary()
    analyzer.plot_detailed_analysis()
    return analyzer

def run_evaluation_suite(model_path: str, num_tests_per_goal: int = 10):
    """Run comprehensive evaluation of a trained model."""
    # Load model
    pepper_gcrl = PepperGazeGCRL()
    pepper_gcrl.load_model(model_path)
    
    print(f"Running evaluation suite with {num_tests_per_goal} tests per goal...")
    
    all_results = {}
    
    for goal_id in range(3):
        print(f"\nEvaluating Goal {goal_id} ({pepper_gcrl.GOAL_NAMES[goal_id]})...")
        
        goal_results = []
        success_count = 0
        
        for test_num in range(num_tests_per_goal):
            # Test with different initial gaze levels
            if test_num < num_tests_per_goal // 3:
                initial_gaze = np.random.uniform(0, 33)    # Low start
            elif test_num < 2 * num_tests_per_goal // 3:
                initial_gaze = np.random.uniform(34, 66)   # Medium start
            else:
                initial_gaze = np.random.uniform(67, 100)  # High start
            
            result = pepper_gcrl.test_goal(goal_id, initial_gaze, verbose=False)
            goal_results.append(result)
            
            if result['success']:
                success_count += 1
        
        # Calculate statistics
        success_rate = success_count / num_tests_per_goal
        avg_steps = np.mean([r['steps_taken'] for r in goal_results])
        avg_gaze_change = np.mean([abs(r['final_gaze'] - r['initial_gaze']) 
                                  for r in goal_results])
        
        all_results[goal_id] = {
            'success_rate': success_rate,
            'average_steps': avg_steps,
            'average_gaze_change': avg_gaze_change,
            'individual_results': goal_results
        }
        
        print(f"Results for Goal {goal_id}:")
        print(f"  Success Rate: {success_rate:.3f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        print(f"  Average Gaze Change: {avg_gaze_change:.1f}")
    
    # Save evaluation results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = f"pepper_gcrl_evaluation_{timestamp}.json"
    
    # Convert for JSON serialization
    serializable_results = {}
    for goal_id, results in all_results.items():
        serializable_results[goal_id] = {
            'success_rate': float(results['success_rate']),
            'average_steps': float(results['average_steps']),
            'average_gaze_change': float(results['average_gaze_change']),
            'num_tests': num_tests_per_goal
        }
    
    with open(eval_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {eval_path}")
    return all_results
