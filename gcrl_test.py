#!/usr/bin/python3.10

import sys
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from pepper import Pepper
from entropy_interfacing import *
from gcrl_train import *
from copy import deepcopy
from time import sleep, time

class GCRL(nn.Module):
    """Deep Q-Network for the gaze-goal environment."""
    
    def __init__(self, state_size: int = 6, action_size: int = 4, hidden_size: int = 128):
        super(GCRL, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class GCRLAgent:
    """DQN Agent for testing."""
    
    def __init__(self, model_path: str):
        self.action_size = 4  # Head movement, navigation, gesture, volume
        self.policy_net = GCRL()
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load the trained model."""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.policy_net.eval()
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded. Epsilon: {self.epsilon:.4f}")
    
    def get_state_encoding(self, gaze_state: int, goal: int) -> torch.Tensor:
        """Convert gaze state and goal to one-hot encoded tensor."""
        state = np.zeros(6)
        state[gaze_state] = 1.0      # Gaze: positions 0, 1, 2
        state[3 + goal] = 1.0        # Goal: positions 3, 4, 5
        return torch.FloatTensor(state).unsqueeze(0)
    
    # def choose_action(self, gaze_state: int, goal: int) -> int:
    #     """Choose action based on Q-values (greedy policy for testing)."""
    #     state = self.get_state_encoding(gaze_state, goal)
    #     with torch.no_grad():
    #         q_values = self.policy_net(state)
    #         action = q_values.argmax().item()
    #     return action
    
    def _state_to_vector(self, state: Dict) -> np.ndarray:
        """Convert state dictionary to vector."""
        return np.array([state['gaze']], dtype=np.float32)

    def _create_network_input(self, state: Dict, goal_id: int) -> torch.Tensor:
        """Create neural network input from state and goal."""
        # Convert state to vector
        state_vector = self._state_to_vector(state)
        self.goal_dim = 3 

        # Create one-hot goal vector
        goal_vector = np.zeros(self.goal_dim, dtype=np.float32)
        goal_vector[goal_id] = 1.0
        
        # Concatenate and convert to tensor
        input_vector = np.concatenate([state_vector, goal_vector])
        return torch.FloatTensor(input_vector).unsqueeze(0)
    
    def choose_action(self, state: Dict, goal_id: int, training: bool = False) -> Tuple[int, int, int, int, int]:
        """Select action using epsilon-greedy policy."""

        with torch.no_grad():
            input_tensor = self._create_network_input(state, goal_id)
            q_values = self.policy_net(input_tensor)
            action_id = q_values.argmax().item()
        
        delta_head, delta_nav, delta_gesture, delta_volume = GazeGCRL.delta_actions[action_id]

        return action_id, delta_head, delta_nav, delta_gesture, delta_volume
    
    def get_q_values(self, gaze_state: int, goal: int) -> np.ndarray:
        """Get Q-values for debugging."""
        state = self.get_state_encoding(gaze_state, goal)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze().numpy()
        return q_values

def test_dqn_model(model_path, duration_minutes, initial_goal):
    """Test the DQN model with Pepper robot."""
    
    # Initialize Pepper
    pepper = Pepper()
    # pepper.connect("localhost", 33577)
    
    # try:
    #     if not pepper.is_connected:
    #         sys.exit(1)
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt detected. Cleaning up...")
    #     del pepper   
    #     cv2.destroyAllWindows()
    #     sys.exit(0)
    
    # Load DQN agent
    agent = GCRLAgent(model_path)
    
    # Initialize gaze controller
    controller = GazeInterfaceController(camera_id=0)
    sleep(1)
    
    # Calibration
    print('Press Enter to start the calibration')
    input()
    controller.calibration_exe()
    controller.start_detecting_attention()
    
    # Start testing
    sleep(1)
    print('Press Enter to start the testing')
    input()
    
    # Testing parameters
    current_time = time()
    current_goal = initial_goal  # Goal: 0=Low, 1=Medium, 2=High
    online_episodes_duration_minutes = duration_minutes * 60
    start_time_inner_loop = time()
    
    # Default behavior values
    head_movement, navigation, gesture, volume = 0, 0, 5, 5  # Added head_movement
    
    action_names = ['Head Movement', 'Navigation', 'Gesture', 'Volume']
    goal_names = ['Low', 'Medium', 'High']
    
    print(f"Starting test with goal: {goal_names[current_goal]} ({current_goal})")
    print("Actions: 0=Head Movement, 0=Navigation, 5=Gesture, 5=Volume\n")
    
    while time() - current_time < online_episodes_duration_minutes:
        # Display camera frame
        frame = controller.get_visualisation_frame()
        if frame is not None:
            f = deepcopy(frame)
            cv2.imshow('DQN Goal-Conditioned HRI Attention Detection', f)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        
        # Action selection every 5 seconds
        if time() - start_time_inner_loop >= 5:
            start_time_inner_loop = time()
            
            # Get gaze state
            gaze_score = controller.get_gaze_score()
            gaze_state = int(round(gaze_score / 20))  # Convert to 0, 1, 2
            gaze_state = max(0, min(2, gaze_state))   # Clamp to valid range
            
            print(f"Gaze score: {gaze_score} -> Gaze state: {gaze_state}")
            print(f"Current goal: {goal_names[current_goal]} ({current_goal})")
            
            # Get Q-values for debugging
            q_values = agent.get_q_values(gaze_state, current_goal)
            print(f"Q-values: {q_values}")
            
            # Choose action
            action = agent.choose_action(gaze_state, current_goal)
            print(f"Chosen action: {action} ({action_names[action]})")
            
            # Apply the behavior updates to Pepper
            head_movement, navigation, gesture, volume = pepper.update_behavior(head_movement, navigation, gesture, volume)
            print("Updated robot behavior\n")
            
            # Optional: Change goal dynamically (uncomment if needed)
            # elapsed_minutes = (time() - current_time) / 60
            # if elapsed_minutes > duration_minutes / 3 and current_goal == initial_goal:
            #     current_goal = (current_goal + 1) % 3
            #     print(f"*** Goal changed to: {goal_names[current_goal]} ({current_goal}) ***\n")
        
        # Loop delay
        sleep(0.18)
    
    del pepper
    cv2.destroyAllWindows()
    print("DQN test completed")

def interactive_goal_test():
    """Interactive testing with goal selection."""
    model_path = "/home/vscode/gaze_ws/L2CS-Net/trained_dqn_episode_44.pth"
    
    print("=== DQN Goal-Conditioned RL Testing ===")
    print("Goals: 0=Low, 1=Medium, 2=High")
    
    try:
        goal = int(input("Enter initial goal (0, 1, or 2): "))
        if goal not in [0, 1, 2]:
            print("Invalid goal. Using default goal=0")
            goal = 0
            
        duration = int(input("Enter test duration in minutes: "))
        if duration <= 0:
            print("Invalid duration. Using default 5 minutes")
            duration = 5
            
    except ValueError:
        print("Invalid input. Using defaults: goal=0, duration=5 minutes")
        goal = 0
        duration = 5
    
    test_dqn_model(model_path, duration, goal)

if __name__ == "__main__":
    interactive_goal_test()