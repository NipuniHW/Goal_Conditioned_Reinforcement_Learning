#!/usr/bin/python3.10
 
import random
import torch
import numpy as np
from collections import deque
import pandas as pd
import torch.nn as nn

def load_dataset_from_csv(filepath):
    df = pd.read_csv(filepath)
    dataset = []
    for _, row in df.iterrows():
        state = eval(row['state'])
        action = eval(row['action'])
        next_state = eval(row['next_state'])
        reward = float(row['reward'])
        done = bool(row['done'])
        goal = int(row['goal'])
        dataset.append((state, action, next_state, reward, done, goal))
    return dataset

class DatasetEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def reset(self):
        self.index = 0
        state, _, _, _, _, _ = self.dataset[self.index]
        return state

    def sample_goal(self):
        return random.choice([0, 1, 2])  # Low, Medium, High

    def step(self, action):
        if self.index >= len(self.dataset):
            return None, 0, True
        _, _, next_state, reward, done, _ = self.dataset[self.index]
        self.index += 1
        return next_state, reward, done

class GoalConditionedAgent:
    def __init__(self, config):
        self.config = config
        self.state_dim = 1
        self.goal_dim = 3
        self.action_dim = 81
        self.delta_actions = [
            (dh, dn, dg, dv)
            for dh in [-1, 0, 1]
            for dn in [-1, 0, 1]
            for dg in [-1, 0, 1]
            for dv in [-1, 0, 1]
        ]
        self._init_networks()

    def _init_networks(self):
        input_dim = self.state_dim + self.goal_dim
        hidden_sizes = self.config['network_hidden_sizes']
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
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])

    def _state_to_vector(self, state):
        return np.array([state['gaze']], dtype=np.float32)

    def _create_network_input(self, state, goal_id):
        state_vector = self._state_to_vector(state)
        goal_vector = np.zeros(self.goal_dim, dtype=np.float32)
        goal_vector[goal_id] = 1.0
        input_vector = np.concatenate([state_vector, goal_vector])
        return torch.FloatTensor(input_vector).unsqueeze(0)

    def save_model(self, filepath):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.config.get('epsilon', 1.0),  # Include epsilon
            'step_count': self.config.get('step_count', 0),
            'episode_count': self.config.get('episode_count', 0),
            'config': self.config
        }, filepath)


class GoalConditionedTrainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.memory = deque(maxlen=config['memory_size'])
        self.epsilon = config['epsilon']
        self.step_count = 0

    def select_action(self, state, goal_id):
        if random.random() < self.epsilon:
            return random.randint(0, self.agent.action_dim - 1)
        input_tensor = self.agent._create_network_input(state, goal_id)
        with torch.no_grad():
            q_values = self.agent.policy_net(input_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done, goal_id):
        self.memory.append((state, action, reward, next_state, done, goal_id))

    def sample_batch(self):
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        state_inputs = torch.cat([self.agent._create_network_input(s, g) for s, g in zip(states, goals)])
        next_inputs = torch.cat([self.agent._create_network_input(ns, g) for ns, g in zip(next_states, goals)])
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return state_inputs, actions, rewards, next_inputs, dones

    def train_step(self):
        if len(self.memory) < self.config['batch_size']:
            return
        state_inputs, actions, rewards, next_inputs, dones = self.sample_batch()
        q_values = self.agent.policy_net(state_inputs).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.agent.target_net(next_inputs).max(1)[0]
            target_q = rewards + self.config['discount_factor'] * next_q_values * (1 - dones)
        loss = torch.nn.functional.mse_loss(q_values, target_q)
        self.agent.optimizer.zero_grad()
        loss.backward()
        self.agent.optimizer.step()
        if self.step_count % self.config['target_update_freq'] == 0:
            self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            goal_id = self.env.sample_goal()
            done = False
            steps = 0
            while not done and steps < self.config['max_episode_steps']:
                action_id = self.select_action(state, goal_id)
                action = self.agent.delta_actions[action_id]
                next_state, reward, done = self.env.step(action)
                self.store_transition(state, action_id, reward, next_state, done, goal_id)
                self.train_step()
                state = next_state
                steps += 1
                self.step_count += 1
            self.epsilon = max(self.config['epsilon_min'], self.epsilon * self.config['epsilon_decay'])
            print(f"Episode {episode+1} complete. Epsilon: {self.epsilon:.3f}")
        self.agent.save_model("goal_conditioned_model.pth")
        print("Training complete and model saved.")

if __name__ == "__main__":
    config = {
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
    }

    dataset = load_dataset_from_csv("/home/vscode/gaze_ws/L2CS-Net/synthetic_gaze_dataset.csv")
    env = DatasetEnv(dataset)
    agent = GoalConditionedAgent(config)
    trainer = GoalConditionedTrainer(env, agent, config)
    trainer.train(num_episodes=100)
