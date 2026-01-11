"""Deep Q-Network agent for load balancing."""
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _to_tensor(array_like, device):
    return torch.as_tensor(array_like, dtype=torch.float32, device=device)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with target network and replay buffer."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        train_start: int = 500,
        gradient_clip: float = 1.0,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_sizes = hidden_sizes or [128, 128]
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_start = train_start
        self.gradient_clip = gradient_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(state_dim, num_actions, self.hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, num_actions, self.hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.update_steps = 0

    def select_action(self, state: np.ndarray, available_actions: Optional[List[int]] = None) -> int:
        if available_actions is None:
            available_actions = list(range(self.num_actions))

        if random.random() < self.epsilon:
            return random.choice(available_actions)

        state_t = _to_tensor(state, self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)
        mask = torch.full((self.num_actions,), float('-inf'), device=self.device)
        mask[available_actions] = 0.0
        masked_q = q_values + mask
        action = int(torch.argmax(masked_q).item())
        return action

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_available_actions: Optional[List[int]] = None,
    ):
        if next_available_actions is None:
            next_available_actions = list(range(self.num_actions))
        mask = np.zeros(self.num_actions, dtype=np.float32)
        mask[next_available_actions] = 1.0
        transition = Transition(state, action, reward, next_state, done, mask)
        self.replay_buffer.push(transition)

    def train_step(self):
        if len(self.replay_buffer) < max(self.train_start, self.batch_size):
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = _to_tensor(np.stack([t.state for t in batch], axis=0), self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = _to_tensor([t.reward for t in batch], self.device)
        next_states = _to_tensor(np.stack([t.next_state for t in batch], axis=0), self.device)
        dones = _to_tensor([float(t.done) for t in batch], self.device)
        next_masks = _to_tensor(np.stack([t.next_action_mask for t in batch], axis=0), self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values[next_masks < 0.5] = float('-inf')
            max_next_q = torch.max(next_q_values, dim=1).values
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clip:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self._decay_epsilon()
        return loss.item()

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_statistics(self):
        return {
            "buffer_size": len(self.replay_buffer),
            "epsilon": self.epsilon,
            "updates": self.update_steps,
            "device": str(self.device),
        }
