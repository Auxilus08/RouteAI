"""
Q-Learning algorithm implementation for load balancing.
"""
import numpy as np
import pickle
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import yaml
from functools import lru_cache


class QLearningAgent:
    """Q-Learning agent for server selection."""
    
    def __init__(
        self,
        num_actions: int,
        config_path: Path = None,
        q_table: Optional[Dict[Tuple[int, ...], np.ndarray]] = None
    ):
        """
        Initialize Q-Learning agent.
        
        Args:
            num_actions: Number of possible actions (number of servers)
            config_path: Path to RL configuration file
            q_table: Pre-existing Q-table (for loading saved models)
        """
        self.num_actions = num_actions
        
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "rl_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        ql_config = config['q_learning']
        self.learning_rate = ql_config['learning_rate']  # Alpha
        self.discount_factor = ql_config['discount_factor']  # Gamma
        self.exploration_rate = ql_config['exploration_rate']  # Epsilon
        self.min_exploration_rate = ql_config['min_exploration_rate']
        self.exploration_decay = ql_config['exploration_decay']
        
        self.reward_config = config['reward']
        
        # Initialize Q-table
        if q_table is not None:
            self.q_table = q_table
        else:
            self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        
        # Statistics
        self.episode_count = 0
        self.total_updates = 0
    
    def get_q_value(self, state: Tuple[int, ...], action: int) -> float:
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: State tuple
            action: Action (server index)
        
        Returns:
            Q-value (0.0 if not initialized)
        """
        self.ensure_state_initialized(state)
        return self.q_table[state][action]

    def ensure_state_initialized(self, state: Tuple[int, ...]):
        """
        Ensure a state's Q-entry exists. Uses setdefault for concise lazy init.
        """
        self.q_table.setdefault(state, np.zeros(self.num_actions))
    
    def select_action(self, state: Tuple[int, ...], available_actions: List[int] = None) -> int:
        """
        Select an action using ε-greedy policy.
        
        Args:
            state: Current state tuple
            available_actions: List of available actions (if None, all actions are available)
        
        Returns:
            Selected action (server index)
        """
        if available_actions is None:
            available_actions = list(range(self.num_actions))

        # Ensure Q-values exist for state
        self.ensure_state_initialized(state)

        # Exploration: random action among available
        if np.random.random() < self.exploration_rate:
            return int(np.random.choice(available_actions))

        # Exploitation: choose best action among available actions using masked Q-values
        q_values = self.q_table[state]
        available_mask = np.zeros(self.num_actions, dtype=bool)
        available_mask[available_actions] = True
        q_values_masked = q_values.copy()
        q_values_masked[~available_mask] = -np.inf
        best_action = int(np.argmax(q_values_masked))
        return best_action
    
    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...]
    ):
        """
        Update Q-value using Bellman equation.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Ensure Q-values exist
        self.ensure_state_initialized(state)
        self.ensure_state_initialized(next_state)

        # Current Q-value
        current_q = self.q_table[state][action]

        # Maximum Q-value for next state (cached wrapper)
        max_next_q = self._cached_max_q(next_state)
        
        # Bellman equation: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
        self.total_updates += 1
        # Invalidate cached max-Qs for affected states to keep cache consistent.
        # We clear the whole cache; acceptable tradeoff for moderate cache sizes.
        try:
            self._cached_max_q.cache_clear()
        except Exception:
            pass

    @lru_cache(maxsize=4096)
    def _cached_max_q(self, state: Tuple[int, ...]) -> float:
        """
        Cached wrapper around max Q computation for a state. Cache is cleared
        when updates occur to keep values consistent.
        """
        return float(np.max(self.q_table[state]))
    
    def calculate_reward(self, response_time_ms: float, success: bool) -> float:
        """
        Calculate reward based on response time and success.
        
        Args:
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
        
        Returns:
            Reward value (negative for latency minimization)
        """
        if not success:
            return self.reward_config['failure_penalty']
        # Normalize reward using configured latency bounds to keep rewards
        # comparable across experiments.
        min_lat = self.reward_config.get('min_latency_ms', 0.0)
        max_lat = self.reward_config.get('max_latency_ms', 1000.0)
        scale = self.reward_config.get('scale', 100.0)
        denom = max(1e-6, (max_lat - min_lat))
        normalized = (response_time_ms - min_lat) / denom
        reward = -normalized * scale
        return float(reward)
    
    def decay_exploration(self):
        """Decay exploration rate after an episode."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.exploration_decay
        )
        self.episode_count += 1
    
    def save_q_table(self, filepath: Path):
        """Save Q-table to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'num_actions': self.num_actions,
                'episode_count': self.episode_count,
                'total_updates': self.total_updates,
                'exploration_rate': self.exploration_rate
            }, f)
    
    @classmethod
    def load_q_table(cls, filepath: Path, config_path: Path = None) -> 'QLearningAgent':
        """Load Q-table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        agent = cls(
            num_actions=data['num_actions'],
            config_path=config_path,
            q_table=data['q_table']
        )
        agent.episode_count = data.get('episode_count', 0)
        agent.total_updates = data.get('total_updates', 0)
        agent.exploration_rate = data.get('exploration_rate', agent.exploration_rate)
        return agent
    
    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics."""
        return {
            'num_states': len(self.q_table),
            'episode_count': self.episode_count,
            'total_updates': self.total_updates,
            'exploration_rate': self.exploration_rate
        }
