"""
RL-based routing strategy for load balancing.
"""
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np
import yaml
from pathlib import Path

from rl.dqn import DQNAgent
from rl.state_encoder import StateEncoder
from metrics.collector import MetricsCollector


class RLRouter:
    """RL-based routing using Q-Learning."""
    
    def __init__(
        self,
        servers: List[Dict[str, Any]],
        metrics_collector: MetricsCollector,
        config_path: Optional[str] = None
    ):
        """
        Initialize RL router.
        
        Args:
            servers: List of server dictionaries
            metrics_collector: Metrics collector instance
            config_path: Path to RL configuration file
        """
        self.servers = servers
        self.server_ids = [s['id'] for s in servers]
        self.metrics_collector = metrics_collector
        
        # Initialize state encoder
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "rl_config.yaml"
        self.config_path = Path(config_path)

        self.state_encoder = StateEncoder(self.config_path)

        with open(self.config_path, 'r') as f:
            self.rl_config = yaml.safe_load(f)

        dqn_cfg = self.rl_config.get('dqn', {})
        state_dim = len(self.server_ids) * 4  # [cpu, active, rt, reachable] per server
        self.agent = DQNAgent(
            state_dim=state_dim,
            num_actions=len(servers),
            hidden_sizes=dqn_cfg.get('hidden_sizes', [128, 128]),
            learning_rate=dqn_cfg.get('learning_rate', 1e-3),
            gamma=dqn_cfg.get('discount_factor', 0.99),
            epsilon_start=dqn_cfg.get('epsilon_start', 1.0),
            epsilon_min=dqn_cfg.get('epsilon_min', 0.05),
            epsilon_decay=dqn_cfg.get('epsilon_decay', 0.995),
            buffer_size=dqn_cfg.get('buffer_size', 50000),
            batch_size=dqn_cfg.get('batch_size', 64),
            target_update_freq=dqn_cfg.get('target_update_freq', 200),
            train_start=dqn_cfg.get('train_start', 500),
            gradient_clip=dqn_cfg.get('gradient_clip', 1.0),
        )

        self.reward_config = self.rl_config.get('reward', {})
        
        self.lock = asyncio.Lock()
        
        # Track current state for updates
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
        self.last_available_actions: Optional[List[int]] = None
    
    async def select_server(self) -> Dict[str, Any]:
        """
        Select a server using the RL agent.
        
        Returns:
            Selected server dictionary
        """
        # Get current state from metrics (async)
        latest_metrics = await self.metrics_collector.get_latest_server_metrics_async()
        current_state = np.array(
            self.state_encoder.encode_state_continuous(latest_metrics, self.server_ids),
            dtype=np.float32,
        )
        available_actions = self._get_available_actions(latest_metrics)

        action = self.agent.select_action(current_state, available_actions)
        selected_server = self.servers[action]
        
        # Store for reward update
        async with self.lock:
            self.last_state = current_state
            self.last_action = action
            self.last_available_actions = available_actions
        
        return selected_server
    
    async def update_reward(self, response_time_ms: float, success: bool):
        """
        Update the RL agent with reward feedback.
        
        Args:
            response_time_ms: Response time in milliseconds
            success: Whether the request was successful
        """
        if self.last_state is None or self.last_action is None:
            return
        
        reward = self._calculate_reward(response_time_ms, success)

        latest_metrics = await self.metrics_collector.get_latest_server_metrics_async()
        next_state = np.array(
            self.state_encoder.encode_state_continuous(latest_metrics, self.server_ids),
            dtype=np.float32,
        )
        next_available = self._get_available_actions(latest_metrics)

        async with self.lock:
            self.agent.remember(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state,
                done=not success,
                next_available_actions=next_available,
            )
            self.agent.train_step()

            # Roll over state/action
            self.last_state = next_state
            self.last_available_actions = next_available
    
    def get_server_url(self, server: Dict[str, Any]) -> str:
        """Construct server URL from server dictionary."""
        return f"http://{server['host']}:{server['port']}"
    
    def get_all_servers(self) -> List[Dict[str, Any]]:
        """Get all available servers."""
        return self.servers
    
    def save_model(self, filepath: str):
        """Save the Q-table to a file."""
        from pathlib import Path
        self.agent.save_q_table(Path(filepath))
    
    def load_model(self, filepath: str):
        """Load a Q-table from a file."""
        from pathlib import Path
        self.agent = QLearningAgent.load_q_table(Path(filepath))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        return self.agent.get_statistics()

    def _calculate_reward(self, response_time_ms: float, success: bool) -> float:
        if not success:
            return float(self.reward_config.get('failure_penalty', -1000.0))
        min_lat = self.reward_config.get('min_latency_ms', 0.0)
        max_lat = self.reward_config.get('max_latency_ms', 1000.0)
        scale = self.reward_config.get('scale', 100.0)
        denom = max(1e-6, (max_lat - min_lat))
        normalized = (response_time_ms - min_lat) / denom
        reward = -normalized * scale
        return float(reward)

    def _get_available_actions(self, latest_metrics: Dict[str, Dict[str, Any]]) -> List[int]:
        available: List[int] = []
        for idx, server in enumerate(self.servers):
            metrics = latest_metrics.get(server['id']) if latest_metrics else None
            if metrics and metrics.get('health_status') != 'unreachable':
                available.append(idx)
        if not available:
            return list(range(len(self.servers)))
        return available
