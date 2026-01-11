"""
RL-based routing strategy for load balancing.
"""
from typing import List, Dict, Any, Optional
import asyncio

from rl.q_learning import QLearningAgent
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
        from pathlib import Path
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "rl_config.yaml"
        
        self.state_encoder = StateEncoder(config_path)
        
        # Initialize Q-Learning agent
        self.agent = QLearningAgent(
            num_actions=len(servers),
            config_path=config_path
        )
        
        self.lock = asyncio.Lock()
        
        # Track current state for updates
        self.last_state: Optional[tuple] = None
        self.last_action: Optional[int] = None
    
    async def select_server(self) -> Dict[str, Any]:
        """
        Select a server using the RL agent.
        
        Returns:
            Selected server dictionary
        """
        # Get current state from metrics (async)
        latest_metrics = await self.metrics_collector.get_latest_server_metrics_async()
        current_state = self.state_encoder.encode_state(latest_metrics, self.server_ids)

        # Select action (server)
        # If the agent's select_action remains sync, call it directly; it's lightweight.
        action = self.agent.select_action(current_state)
        selected_server = self.servers[action]
        
        # Store for reward update
        async with self.lock:
            self.last_state = current_state
            self.last_action = action
        
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
        
        # Calculate reward
        reward = self.agent.calculate_reward(response_time_ms, success)

        # Get next state (async)
        latest_metrics = await self.metrics_collector.get_latest_server_metrics_async()
        next_state = self.state_encoder.encode_state(latest_metrics, self.server_ids)

        # Update Q-table
        async with self.lock:
            self.agent.update(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state
            )
    
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
