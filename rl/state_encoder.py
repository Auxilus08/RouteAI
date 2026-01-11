"""
State encoder for converting server metrics to discrete state representation.
"""
from typing import Dict, Any, Tuple
import yaml
from pathlib import Path


class StateEncoder:
    """Encodes server metrics into discrete state space for Q-Learning."""
    
    def __init__(self, config_path: Path = None):
        """
        Initialize state encoder with threshold configuration.
        
        Args:
            config_path: Path to RL configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "rl_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        thresholds = config['state_encoding']
        
        self.cpu_thresholds = thresholds['cpu_thresholds']
        self.active_requests_thresholds = thresholds['active_requests_thresholds']
        self.response_time_thresholds = thresholds['response_time_thresholds_ms']
        
        # Load levels: 0 = low, 1 = medium, 2 = high
        self.load_levels = ["low", "medium", "high"]
    
    def encode_load_level(self, cpu: float, active_requests: int, response_time: float) -> int:
        """
        Encode a single server's metrics into a load level.
        
        Args:
            cpu: CPU utilization percentage
            active_requests: Number of active requests
            response_time: Average response time in milliseconds
        
        Returns:
            Load level: 0 (low), 1 (medium), or 2 (high)
        """
        # Use the worst metric to determine load level
        cpu_level = 0
        if cpu >= self.cpu_thresholds['medium']:
            cpu_level = 2
        elif cpu >= self.cpu_thresholds['low']:
            cpu_level = 1
        
        req_level = 0
        if active_requests >= self.active_requests_thresholds['medium']:
            req_level = 2
        elif active_requests >= self.active_requests_thresholds['low']:
            req_level = 1
        
        rt_level = 0
        if response_time >= self.response_time_thresholds['medium']:
            rt_level = 2
        elif response_time >= self.response_time_thresholds['low']:
            rt_level = 1
        
        # Take maximum (worst metric determines load level)
        load_level = max(cpu_level, req_level, rt_level)
        return load_level
    
    def encode_state(self, server_metrics: Dict[str, Dict[str, Any]], server_ids: list) -> Tuple[int, ...]:
        """
        Encode all server metrics into a state tuple.
        
        Args:
            server_metrics: Dictionary mapping server_id to metrics
            server_ids: List of server IDs in order
        
        Returns:
            State tuple: (server1_load, server2_load, server3_load, ...)
        """
        state = []
        for server_id in server_ids:
            if server_id in server_metrics:
                metrics = server_metrics[server_id]
                load_level = self.encode_load_level(
                    cpu=metrics.get('cpu_utilization', 0.0),
                    active_requests=metrics.get('active_requests', 0),
                    response_time=metrics.get('avg_response_time', 0.0)
                )
            else:
                # Server unreachable = high load
                load_level = 2
            state.append(load_level)
        
        return tuple(state)
    
    def get_state_space_size(self, num_servers: int) -> int:
        """
        Calculate the size of the state space.
        
        Args:
            num_servers: Number of servers
        
        Returns:
            Total number of possible states (3^num_servers)
        """
        return 3 ** num_servers
