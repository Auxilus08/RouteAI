"""
Round Robin routing strategy for load balancing.
"""
from typing import List, Dict, Any
import threading


class RoundRobinRouter:
    """Round Robin routing algorithm."""
    
    def __init__(self, servers: List[Dict[str, Any]]):
        """
        Initialize Round Robin router.
        
        Args:
            servers: List of server dictionaries with 'id', 'host', 'port' keys
        """
        self.servers = servers
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select_server(self) -> Dict[str, Any]:
        """
        Select the next server using Round Robin.
        
        Returns:
            Selected server dictionary
        """
        with self.lock:
            server = self.servers[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.servers)
            return server
    
    def get_server_url(self, server: Dict[str, Any]) -> str:
        """
        Construct server URL from server dictionary.
        
        Args:
            server: Server dictionary
        
        Returns:
            Server URL string
        """
        return f"http://{server['host']}:{server['port']}"
    
    def get_all_servers(self) -> List[Dict[str, Any]]:
        """Get all available servers."""
        return self.servers
