"""
Locust file for generating traffic to the load balancer.
"""
from locust import HttpUser, task, between
import yaml
from pathlib import Path


class LoadBalancerUser(HttpUser):
    """Locust user that sends requests to the load balancer."""
    
    wait_time = between(0.1, 0.5)  # Wait between requests
    
    def on_start(self):
        """Called when a user starts."""
        # Load config to get target URL
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.host = config.get('traffic', {}).get('target_url', 'http://localhost:8080')
    
    @task
    def get_root(self):
        """Send GET request to root path."""
        self.client.get("/", name="root")
    
    @task(3)
    def get_health(self):
        """Send GET request to health endpoint (less frequent)."""
        self.client.get("/health", name="health")
