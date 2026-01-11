"""
Evaluation runner for comparing Round Robin vs RL agent.
"""
import asyncio
import time
import subprocess
import signal
import sys
from pathlib import Path
from typing import Dict, Any
import yaml

import httpx

from evaluation.metrics_analyzer import MetricsAnalyzer
from evaluation.export import ResultsExporter


class EvaluationRunner:
    """Runs evaluation experiments comparing routing strategies."""
    
    def __init__(self, config_path: Path = None):
        """
        Initialize evaluation runner.
        
        Args:
            config_path: Path to configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.servers = self.config['servers']
        self.lb_config = self.config['load_balancer']
        self.traffic_config = self.config['traffic']
        
        # Process tracking
        self.server_processes = []
        self.lb_process = None
        
        # Results storage
        self.results = {}
    
    async def start_servers(self):
        """Start backend servers."""
        print("Starting backend servers...")
        for server in self.servers:
            cmd = [
                sys.executable,
                str(Path(__file__).parent.parent / "backend" / "server.py"),
                server['id'],
                str(server['port']),
                str(server['base_delay_ms'])
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.server_processes.append(process)
            await asyncio.sleep(0.5)  # Stagger starts
        
        # Wait for servers to be ready
        print("Waiting for servers to be ready...")
        await asyncio.sleep(2)
        
        # Verify servers are up
        for server in self.servers:
            url = f"http://{server['host']}:{server['port']}/health"
            for _ in range(10):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, timeout=2.0)
                        if response.status_code == 200:
                            print(f"Server {server['id']} is ready")
                            break
                except:
                    await asyncio.sleep(0.5)
    
    async def start_load_balancer(self, strategy: str = "round_robin"):
        """Start load balancer with specified strategy."""
        print(f"Starting load balancer with {strategy} strategy...")
        
        # Import here to avoid circular imports
        from load_balancer.main import app, initialize_load_balancer
        import uvicorn
        
        # Initialize load balancer
        initialize_load_balancer(self.servers, strategy)
        
        # Run in background
        config = uvicorn.Config(
            app,
            host=self.lb_config['host'],
            port=self.lb_config['port'],
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        # Start server in background task
        self.lb_task = asyncio.create_task(server.serve())
        
        # Wait for LB to be ready
        await asyncio.sleep(2)
        lb_url = f"http://{self.lb_config['host']}:{self.lb_config['port']}/health"
        for _ in range(10):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(lb_url, timeout=2.0)
                    if response.status_code == 200:
                        print(f"Load balancer is ready")
                        break
            except:
                await asyncio.sleep(0.5)
    
    async def generate_traffic(self, duration_seconds: float, request_rate: float):
        """Generate traffic to the load balancer."""
        print(f"Generating traffic: {request_rate} req/s for {duration_seconds}s")
        
        from traffic.generator import generate_traffic
        
        lb_url = self.traffic_config['target_url']
        await generate_traffic(
            target_url=lb_url,
            request_rate=request_rate,
            duration_seconds=duration_seconds,
            max_concurrent=100
        )
    
    async def run_experiment(self, strategy: str, num_requests: int = 1000, request_rate: float = 10.0):
        """
        Run a single experiment with a routing strategy.
        
        Args:
            strategy: Routing strategy ("round_robin" or "rl_agent")
            num_requests: Number of requests to send
            request_rate: Requests per second
        """
        print(f"\n{'='*60}")
        print(f"Running experiment: {strategy}")
        print(f"{'='*60}\n")
        
        # Start load balancer
        await self.start_load_balancer(strategy)
        
        # Calculate duration
        duration = num_requests / request_rate
        
        # Generate traffic
        await self.generate_traffic(duration, request_rate)
        
        # Stop load balancer
        if hasattr(self, 'lb_task'):
            self.lb_task.cancel()
            try:
                await self.lb_task
            except asyncio.CancelledError:
                pass
        
        await asyncio.sleep(1)
        print(f"Experiment {strategy} complete\n")
    
    async def run_comparison(
        self,
        num_requests_per_strategy: int = 1000,
        request_rate: float = 10.0
    ) -> Dict[str, Any]:
        """
        Run comparison between Round Robin and RL agent.
        
        Args:
            num_requests_per_strategy: Number of requests per strategy
            request_rate: Requests per second
        
        Returns:
            Comparison results dictionary
        """
        try:
            # Start servers
            await self.start_servers()
            
            # Run Round Robin experiment
            await self.run_experiment("round_robin", num_requests_per_strategy, request_rate)
            
            # Wait a bit between experiments
            await asyncio.sleep(2)
            
            # Run RL agent experiment
            await self.run_experiment("rl_agent", num_requests_per_strategy, request_rate)
            
            # Get metrics from load balancer (this would need to be implemented)
            # For now, we'll need to collect metrics differently
            
            print("Comparison complete!")
            
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Stop all processes."""
        print("Cleaning up...")
        
        # Stop load balancer
        if hasattr(self, 'lb_task'):
            self.lb_task.cancel()
        
        # Stop servers
        for process in self.server_processes:
            process.terminate()
            process.wait()
        
        self.server_processes.clear()
        print("Cleanup complete")


async def main():
    """Main entry point for evaluation runner."""
    runner = EvaluationRunner()
    
    # Run comparison
    await runner.run_comparison(
        num_requests_per_strategy=500,  # Smaller for testing
        request_rate=10.0
    )


if __name__ == "__main__":
    asyncio.run(main())
