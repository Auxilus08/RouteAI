"""
Main script to run a complete evaluation experiment.
"""
import asyncio
import time
import subprocess
import sys
import signal
import socket
from pathlib import Path
from typing import Optional
import yaml
import pandas as pd

from load_balancer.main import initialize_load_balancer, app
from metrics.collector import MetricsCollector
from evaluation.metrics_analyzer import MetricsAnalyzer
from evaluation.export import ResultsExporter
from visualization.plots import create_all_visualizations
import uvicorn


def is_port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
        except OSError:
            return False


async def wait_for_port_release(host: str, port: int, timeout: float = 10.0) -> bool:
    """Wait for a port to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if is_port_available(host, port):
            return True
        await asyncio.sleep(0.5)
    return False


class ExperimentRunner:
    """Runs a complete evaluation experiment."""
    
    def __init__(self):
        """Initialize experiment runner."""
        config_path = Path(__file__).parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.servers = self.config['servers']
        self.lb_config = self.config['load_balancer']
        self.server_processes = []
        self.server_task = None
    
    async def start_servers(self):
        """Start backend servers."""
        print("Starting backend servers...")
        
        for server in self.servers:
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "backend" / "server.py"),
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
            await asyncio.sleep(0.5)
        
        # Wait for servers
        await asyncio.sleep(2)
        print("Backend servers started")
    
    async def run_strategy_experiment(
        self,
        strategy: str,
        metrics_collector: MetricsCollector,
        num_requests: int = 1000,
        request_rate: float = 10.0
    ):
        """Run experiment with a specific routing strategy."""
        print(f"\n{'='*70}")
        print(f"Running experiment: {strategy.upper()}")
        print(f"{'='*70}\n")
        
        # Clear metrics before experiment
        metrics_collector.clear_metrics()
        
        # Initialize load balancer with strategy and existing metrics collector
        initialize_load_balancer(self.servers, strategy, metrics_collector_instance=metrics_collector)
        
        # Ensure port is available before starting
        await wait_for_port_release(self.lb_config['host'], self.lb_config['port'], timeout=5.0)
        
        # Start load balancer
        config = uvicorn.Config(
            app,
            host=self.lb_config['host'],
            port=self.lb_config['port'],
            log_level="warning"  # Reduce logging
        )
        server = uvicorn.Server(config)
        lb_task = asyncio.create_task(server.serve())
        
        # Wait for LB to start
        await asyncio.sleep(2)
        
        try:
            # Generate traffic
            from traffic.generator import generate_traffic
            lb_url = f"http://{self.lb_config['host']}:{self.lb_config['port']}"
            duration = num_requests / request_rate
            
            print(f"Generating {num_requests} requests at {request_rate} req/s...")
            await generate_traffic(lb_url, request_rate, duration, max_concurrent=50)
        finally:
            # Properly shut down the server - let it exit gracefully
            server.should_exit = True
            
            # Wait for server to exit gracefully (don't cancel immediately)
            if not lb_task.done():
                try:
                    # Give server time to exit gracefully (it checks should_exit periodically)
                    await asyncio.wait_for(lb_task, timeout=5.0)
                except asyncio.TimeoutError:
                    # If graceful shutdown times out, force exit
                    server.force_exit = True
                    # Give it one more chance
                    try:
                        await asyncio.wait_for(lb_task, timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        # Last resort: cancel the task
                        lb_task.cancel()
                        try:
                            await asyncio.wait_for(lb_task, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
            
            # Cleanup proxy
            from load_balancer.main import proxy as global_proxy
            if global_proxy:
                try:
                    await global_proxy.close()
                except Exception:
                    pass
            
            # Wait for port to be released before proceeding
            if not await wait_for_port_release(self.lb_config['host'], self.lb_config['port'], timeout=10.0):
                print(f"Warning: Port {self.lb_config['port']} still in use after shutdown")
        
        print(f"Experiment {strategy} complete\n")
    
    async def run_full_experiment(
        self,
        num_requests: int = 1000,
        request_rate: float = 10.0,
        output_dir: Path = None
    ):
        """Run full comparison experiment."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Start servers
            await self.start_servers()
            
            # Create metrics collector (will be reused)
            server_urls = [f"http://{s['host']}:{s['port']}" for s in self.servers]
            metrics_collector = MetricsCollector(server_urls, [s['id'] for s in self.servers])
            await metrics_collector.start_polling()
            
            # Run Round Robin experiment
            await self.run_strategy_experiment("round_robin", metrics_collector, num_requests, request_rate)
            rr_logs = metrics_collector.get_request_logs_df().copy()
            rr_logs['routing_strategy'] = 'round_robin'  # Ensure label
            
            # Wait between experiments
            await asyncio.sleep(2)
            
            # Run RL agent experiment (metrics will be appended)
            await self.run_strategy_experiment("rl_agent", metrics_collector, num_requests, request_rate)
            rl_logs = metrics_collector.get_request_logs_df().copy()
            # Get only the new logs (after Round Robin)
            if len(rl_logs) > len(rr_logs):
                rl_logs_only = rl_logs.iloc[len(rr_logs):].copy()
                rl_logs_only['routing_strategy'] = 'rl_agent'
                all_logs = pd.concat([rr_logs, rl_logs_only], ignore_index=True)
            else:
                all_logs = metrics_collector.get_request_logs_df()
            
            # Stop metrics collection
            await metrics_collector.stop_polling()
            
            # Analyze results
            print("Analyzing results...")
            analyzer = MetricsAnalyzer(all_logs)
            comparison = analyzer.compare_strategies()
            
            # Export results
            exporter = ResultsExporter(output_dir)
            exporter.export_json(comparison, "comparison_results.json")
            exporter.export_csv(all_logs, "request_logs.csv")
            exporter.export_summary_report(comparison, "summary_report.md")
            
            # Create visualizations
            print("Creating visualizations...")
            create_all_visualizations(all_logs, comparison, output_dir)
            
            print(f"\n{'='*70}")
            print("EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"\nResults saved to: {output_dir}")
            print(f"\nSummary:")
            rr = comparison['round_robin']
            rl = comparison['rl_agent']
            improvement = comparison['improvement']
            print(f"  Round Robin - Mean Latency: {rr['latency_stats']['mean']:.2f} ms")
            print(f"  RL Agent - Mean Latency: {rl['latency_stats']['mean']:.2f} ms")
            print(f"  Improvement: {improvement['latency_reduction_percent']:.2f}%")
            print(f"\n")
            
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup processes."""
        print("\nCleaning up...")
        for process in self.server_processes:
            process.terminate()
            process.wait()
        self.server_processes.clear()
        print("Cleanup complete")


async def main():
    """Main entry point."""
    runner = ExperimentRunner()
    
    # Run experiment
    await runner.run_full_experiment(
        num_requests=1000,
        request_rate=10.0
    )


if __name__ == "__main__":
    asyncio.run(main())
