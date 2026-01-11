"""
Script to launch multiple backend server instances.
"""
import subprocess
import sys
import time
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config():
    """Load server configuration."""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Launch all backend servers."""
    config = load_config()
    servers = config['servers']
    
    processes = []
    
    print("Starting backend servers...")
    for server in servers:
        server_id = server['id']
        port = server['port']
        delay = server['base_delay_ms']
        
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "server.py"),
            server_id,
            str(port),
            str(delay)
        ]
        
        print(f"Starting {server_id} on port {port} with delay {delay}ms")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append((server_id, process))
        time.sleep(1)  # Stagger server starts
    
    print(f"\nStarted {len(processes)} servers. Press Ctrl+C to stop all servers.")
    
    try:
        # Wait for all processes
        for server_id, process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        for server_id, process in processes:
            process.terminate()
            process.wait()
        print("All servers stopped.")


if __name__ == "__main__":
    main()
