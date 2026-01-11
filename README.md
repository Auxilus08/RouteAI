# Smart Load Balancer for Web Servers

A Reinforcement Learning-based load balancer that dynamically optimizes request routing based on real-time traffic patterns and server health metrics.

## Overview

This project implements an adaptive, self-learning load balancing mechanism using Q-Learning. The load balancer continuously observes system behavior, learns from routing outcomes, and improves future routing decisions to maintain optimal performance.

### Key Features

- **Multi-Server Backend**: 3+ FastAPI backend servers with health monitoring
- **Round Robin Baseline**: Traditional load balancing algorithm for comparison
- **RL-Based Routing**: Q-Learning agent that adapts to server conditions
- **Real-Time Metrics**: Server health monitoring (CPU, active requests, response time)
- **Performance Evaluation**: Comprehensive comparison framework with visualizations
- **Modular Architecture**: Clean, extensible codebase

## Architecture

```
┌─────────────┐
│   Client    │
│  (Traffic)  │
└──────┬──────┘
       │ HTTP Requests
       ▼
┌─────────────────────────────────────┐
│      Load Balancer (FastAPI)        │
│  ┌──────────────────────────────┐  │
│  │  Routing Strategy            │  │
│  │  - Round Robin (Baseline)    │  │
│  │  - RL Agent (Q-Learning)     │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  Metrics Collector           │  │
│  │  - Server Health Polling     │  │
│  │  - Request Performance       │  │
│  └──────────┬───────────────────┘  │
│             │                       │
│  ┌──────────▼───────────────────┐  │
│  │  RL Agent (Q-Learning)       │  │
│  │  - State: Server Health      │  │
│  │  - Action: Server Selection  │  │
│  │  - Reward: Response Time     │  │
│  └──────────────────────────────┘  │
└──────┬──────┬──────┬───────────────┘
       │      │      │
       ▼      ▼      ▼
  ┌────┐  ┌────┐  ┌────┐
  │ S1 │  │ S2 │  │ S3 │
  │FastAPI│FastAPI│FastAPI│
  └────┘  └────┘  └────┘
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python --version  # Should be 3.10+
   ```

## Configuration

Configuration files are located in the `config/` directory:

- **`config/config.yaml`**: Server endpoints, load balancer settings, traffic configuration
- **`config/rl_config.yaml`**: RL agent hyperparameters and state encoding thresholds

### Example Configuration

```yaml
# config/config.yaml
servers:
  - id: "server1"
    host: "localhost"
    port: 8001
    base_delay_ms: 50
  - id: "server2"
    host: "localhost"
    port: 8002
    base_delay_ms: 75
  - id: "server3"
    host: "localhost"
    port: 8003
    base_delay_ms: 100

load_balancer:
  host: "localhost"
  port: 8080
```

## Usage

### Quick Start: Run Full Experiment

Run a complete comparison experiment (Round Robin vs RL Agent):

```bash
python run_experiment.py
```

This will:
1. Start 3 backend servers
2. Run Round Robin experiment
3. Run RL Agent experiment
4. Generate comparison reports and visualizations
5. Save results to `reports/` directory

### Manual Execution

#### 1. Start Backend Servers

```bash
python backend/run_servers.py
```

Or start servers individually:
```bash
python backend/server.py server1 8001 50
python backend/server.py server2 8002 75
python backend/server.py server3 8003 100
```

#### 2. Start Load Balancer

**Round Robin mode:**
```bash
python load_balancer/main.py
```

**RL Agent mode:**
Modify `load_balancer/main.py` to use `strategy="rl_agent"` or use the experiment runner.

#### 3. Generate Traffic

**Using custom generator:**
```bash
python traffic/generator.py --url http://localhost:8080 --rate 10 --duration 60
```

**Using Locust:**
```bash
locust -f traffic/locustfile.py --host http://localhost:8080
```

### Running Experiments

The evaluation framework provides tools for systematic comparison:

```python
from evaluation.runner import EvaluationRunner
import asyncio

runner = EvaluationRunner()
asyncio.run(runner.run_comparison(
    num_requests_per_strategy=1000,
    request_rate=10.0
))
```

## Project Structure

```
smart-load-balancer/
├── backend/
│   ├── server.py           # FastAPI server implementation
│   └── run_servers.py      # Launch multiple server instances
├── load_balancer/
│   ├── main.py             # Load balancer FastAPI app
│   ├── proxy.py            # HTTP forwarding core
│   ├── round_robin.py      # Round Robin routing
│   └── rl_routing.py       # RL-based routing
├── rl/
│   ├── q_learning.py       # Q-Learning algorithm
│   └── state_encoder.py    # Metrics → discrete state
├── metrics/
│   └── collector.py        # Metrics collection & storage
├── traffic/
│   ├── generator.py        # Custom traffic generator
│   └── locustfile.py       # Locust traffic script
├── evaluation/
│   ├── runner.py           # Experiment execution
│   ├── metrics_analyzer.py # Metrics aggregation
│   └── export.py           # Results export
├── visualization/
│   └── plots.py            # Visualization scripts
├── config/
│   ├── config.yaml         # Main configuration
│   └── rl_config.yaml      # RL configuration
├── docs/
│   ├── architecture.md     # Architecture documentation
│   └── experimental_report.md
├── reports/                # Generated reports & visualizations
├── requirements.txt
└── README.md
```

## RL Agent Details

### State Representation

The RL agent uses a discrete state space:
- Each server's metrics are mapped to load levels: `low`, `medium`, `high`
- State tuple: `(server1_load, server2_load, server3_load)`
- Total state space: 3^N (where N = number of servers)

### Action Space

- Action: Server index (0, 1, 2, ...)
- Number of actions: Number of servers

### Reward Function

- Reward = -response_time_ms / 10.0 (minimize latency)
- Failure penalty: -1000.0

### Learning Algorithm

- Algorithm: Q-Learning (tabular)
- Exploration: ε-greedy (decays over time)
- Default hyperparameters:
  - Learning rate (α): 0.1
  - Discount factor (γ): 0.9
  - Initial exploration rate (ε): 0.1
  - Exploration decay: 0.995

## Evaluation Metrics

The system tracks and compares:

1. **Latency Statistics**: Mean, P50, P95, P99, standard deviation
2. **Throughput**: Requests per second over time
3. **Success Rate**: Percentage of successful requests
4. **Server Distribution**: Request distribution across servers
5. **Learning Curve**: RL agent performance over time
6. **Stability**: Coefficient of variation of response times

## Results

Results are saved to the `reports/` directory:

- **`comparison_results.json`**: Complete comparison data
- **`request_logs.csv`**: Request-level performance data
- **`summary_report.md`**: Human-readable summary
- **`*.png`**: Visualization plots:
  - `latency_over_time.png`
  - `throughput_comparison.png`
  - `server_utilization.png`
  - `learning_curve.png`
  - `latency_distribution.png`
  - `comparison_summary.png`

## Testing

The project includes a comprehensive test suite using pytest.

### Running Tests

```bash
# Install test dependencies (if not already installed)
pip install -r requirements.txt

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_round_robin.py

# Run tests with coverage
pytest --cov=. --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration
```

### Test Coverage

The test suite covers:
- Round Robin routing algorithm
- Q-Learning agent (initialization, action selection, Q-value updates)
- State encoder (state representation)
- Metrics collector (logging, polling)
- RL routing integration
- Metrics analyzer (statistics, comparisons)

### Writing New Tests

Tests are located in the `tests/` directory. Follow these conventions:
- Test files: `test_*.py`
- Test functions: `test_*`
- Use pytest fixtures for setup
- Mark slow tests with `@pytest.mark.slow`
- Mark integration tests with `@pytest.mark.integration`

## Troubleshooting

### Servers Not Starting

- Check if ports are already in use: `lsof -i :8001`
- Ensure Python 3.10+ is being used
- Check server logs for errors

### Load Balancer Connection Issues

- Verify servers are running: `curl http://localhost:8001/health`
- Check load balancer port: `curl http://localhost:8080/health`
- Ensure firewall allows local connections

### RL Agent Not Learning

- Check metrics collection is working
- Verify state encoding thresholds in `config/rl_config.yaml`
- Increase number of requests for training
- Check exploration rate decay

## Performance Expectations

Under typical conditions:
- **Latency Reduction**: 15-30% improvement over Round Robin
- **Learning Convergence**: Visible improvement after 1000+ requests
- **Stability**: Lower coefficient of variation compared to baseline

## Future Work

Potential extensions (not in MVP scope):

- Deep Q-Network (DQN) for continuous state spaces
- Server failure simulation and recovery
- Docker containerization
- Real-time dashboard (Streamlit/Grafana)
- Kubernetes integration
- Multi-region load balancing
- HTTPS/TLS termination

## License

This project is provided as-is for educational and research purposes.

## Contributing

This is an MVP implementation. Suggestions and improvements are welcome!

## References

- Q-Learning algorithm: Reinforcement Learning fundamentals
- FastAPI: Modern Python web framework
- Load Balancing: Distributed systems principles
