# System Architecture

## Overview

The Smart Load Balancer system consists of multiple components working together to provide adaptive request routing using Reinforcement Learning. This document describes the system architecture, data flows, and component interactions.

## System Components

### 1. Backend Servers

**Location**: `backend/server.py`

FastAPI-based HTTP servers that simulate backend application servers.

**Features**:
- Configurable response delays (base_delay_ms)
- CPU utilization monitoring (psutil)
- Active request tracking
- Health endpoint (`/health`)
- Response time calculation

**Metrics Exposed**:
- CPU utilization (%)
- Active request count
- Average response time (ms)
- Health status (healthy/degraded/unhealthy)

### 2. Load Balancer

**Location**: `load_balancer/main.py`

FastAPI application that acts as a reverse proxy, routing client requests to backend servers.

**Components**:
- **Proxy** (`load_balancer/proxy.py`): HTTP request forwarding
- **Router Interface**: Strategy pattern for routing algorithms
  - Round Robin (`load_balancer/round_robin.py`)
  - RL Agent (`load_balancer/rl_routing.py`)

**Responsibilities**:
- Accept incoming HTTP requests
- Select backend server using routing strategy
- Forward requests and responses
- Log request metrics
- Update RL agent with rewards (if using RL routing)

### 3. Metrics Collector

**Location**: `metrics/collector.py`

Collects and stores system metrics from servers and requests.

**Data Collected**:
- Server health metrics (periodic polling)
- Request performance logs (per-request)

**Storage**:
- In-memory: Pandas DataFrames
- Thread-safe operations
- Exportable to CSV/JSON

### 4. RL Agent

**Location**: `rl/q_learning.py`, `rl/state_encoder.py`

Q-Learning-based reinforcement learning agent for server selection.

**Components**:
- **State Encoder**: Converts server metrics to discrete states
- **Q-Learning Agent**: Implements Q-Learning algorithm
  - Q-table (state → action values)
  - ε-greedy exploration/exploitation
  - Bellman equation updates
  - Model persistence (save/load)

**State Space**:
- Discrete states: `(server1_load, server2_load, server3_load)`
- Load levels: 0 (low), 1 (medium), 2 (high)
- State space size: 3^N (N = number of servers)

**Action Space**:
- Actions: Server indices (0, 1, 2, ...)
- Number of actions: Number of servers

**Reward Function**:
- Primary: Minimize response time
- Reward = -response_time_ms / 10.0
- Failure penalty: -1000.0

### 5. Evaluation Framework

**Location**: `evaluation/`

Tools for systematic performance evaluation and comparison.

**Components**:
- **Metrics Analyzer** (`metrics_analyzer.py`): Statistical analysis
- **Results Exporter** (`export.py`): Export to JSON, CSV, Markdown
- **Evaluation Runner** (`runner.py`): Orchestrates experiments

**Metrics Calculated**:
- Latency statistics (mean, percentiles, std dev)
- Throughput over time
- Success rates
- Server distribution
- Learning curves
- Coefficient of variation (stability)

### 6. Visualization

**Location**: `visualization/plots.py`

Generates plots and charts for performance analysis.

**Visualizations**:
- Latency over time (moving average)
- Throughput comparison
- Server utilization (request distribution)
- Learning curve (RL agent)
- Latency distribution (histogram)
- Comparison summary (multi-panel)

### 7. Traffic Generator

**Location**: `traffic/generator.py`, `traffic/locustfile.py`

Generates synthetic HTTP traffic for load testing.

**Features**:
- Configurable request rate
- Duration control
- Concurrent request support
- Performance statistics

## Data Flow

### Request Flow

```
1. Client → Load Balancer (HTTP request)
2. Load Balancer → Router.select_server()
   - Round Robin: Sequential selection
   - RL Agent: Q-Learning policy selection
3. Load Balancer → Proxy.forward_request()
4. Proxy → Backend Server (HTTP request)
5. Backend Server → Proxy (HTTP response)
6. Proxy → Load Balancer (response data)
7. Load Balancer → MetricsCollector.log_request()
8. Load Balancer → RL Agent.update_reward() (if RL routing)
9. Load Balancer → Client (HTTP response)
```

### Metrics Collection Flow

```
1. MetricsCollector.start_polling() (background task)
2. Periodic polling loop:
   - For each server:
     - HTTP GET /health
     - Parse metrics
     - Store in server_metrics list
3. Request logging:
   - On each request completion:
     - Log request_id, strategy, server, latency, status
     - Store in request_logs list
```

### RL Learning Flow

```
1. Request arrives → RL Router.select_server()
2. State Encoder:
   - Fetch latest server metrics
   - Encode to discrete state tuple
3. Q-Learning Agent:
   - Lookup Q-values for state
   - Select action (server) using ε-greedy
4. Request executes → Response received
5. Reward Calculation:
   - Calculate reward from response time
6. Q-Value Update:
   - Observe next state
   - Update Q-table using Bellman equation
```

## Configuration

### Server Configuration

**File**: `config/config.yaml`

```yaml
servers:
  - id: "server1"
    host: "localhost"
    port: 8001
    base_delay_ms: 50
```

### RL Configuration

**File**: `config/rl_config.yaml`

```yaml
q_learning:
  learning_rate: 0.1
  discount_factor: 0.9
  exploration_rate: 0.1
  exploration_decay: 0.995

state_encoding:
  cpu_thresholds:
    low: 50.0
    medium: 80.0
  active_requests_thresholds:
    low: 5
    medium: 10
  response_time_thresholds_ms:
    low: 100.0
    medium: 200.0
```

## Concurrency Model

- **Backend Servers**: Async FastAPI (uvicorn)
- **Load Balancer**: Async FastAPI (uvicorn)
- **Metrics Collector**: Async polling task (asyncio)
- **Traffic Generator**: Async HTTP client (httpx)
- **RL Agent**: Thread-safe operations (locks for Q-table updates)

## State Management

### Global State (Load Balancer)

- `proxy`: HTTP client for forwarding
- `router`: Routing strategy instance
- `metrics_collector`: Metrics collection instance
- `routing_strategy`: Current strategy name

### RL Agent State

- `q_table`: Dictionary mapping states to action values
- `exploration_rate`: Current exploration probability
- `episode_count`: Number of training episodes
- `total_updates`: Number of Q-value updates

### Metrics State

- `server_metrics`: List of server health snapshots
- `request_logs`: List of request performance records
- Thread-safe access via locks

## Error Handling

- **Server Unreachable**: Metrics marked as "unreachable", RL agent avoids
- **Request Failures**: Logged with success=False, RL agent receives penalty
- **Timeout**: Proxy timeout (30s default), request marked as failure
- **State Encoding Errors**: Fallback to high load level

## Performance Considerations

- **Metrics Polling**: Configurable interval (default: 1s)
- **Async Operations**: Non-blocking I/O for all HTTP operations
- **Memory Management**: In-memory storage (suitable for experiments)
- **Q-Table Size**: Grows with state space (3^N states)

## Extensibility

The system is designed for extensibility:

1. **New Routing Strategies**: Implement router interface (select_server, get_server_url)
2. **New Metrics**: Extend MetricsCollector with new data sources
3. **New RL Algorithms**: Replace Q-Learning agent (interface: select_action, update)
4. **New Visualizations**: Add functions to visualization/plots.py
5. **New Backend Types**: Implement server interface (health endpoint)

## Limitations (MVP Scope)

- Single-machine deployment (local simulation)
- HTTP only (no HTTPS)
- In-memory metrics storage (not persistent)
- Tabular Q-Learning (discrete states only)
- No server failure recovery
- No distributed deployment

## Future Architecture Enhancements

- Distributed metrics storage (Redis, database)
- Deep RL (DQN, PPO) for continuous states
- Kubernetes integration
- Service mesh integration
- Multi-region support
- Real-time streaming analytics
