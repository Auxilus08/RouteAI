"""
Tests for state encoder.
"""
import pytest
from pathlib import Path
from rl.state_encoder import StateEncoder


@pytest.fixture
def state_encoder():
    """Create a state encoder for testing."""
    config_path = Path(__file__).parent.parent / "config" / "rl_config.yaml"
    return StateEncoder(config_path)


@pytest.mark.unit
def test_state_encoder_initialization(state_encoder):
    """Test StateEncoder initialization."""
    assert state_encoder.cpu_thresholds is not None
    assert state_encoder.active_requests_thresholds is not None
    assert state_encoder.response_time_thresholds is not None
    assert len(state_encoder.load_levels) == 3


@pytest.mark.unit
def test_encode_load_level_low(state_encoder):
    """Test encoding low load level."""
    # Low CPU, low requests, low response time
    load_level = state_encoder.encode_load_level(
        cpu=30.0,
        active_requests=2,
        response_time=50.0
    )
    assert load_level == 0  # Low


@pytest.mark.unit
def test_encode_load_level_medium(state_encoder):
    """Test encoding medium load level."""
    # Medium CPU
    load_level = state_encoder.encode_load_level(
        cpu=60.0,  # Between low and high threshold
        active_requests=2,
        response_time=50.0
    )
    assert load_level == 1  # Medium


@pytest.mark.unit
def test_encode_load_level_high(state_encoder):
    """Test encoding high load level."""
    # High CPU
    load_level = state_encoder.encode_load_level(
        cpu=90.0,
        active_requests=15,
        response_time=300.0
    )
    assert load_level == 2  # High


@pytest.mark.unit
def test_encode_state(state_encoder):
    """Test encoding full state from server metrics."""
    server_metrics = {
        "server1": {
            "cpu_utilization": 30.0,
            "active_requests": 2,
            "avg_response_time": 50.0
        },
        "server2": {
            "cpu_utilization": 70.0,
            "active_requests": 8,
            "avg_response_time": 150.0
        },
        "server3": {
            "cpu_utilization": 90.0,
            "active_requests": 15,
            "avg_response_time": 300.0
        }
    }
    server_ids = ["server1", "server2", "server3"]
    
    state = state_encoder.encode_state(server_metrics, server_ids)
    assert isinstance(state, tuple)
    assert len(state) == 3
    assert state[0] == 0  # server1: low
    assert state[1] == 1  # server2: medium
    assert state[2] == 2  # server3: high


@pytest.mark.unit
def test_encode_state_missing_server(state_encoder):
    """Test encoding state when a server is missing."""
    server_metrics = {
        "server1": {
            "cpu_utilization": 30.0,
            "active_requests": 2,
            "avg_response_time": 50.0
        }
    }
    server_ids = ["server1", "server2", "server3"]
    
    state = state_encoder.encode_state(server_metrics, server_ids)
    assert state[0] == 0  # server1: low
    assert state[1] == 2  # server2: high (unreachable)
    assert state[2] == 2  # server3: high (unreachable)


@pytest.mark.unit
def test_get_state_space_size(state_encoder):
    """Test state space size calculation."""
    size = state_encoder.get_state_space_size(num_servers=3)
    assert size == 3 ** 3  # 27 states
