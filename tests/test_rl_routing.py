"""
Tests for RL-based routing.
"""
import pytest
from unittest.mock import Mock, MagicMock
from load_balancer.rl_routing import RLRouter
from metrics.collector import MetricsCollector


@pytest.fixture
def sample_servers():
    """Sample server configuration."""
    return [
        {"id": "server1", "host": "localhost", "port": 8001},
        {"id": "server2", "host": "localhost", "port": 8002},
        {"id": "server3", "host": "localhost", "port": 8003},
    ]


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector."""
    collector = Mock(spec=MetricsCollector)
    collector.get_latest_server_metrics.return_value = {
        "server1": {
            "cpu_utilization": 30.0,
            "active_requests": 2,
            "avg_response_time": 50.0
        },
        "server2": {
            "cpu_utilization": 60.0,
            "active_requests": 5,
            "avg_response_time": 100.0
        },
        "server3": {
            "cpu_utilization": 90.0,
            "active_requests": 15,
            "avg_response_time": 300.0
        }
    }
    return collector


@pytest.mark.unit
def test_rl_router_initialization(sample_servers, mock_metrics_collector):
    """Test RLRouter initialization."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    assert router.servers == sample_servers
    assert router.metrics_collector == mock_metrics_collector
    assert router.agent is not None
    assert router.state_encoder is not None


@pytest.mark.unit
def test_rl_router_select_server(sample_servers, mock_metrics_collector):
    """Test server selection using RL agent."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    
    # Select a server
    server = router.select_server()
    
    # Should return one of the servers
    assert server in sample_servers
    assert 'id' in server
    assert 'host' in server
    assert 'port' in server


@pytest.mark.unit
def test_rl_router_update_reward(sample_servers, mock_metrics_collector):
    """Test updating reward."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    
    # Select a server first (this sets last_state and last_action)
    router.select_server()
    
    # Update reward
    router.update_reward(response_time_ms=100.0, success=True)
    
    # Should not raise an exception
    # The agent's Q-table should be updated


@pytest.mark.unit
def test_rl_router_get_server_url(sample_servers, mock_metrics_collector):
    """Test server URL construction."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    server = sample_servers[0]
    url = router.get_server_url(server)
    assert url == "http://localhost:8001"


@pytest.mark.unit
def test_rl_router_get_all_servers(sample_servers, mock_metrics_collector):
    """Test getting all servers."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    all_servers = router.get_all_servers()
    assert all_servers == sample_servers


@pytest.mark.unit
def test_rl_router_get_statistics(sample_servers, mock_metrics_collector):
    """Test getting router statistics."""
    router = RLRouter(sample_servers, mock_metrics_collector)
    stats = router.get_statistics()
    assert isinstance(stats, dict)
    assert 'num_states' in stats
    assert 'episode_count' in stats
