"""
Tests for Round Robin routing algorithm.
"""
import pytest
from load_balancer.round_robin import RoundRobinRouter


@pytest.fixture
def sample_servers():
    """Sample server configuration for testing."""
    return [
        {"id": "server1", "host": "localhost", "port": 8001},
        {"id": "server2", "host": "localhost", "port": 8002},
        {"id": "server3", "host": "localhost", "port": 8003},
    ]


@pytest.mark.unit
def test_round_robin_initialization(sample_servers):
    """Test RoundRobinRouter initialization."""
    router = RoundRobinRouter(sample_servers)
    assert router.servers == sample_servers
    assert router.current_index == 0


@pytest.mark.unit
def test_round_robin_server_selection(sample_servers):
    """Test that Round Robin selects servers in order."""
    router = RoundRobinRouter(sample_servers)
    
    # First selection
    server1 = router.select_server()
    assert server1 == sample_servers[0]
    
    # Second selection
    server2 = router.select_server()
    assert server2 == sample_servers[1]
    
    # Third selection
    server3 = router.select_server()
    assert server3 == sample_servers[2]
    
    # Fourth selection (should wrap around)
    server4 = router.select_server()
    assert server4 == sample_servers[0]


@pytest.mark.unit
def test_round_robin_get_server_url(sample_servers):
    """Test server URL construction."""
    router = RoundRobinRouter(sample_servers)
    server = sample_servers[0]
    url = router.get_server_url(server)
    assert url == "http://localhost:8001"


@pytest.mark.unit
def test_round_robin_get_all_servers(sample_servers):
    """Test getting all servers."""
    router = RoundRobinRouter(sample_servers)
    all_servers = router.get_all_servers()
    assert all_servers == sample_servers
    assert len(all_servers) == 3


@pytest.mark.unit
def test_round_robin_thread_safety(sample_servers):
    """Test that Round Robin is thread-safe."""
    import threading
    
    router = RoundRobinRouter(sample_servers)
    selections = []
    lock = threading.Lock()
    
    def select_and_append():
        server = router.select_server()
        with lock:
            selections.append(server['id'])
    
    # Create multiple threads
    threads = [threading.Thread(target=select_and_append) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # All selections should be valid server IDs
    assert len(selections) == 10
    valid_ids = {s['id'] for s in sample_servers}
    assert all(sid in valid_ids for sid in selections)
