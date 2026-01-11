"""
Tests for metrics analyzer.
"""
import pytest
import pandas as pd
import numpy as np
from evaluation.metrics_analyzer import MetricsAnalyzer


@pytest.fixture
def sample_request_logs():
    """Sample request logs for testing."""
    return pd.DataFrame({
        'request_id': [f'req{i}' for i in range(10)],
        'timestamp': [i * 0.1 for i in range(10)],
        'routing_strategy': ['round_robin'] * 5 + ['rl_agent'] * 5,
        'selected_server': ['server1', 'server2', 'server3'] * 3 + ['server1'],
        'response_time_ms': [100.0, 110.0, 120.0, 105.0, 115.0, 90.0, 95.0, 85.0, 100.0, 92.0],
        'status_code': [200] * 10,
        'success': [True] * 10
    })


@pytest.mark.unit
def test_metrics_analyzer_initialization(sample_request_logs):
    """Test MetricsAnalyzer initialization."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    assert len(analyzer.df) == 10


@pytest.mark.unit
def test_calculate_latency_stats(sample_request_logs):
    """Test latency statistics calculation."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    stats = analyzer.calculate_latency_stats()
    
    assert 'mean' in stats
    assert 'p50' in stats
    assert 'p95' in stats
    assert 'p99' in stats
    assert stats['mean'] > 0
    assert stats['p50'] > 0


@pytest.mark.unit
def test_calculate_latency_stats_by_strategy(sample_request_logs):
    """Test latency statistics calculation for specific strategy."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    rr_stats = analyzer.calculate_latency_stats("round_robin")
    rl_stats = analyzer.calculate_latency_stats("rl_agent")
    
    assert 'mean' in rr_stats
    assert 'mean' in rl_stats
    assert rr_stats['mean'] > 0
    assert rl_stats['mean'] > 0


@pytest.mark.unit
def test_calculate_success_rate(sample_request_logs):
    """Test success rate calculation."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    success_rate = analyzer.calculate_success_rate()
    
    assert 0 <= success_rate <= 1
    assert success_rate == 1.0  # All requests are successful in sample


@pytest.mark.unit
def test_calculate_server_distribution(sample_request_logs):
    """Test server distribution calculation."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    distribution = analyzer.calculate_server_distribution()
    
    assert isinstance(distribution, dict)
    assert 'server1' in distribution
    assert 'server2' in distribution
    assert 'server3' in distribution


@pytest.mark.unit
def test_calculate_throughput(sample_request_logs):
    """Test throughput calculation."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    throughput_data = analyzer.calculate_throughput(window_size=3)
    
    assert isinstance(throughput_data, list)
    if throughput_data:
        assert 'throughput' in throughput_data[0]
        assert 'avg_latency' in throughput_data[0]


@pytest.mark.unit
def test_compare_strategies(sample_request_logs):
    """Test strategy comparison."""
    analyzer = MetricsAnalyzer(sample_request_logs)
    comparison = analyzer.compare_strategies()
    
    assert 'round_robin' in comparison
    assert 'rl_agent' in comparison
    assert 'improvement' in comparison
    assert 'latency_stats' in comparison['round_robin']
    assert 'latency_stats' in comparison['rl_agent']
