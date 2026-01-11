"""
Tests for metrics collector.
"""
import pytest
import asyncio
from metrics.collector import MetricsCollector


@pytest.fixture
def metrics_collector():
    """Create a metrics collector for testing."""
    server_urls = ["http://localhost:8001", "http://localhost:8002"]
    server_ids = ["server1", "server2"]
    return MetricsCollector(server_urls, server_ids, polling_interval=0.1)


@pytest.mark.unit
def test_metrics_collector_initialization(metrics_collector):
    """Test MetricsCollector initialization."""
    assert len(metrics_collector.server_urls) == 2
    assert len(metrics_collector.server_ids) == 2
    assert metrics_collector.polling_interval == 0.1
    assert not metrics_collector.is_polling


@pytest.mark.unit
def test_log_request(metrics_collector):
    """Test logging a request."""
    metrics_collector.log_request(
        request_id="req1",
        routing_strategy="round_robin",
        selected_server="server1",
        response_time_ms=100.0,
        status_code=200,
        success=True
    )
    
    logs_df = metrics_collector.get_request_logs_df()
    assert len(logs_df) == 1
    assert logs_df.iloc[0]['request_id'] == "req1"
    assert logs_df.iloc[0]['routing_strategy'] == "round_robin"
    assert logs_df.iloc[0]['selected_server'] == "server1"
    assert logs_df.iloc[0]['response_time_ms'] == 100.0
    assert logs_df.iloc[0]['success'] == True  # Use == instead of is for pandas boolean


@pytest.mark.unit
def test_get_latest_server_metrics(metrics_collector):
    """Test getting latest server metrics."""
    # Add some metrics
    metrics_collector.server_metrics = [
        {
            "server_id": "server1",
            "timestamp": 1.0,
            "cpu_utilization": 50.0,
            "active_requests": 5,
            "avg_response_time": 100.0,
            "health_status": "healthy"
        },
        {
            "server_id": "server1",
            "timestamp": 2.0,
            "cpu_utilization": 60.0,
            "active_requests": 6,
            "avg_response_time": 110.0,
            "health_status": "degraded"
        },
        {
            "server_id": "server2",
            "timestamp": 1.5,
            "cpu_utilization": 40.0,
            "active_requests": 3,
            "avg_response_time": 90.0,
            "health_status": "healthy"
        }
    ]
    
    latest = metrics_collector.get_latest_server_metrics()
    assert "server1" in latest
    assert "server2" in latest
    # server1 should have the latest timestamp (2.0)
    assert latest["server1"]["cpu_utilization"] == 60.0
    assert latest["server2"]["cpu_utilization"] == 40.0


@pytest.mark.unit
def test_clear_metrics(metrics_collector):
    """Test clearing metrics."""
    # Add some data
    metrics_collector.log_request("req1", "round_robin", "server1", 100.0, 200, True)
    metrics_collector.server_metrics.append({"server_id": "server1", "timestamp": 1.0})
    
    # Clear
    metrics_collector.clear_metrics()
    
    assert len(metrics_collector.server_metrics) == 0
    assert len(metrics_collector.get_request_logs_df()) == 0


@pytest.mark.asyncio
async def test_start_stop_polling(metrics_collector):
    """Test starting and stopping polling."""
    # Start polling
    await metrics_collector.start_polling()
    assert metrics_collector.is_polling
    
    # Wait a bit
    await asyncio.sleep(0.2)
    
    # Stop polling
    await metrics_collector.stop_polling()
    assert not metrics_collector.is_polling


@pytest.mark.unit
def test_get_server_metrics_df_empty(metrics_collector):
    """Test getting empty server metrics DataFrame."""
    df = metrics_collector.get_server_metrics_df()
    assert len(df) == 0


@pytest.mark.unit
def test_get_request_logs_df_empty(metrics_collector):
    """Test getting empty request logs DataFrame."""
    df = metrics_collector.get_request_logs_df()
    assert len(df) == 0
