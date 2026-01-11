"""
Tests for HTTP proxy.
"""
import pytest
from load_balancer.proxy import Proxy


@pytest.mark.unit
def test_proxy_initialization():
    """Test Proxy initialization."""
    proxy = Proxy(timeout=10.0)
    assert proxy.timeout == 10.0
    assert proxy.client is not None


@pytest.mark.asyncio
async def test_proxy_close():
    """Test proxy cleanup."""
    proxy = Proxy()
    await proxy.close()
    # Should not raise an exception
