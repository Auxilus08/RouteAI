"""
Metrics collection system for server health and request performance.
"""
import asyncio
import time
import httpx
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import threading
from collections import defaultdict


class MetricsCollector:
    """Collects and stores metrics from servers and requests."""
    
    def __init__(
        self,
        server_urls: List[str],
        server_ids: List[str],
        polling_interval: float = 1.0
    ):
        """
        Initialize metrics collector.
        
        Args:
            server_urls: List of server base URLs
            server_ids: List of server IDs (same order as URLs)
            polling_interval: How often to poll server health (seconds)
        """
        self.server_urls = server_urls
        self.server_ids = server_ids
        self.polling_interval = polling_interval
        self.client = httpx.AsyncClient(timeout=5.0)
        
        # Data storage
        self.server_metrics: List[Dict[str, Any]] = []
        self.request_logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Polling control
        self.polling_task: Optional[asyncio.Task] = None
        self.is_polling = False
    
    async def fetch_server_health(self, server_url: str, server_id: str) -> Optional[Dict[str, Any]]:
        """Fetch health metrics from a server."""
        try:
            response = await self.client.get(f"{server_url}/health")
            if response.status_code == 200:
                data = response.json()
                return {
                    "server_id": server_id,
                    "timestamp": time.time(),
                    "cpu_utilization": data.get("cpu_utilization", 0.0),
                    "active_requests": data.get("active_requests", 0),
                    "avg_response_time": data.get("avg_response_time", 0.0),
                    "health_status": data.get("health_status", "unknown")
                }
        except Exception as e:
            # Server might be down or unreachable
            return {
                "server_id": server_id,
                "timestamp": time.time(),
                "cpu_utilization": 0.0,
                "active_requests": 0,
                "avg_response_time": 0.0,
                "health_status": "unreachable"
            }
        return None
    
    async def poll_servers(self):
        """Periodically poll all servers for health metrics."""
        while self.is_polling:
            for server_url, server_id in zip(self.server_urls, self.server_ids):
                metrics = await self.fetch_server_health(server_url, server_id)
                if metrics:
                    with self.lock:
                        self.server_metrics.append(metrics)
            await asyncio.sleep(self.polling_interval)
    
    async def start_polling(self):
        """Start the polling task."""
        if not self.is_polling:
            self.is_polling = True
            self.polling_task = asyncio.create_task(self.poll_servers())
    
    async def stop_polling(self):
        """Stop the polling task."""
        self.is_polling = False
        if self.polling_task:
            self.polling_task.cancel()
            try:
                await self.polling_task
            except asyncio.CancelledError:
                pass
        await self.client.aclose()
    
    def log_request(
        self,
        request_id: str,
        routing_strategy: str,
        selected_server: str,
        response_time_ms: float,
        status_code: int,
        success: bool
    ):
        """
        Log a request's performance metrics.
        
        Args:
            request_id: Unique request identifier
            routing_strategy: Strategy used ("round_robin" or "rl_agent")
            selected_server: ID of the server that handled the request
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            success: Whether the request was successful
        """
        with self.lock:
            self.request_logs.append({
                "request_id": request_id,
                "timestamp": time.time(),
                "routing_strategy": routing_strategy,
                "selected_server": selected_server,
                "response_time_ms": response_time_ms,
                "status_code": status_code,
                "success": success
            })
    
    def get_server_metrics_df(self) -> pd.DataFrame:
        """Get server metrics as a pandas DataFrame."""
        with self.lock:
            if not self.server_metrics:
                return pd.DataFrame()
            return pd.DataFrame(self.server_metrics)
    
    def get_request_logs_df(self) -> pd.DataFrame:
        """Get request logs as a pandas DataFrame."""
        with self.lock:
            if not self.request_logs:
                return pd.DataFrame()
            return pd.DataFrame(self.request_logs)
    
    def get_latest_server_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the latest metrics for each server.
        
        Returns:
            Dictionary mapping server_id to latest metrics
        """
        with self.lock:
            if not self.server_metrics:
                return {}
            
            latest = {}
            for metric in reversed(self.server_metrics):
                server_id = metric["server_id"]
                if server_id not in latest:
                    latest[server_id] = metric
            return latest
    
    def clear_metrics(self):
        """Clear all collected metrics (useful for starting a new experiment)."""
        with self.lock:
            self.server_metrics.clear()
            self.request_logs.clear()
