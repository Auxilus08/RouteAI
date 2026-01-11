"""
Main load balancer application.
"""
import asyncio
import time
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

from load_balancer.proxy import Proxy
from load_balancer.round_robin import RoundRobinRouter
from load_balancer.rl_routing import RLRouter
from metrics.collector import MetricsCollector


app = FastAPI(title="Smart Load Balancer")

# Global state
proxy: Optional[Proxy] = None
router: Optional[Any] = None  # Can be RoundRobinRouter or RLRouter
metrics_collector: Optional[MetricsCollector] = None
routing_strategy: str = "round_robin"  # "round_robin" or "rl_agent"


def initialize_load_balancer(servers: list, strategy: str = "round_robin", metrics_collector_instance: Optional[MetricsCollector] = None):
    """Initialize load balancer components."""
    global proxy, router, metrics_collector, routing_strategy
    
    routing_strategy = strategy
    proxy = Proxy()
    
    # Use provided metrics collector or create new one
    if metrics_collector_instance:
        metrics_collector = metrics_collector_instance
    else:
        server_urls = [f"http://{s['host']}:{s['port']}" for s in servers]
        metrics_collector = MetricsCollector(server_urls, [s['id'] for s in servers])
        # Start metrics collection
        asyncio.create_task(metrics_collector.start_polling())
    
    # Initialize router based on strategy
    if strategy == "rl_agent":
        router = RLRouter(servers, metrics_collector)
    else:
        router = RoundRobinRouter(servers)


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # This will be set by the runner
    pass


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global proxy, metrics_collector
    if proxy:
        await proxy.close()
    if metrics_collector:
        await metrics_collector.stop_polling()


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_request(request: Request, path: str):
    """
    Proxy requests to backend servers using the selected routing strategy.
    """
    global proxy, router, metrics_collector, routing_strategy
    
    if not proxy or not router:
        return Response(
            status_code=503,
            content="Load balancer not initialized"
        )
    
    # Select server based on strategy
    selected_server = router.select_server()
    server_url = router.get_server_url(selected_server)
    
    # Get request body if present
    body = await request.body() if request.method in ["POST", "PUT", "PATCH"] else None
    
    # Forward request
    result = await proxy.forward_request(
        server_url=server_url,
        method=request.method,
        path=f"/{path}" if path else "/",
        headers=dict(request.headers),
        body=body
    )
    
    # Log metrics
    request_id = str(uuid.uuid4())
    if metrics_collector:
        metrics_collector.log_request(
            request_id=request_id,
            routing_strategy=routing_strategy,
            selected_server=selected_server['id'],
            response_time_ms=result['response_time_ms'],
            status_code=result['status_code'],
            success=result['success']
        )
    
    # Update RL agent with reward if using RL routing
    if routing_strategy == "rl_agent" and isinstance(router, RLRouter):
        router.update_reward(result['response_time_ms'], result['success'])
    
    # Return response
    response = Response(
        status_code=result['status_code'],
        content=result['body'],
        headers=result['headers']
    )
    return response


@app.get("/health")
async def health():
    """Load balancer health check."""
    return {
        "status": "healthy",
        "routing_strategy": routing_strategy,
        "servers_count": len(router.get_all_servers()) if router else 0
    }


def run(host: str = "0.0.0.0", port: int = 8080, servers: list = None, strategy: str = "round_robin"):
    """Run the load balancer."""
    if servers:
        initialize_load_balancer(servers, strategy)
    
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    run(
        host=config['load_balancer']['host'],
        port=config['load_balancer']['port'],
        servers=config['servers'],
        strategy="round_robin"
    )
