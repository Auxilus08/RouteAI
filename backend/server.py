"""
FastAPI backend server with health monitoring and configurable response delays.
"""
import asyncio
import time
import threading
from typing import Dict, List
from collections import defaultdict
from datetime import datetime

import psutil
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Global state
server_id: str = ""
server_port: int = 0
base_delay_ms: float = 50.0
active_requests: int = 0
request_lock = threading.Lock()
response_times: List[float] = []


def get_cpu_utilization() -> float:
    """Get current CPU utilization percentage."""
    return psutil.cpu_percent(interval=0.1)


def get_avg_response_time() -> float:
    """Calculate average response time from recent requests."""
    if not response_times:
        return 0.0
    # Use last 100 response times for rolling average
    recent_times = response_times[-100:]
    return sum(recent_times) / len(recent_times)


def get_health_status(cpu: float, active_reqs: int, avg_rt: float) -> str:
    """Determine health status based on metrics."""
    if cpu > 80 or active_reqs > 20 or avg_rt > 500:
        return "unhealthy"
    elif cpu > 50 or active_reqs > 10 or avg_rt > 200:
        return "degraded"
    return "healthy"


@app.get("/")
async def root():
    """Root endpoint that simulates work with configurable delay."""
    start_time = time.time()
    
    with request_lock:
        global active_requests
        active_requests += 1
    
    try:
        # Simulate processing delay
        await asyncio.sleep(base_delay_ms / 1000.0)
        
        # Add some CPU variability based on current load
        cpu_load = get_cpu_utilization()
        if cpu_load > 50:
            # Add extra delay when CPU is high
            await asyncio.sleep((cpu_load - 50) / 1000.0)
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        with request_lock:
            active_requests -= 1
            response_times.append(response_time)
        
        return JSONResponse(
            content={
                "server_id": server_id,
                "message": "Request processed",
                "response_time_ms": round(response_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        with request_lock:
            active_requests -= 1
        raise


@app.get("/health")
async def health():
    """Health endpoint returning server metrics."""
    cpu = get_cpu_utilization()
    
    with request_lock:
        active_reqs = active_requests
        avg_rt = get_avg_response_time()
    
    health_status = get_health_status(cpu, active_reqs, avg_rt)
    
    return JSONResponse(
        content={
            "server_id": server_id,
            "timestamp": time.time(),
            "cpu_utilization": round(cpu, 2),
            "active_requests": active_reqs,
            "avg_response_time": round(avg_rt, 2),
            "health_status": health_status
        }
    )


def create_server(server_id_param: str, port: int, delay_ms: float):
    """Create and configure a server instance."""
    global server_id, server_port, base_delay_ms
    server_id = server_id_param
    server_port = port
    base_delay_ms = delay_ms
    
    return app


def run_server(server_id_param: str, port: int, delay_ms: float):
    """Run a server instance."""
    global server_id, server_port, base_delay_ms
    server_id = server_id_param
    server_port = port
    base_delay_ms = delay_ms
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python server.py <server_id> <port> <base_delay_ms>")
        sys.exit(1)
    
    server_id_arg = sys.argv[1]
    port_arg = int(sys.argv[2])
    delay_arg = float(sys.argv[3])
    
    run_server(server_id_arg, port_arg, delay_arg)
