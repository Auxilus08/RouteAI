"""
Core HTTP proxy functionality for the load balancer.
"""
import time
import httpx
from typing import Optional, Dict, Any
from urllib.parse import urlparse


class Proxy:
    """HTTP proxy for forwarding requests to backend servers."""
    
    def __init__(self, timeout: float = 30.0):
        """
        Initialize the proxy.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
    
    async def forward_request(
        self,
        server_url: str,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Forward a request to a backend server.
        
        Args:
            server_url: Base URL of the backend server
            method: HTTP method (GET, POST, etc.)
            path: Request path
            headers: Request headers
            body: Request body (if any)
        
        Returns:
            Dictionary containing response data and timing information
        """
        full_url = f"{server_url}{path}"
        start_time = time.time()
        
        try:
            response = await self.client.request(
                method=method,
                url=full_url,
                headers=headers,
                content=body
            )
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Read response body
            response_body = await response.aread()
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body,
                "response_time_ms": response_time,
                "success": 200 <= response.status_code < 400,
                "error": None
            }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "status_code": 0,
                "headers": {},
                "body": b"",
                "response_time_ms": response_time,
                "success": False,
                "error": str(e)
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
