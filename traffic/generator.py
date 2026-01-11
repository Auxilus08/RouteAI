"""
Traffic generator for load testing the load balancer.
"""
import asyncio
import time
import httpx
from typing import Optional
import argparse


async def send_request(client: httpx.AsyncClient, url: str, request_id: int) -> dict:
    """Send a single HTTP request."""
    start_time = time.time()
    try:
        response = await client.get(url)
        response_time = (time.time() - start_time) * 1000
        return {
            "request_id": request_id,
            "status_code": response.status_code,
            "response_time_ms": response_time,
            "success": 200 <= response.status_code < 400
        }
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "request_id": request_id,
            "status_code": 0,
            "response_time_ms": response_time,
            "success": False,
            "error": str(e)
        }


async def generate_traffic(
    target_url: str,
    request_rate: float = 10.0,
    duration_seconds: float = 60.0,
    max_concurrent: int = 100
):
    """
    Generate HTTP traffic to the target URL.
    
    Args:
        target_url: URL to send requests to
        request_rate: Requests per second
        duration_seconds: How long to generate traffic
        max_concurrent: Maximum concurrent requests
    """
    client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_connections=max_concurrent))
    
    interval = 1.0 / request_rate  # Time between requests
    start_time = time.time()
    request_id = 0
    results = []
    
    print(f"Generating traffic to {target_url}")
    print(f"Rate: {request_rate} req/s, Duration: {duration_seconds}s")
    
    try:
        while time.time() - start_time < duration_seconds:
            request_start = time.time()
            
            # Send request
            result = await send_request(client, target_url, request_id)
            results.append(result)
            request_id += 1
            
            # Print progress every 100 requests
            if request_id % 100 == 0:
                elapsed = time.time() - start_time
                actual_rate = request_id / elapsed
                print(f"Sent {request_id} requests ({actual_rate:.2f} req/s)")
            
            # Wait to maintain request rate
            elapsed_in_interval = time.time() - request_start
            sleep_time = max(0, interval - elapsed_in_interval)
            await asyncio.sleep(sleep_time)
    
    finally:
        await client.aclose()
    
    # Print summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    avg_response_time = sum(r['response_time_ms'] for r in results) / len(results) if results else 0
    
    print(f"\nTraffic generation complete:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {successful} ({100*successful/len(results):.1f}%)")
    print(f"  Average response time: {avg_response_time:.2f} ms")
    print(f"  Duration: {total_time:.2f} s")
    print(f"  Actual rate: {len(results)/total_time:.2f} req/s")
    
    return results


def main():
    """CLI entry point for traffic generator."""
    parser = argparse.ArgumentParser(description="Generate HTTP traffic")
    parser.add_argument("--url", type=str, default="http://localhost:8080",
                        help="Target URL")
    parser.add_argument("--rate", type=float, default=10.0,
                        help="Requests per second")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration in seconds")
    parser.add_argument("--concurrent", type=int, default=100,
                        help="Maximum concurrent requests")
    
    args = parser.parse_args()
    
    asyncio.run(generate_traffic(
        target_url=args.url,
        request_rate=args.rate,
        duration_seconds=args.duration,
        max_concurrent=args.concurrent
    ))


if __name__ == "__main__":
    main()
