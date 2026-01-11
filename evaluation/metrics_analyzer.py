"""
Metrics analyzer for performance evaluation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class MetricsAnalyzer:
    """Analyzes metrics from load balancing experiments."""
    
    def __init__(self, request_logs_df: pd.DataFrame):
        """
        Initialize analyzer with request logs.
        
        Args:
            request_logs_df: DataFrame with request logs
        """
        self.df = request_logs_df
    
    def calculate_latency_stats(self, strategy: str = None) -> Dict[str, float]:
        """
        Calculate latency statistics.
        
        Args:
            strategy: Filter by routing strategy (None for all)
        
        Returns:
            Dictionary with latency statistics
        """
        df = self.df if strategy is None else self.df[self.df['routing_strategy'] == strategy]
        
        if df.empty:
            return {
                'mean': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        response_times = df['response_time_ms'].values
        
        return {
            'mean': np.mean(response_times),
            'p50': np.percentile(response_times, 50),
            'p95': np.percentile(response_times, 95),
            'p99': np.percentile(response_times, 99),
            'std': np.std(response_times),
            'min': np.min(response_times),
            'max': np.max(response_times)
        }
    
    def calculate_throughput(self, strategy: str = None, window_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Calculate throughput over time.
        
        Args:
            strategy: Filter by routing strategy
            window_size: Number of requests per window
        
        Returns:
            List of throughput measurements per window
        """
        df = self.df if strategy is None else self.df[self.df['routing_strategy'] == strategy]
        
        if df.empty:
            return []
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp')
        
        throughput_data = []
        for i in range(0, len(df_sorted), window_size):
            window_df = df_sorted.iloc[i:i+window_size]
            if len(window_df) == 0:
                continue
            
            start_time = window_df['timestamp'].min()
            end_time = window_df['timestamp'].max()
            duration = end_time - start_time
            
            if duration > 0:
                throughput = len(window_df) / duration  # requests per second
                avg_latency = window_df['response_time_ms'].mean()
                success_rate = window_df['success'].sum() / len(window_df)
                
                throughput_data.append({
                    'window_start': start_time,
                    'window_end': end_time,
                    'window_index': i // window_size,
                    'throughput': throughput,
                    'avg_latency': avg_latency,
                    'success_rate': success_rate,
                    'request_count': len(window_df)
                })
        
        return throughput_data
    
    def calculate_success_rate(self, strategy: str = None) -> float:
        """Calculate request success rate."""
        df = self.df if strategy is None else self.df[self.df['routing_strategy'] == strategy]
        
        if df.empty:
            return 0.0
        
        return df['success'].sum() / len(df)
    
    def calculate_server_distribution(self, strategy: str = None) -> Dict[str, int]:
        """
        Calculate request distribution across servers.
        
        Args:
            strategy: Filter by routing strategy
        
        Returns:
            Dictionary mapping server_id to request count
        """
        df = self.df if strategy is None else self.df[self.df['routing_strategy'] == strategy]
        
        if df.empty:
            return {}
        
        return df['selected_server'].value_counts().to_dict()
    
    def calculate_learning_curve(self, strategy: str = "rl_agent", window_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Calculate learning curve (average latency over time for RL agent).
        
        Args:
            strategy: Routing strategy (should be "rl_agent")
            window_size: Number of requests per window
        
        Returns:
            List of learning curve points
        """
        df = self.df[self.df['routing_strategy'] == strategy].copy()
        
        if df.empty:
            return []
        
        df_sorted = df.sort_values('timestamp')
        
        learning_curve = []
        for i in range(0, len(df_sorted), window_size):
            window_df = df_sorted.iloc[i:i+window_size]
            if len(window_df) == 0:
                continue
            
            avg_latency = window_df['response_time_ms'].mean()
            learning_curve.append({
                'window_index': i // window_size,
                'request_count': i + len(window_df),
                'avg_latency': avg_latency,
                'timestamp': window_df['timestamp'].mean()
            })
        
        return learning_curve
    
    def compare_strategies(self) -> Dict[str, Any]:
        """
        Compare Round Robin vs RL agent performance.
        
        Returns:
            Dictionary with comparison metrics
        """
        rr_stats = self.calculate_latency_stats("round_robin")
        rl_stats = self.calculate_latency_stats("rl_agent")
        
        rr_throughput = self.calculate_throughput("round_robin")
        rl_throughput = self.calculate_throughput("rl_agent")
        
        rr_success = self.calculate_success_rate("round_robin")
        rl_success = self.calculate_success_rate("rl_agent")
        
        # Calculate coefficient of variation for stability
        rr_df = self.df[self.df['routing_strategy'] == "round_robin"]
        rl_df = self.df[self.df['routing_strategy'] == "rl_agent"]
        
        rr_cv = (rr_stats['std'] / rr_stats['mean']) if rr_stats['mean'] > 0 else 0
        rl_cv = (rl_stats['std'] / rl_stats['mean']) if rl_stats['mean'] > 0 else 0
        
        return {
            'round_robin': {
                'latency_stats': rr_stats,
                'throughput_data': rr_throughput,
                'success_rate': rr_success,
                'coefficient_of_variation': rr_cv,
                'server_distribution': self.calculate_server_distribution("round_robin")
            },
            'rl_agent': {
                'latency_stats': rl_stats,
                'throughput_data': rl_throughput,
                'success_rate': rl_success,
                'coefficient_of_variation': rl_cv,
                'server_distribution': self.calculate_server_distribution("rl_agent"),
                'learning_curve': self.calculate_learning_curve("rl_agent")
            },
            'improvement': {
                'latency_reduction_percent': ((rr_stats['mean'] - rl_stats['mean']) / rr_stats['mean'] * 100) if rr_stats['mean'] > 0 else 0,
                'stability_improvement': rr_cv - rl_cv,  # Positive = RL more stable
                'throughput_improvement_percent': 0  # Will be calculated from throughput data
            }
        }
