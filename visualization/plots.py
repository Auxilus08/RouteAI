"""
Visualization scripts for load balancing performance metrics.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_latency_over_time(
    request_logs_df: pd.DataFrame,
    output_path: Path,
    window_size: int = 100
):
    """
    Plot latency over time for both strategies.
    
    Args:
        request_logs_df: DataFrame with request logs
        output_path: Path to save the plot
        window_size: Window size for moving average
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for strategy in ["round_robin", "rl_agent"]:
        df_strategy = request_logs_df[request_logs_df['routing_strategy'] == strategy].copy()
        if df_strategy.empty:
            continue
        
        df_strategy = df_strategy.sort_values('timestamp')
        
        # Calculate moving average
        df_strategy['latency_ma'] = df_strategy['response_time_ms'].rolling(window=window_size, min_periods=1).mean()
        
        # Plot
        ax.plot(
            df_strategy['timestamp'],
            df_strategy['latency_ma'],
            label=f"{strategy.replace('_', ' ').title()}",
            alpha=0.8,
            linewidth=2
        )
    
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Average Response Time (ms)", fontsize=12)
    ax.set_title("Response Time Over Time (Moving Average)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latency over time plot to {output_path}")


def plot_throughput_comparison(
    throughput_data_rr: List[Dict[str, Any]],
    throughput_data_rl: List[Dict[str, Any]],
    output_path: Path
):
    """
    Plot throughput comparison between strategies.
    
    Args:
        throughput_data_rr: Throughput data for Round Robin
        throughput_data_rl: Throughput data for RL agent
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Throughput plot
    if throughput_data_rr:
        df_rr = pd.DataFrame(throughput_data_rr)
        ax1.plot(df_rr['window_index'], df_rr['throughput'], label="Round Robin", marker='o', markersize=4)
    
    if throughput_data_rl:
        df_rl = pd.DataFrame(throughput_data_rl)
        ax1.plot(df_rl['window_index'], df_rl['throughput'], label="RL Agent", marker='s', markersize=4)
    
    ax1.set_xlabel("Time Window", fontsize=12)
    ax1.set_ylabel("Throughput (req/s)", fontsize=12)
    ax1.set_title("Throughput Comparison", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Average latency in throughput windows
    if throughput_data_rr:
        ax2.plot(df_rr['window_index'], df_rr['avg_latency'], label="Round Robin", marker='o', markersize=4)
    
    if throughput_data_rl:
        ax2.plot(df_rl['window_index'], df_rl['avg_latency'], label="RL Agent", marker='s', markersize=4)
    
    ax2.set_xlabel("Time Window", fontsize=12)
    ax2.set_ylabel("Average Latency (ms)", fontsize=12)
    ax2.set_title("Average Latency per Window", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved throughput comparison plot to {output_path}")


def plot_server_utilization(
    server_distribution_rr: Dict[str, int],
    server_distribution_rl: Dict[str, int],
    output_path: Path
):
    """
    Plot request distribution across servers.
    
    Args:
        server_distribution_rr: Server distribution for Round Robin
        server_distribution_rl: Server distribution for RL agent
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Round Robin distribution
    if server_distribution_rr:
        servers = list(server_distribution_rr.keys())
        counts = list(server_distribution_rr.values())
        ax1.bar(servers, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel("Server", fontsize=12)
        ax1.set_ylabel("Request Count", fontsize=12)
        ax1.set_title("Round Robin: Request Distribution", fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # RL Agent distribution
    if server_distribution_rl:
        servers = list(server_distribution_rl.keys())
        counts = list(server_distribution_rl.values())
        ax2.bar(servers, counts, color='coral', alpha=0.7)
        ax2.set_xlabel("Server", fontsize=12)
        ax2.set_ylabel("Request Count", fontsize=12)
        ax2.set_title("RL Agent: Request Distribution", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved server utilization plot to {output_path}")


def plot_learning_curve(
    learning_curve: List[Dict[str, Any]],
    output_path: Path
):
    """
    Plot RL agent learning curve.
    
    Args:
        learning_curve: Learning curve data
        output_path: Path to save the plot
    """
    if not learning_curve:
        print("No learning curve data available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = pd.DataFrame(learning_curve)
    
    ax.plot(df['request_count'], df['avg_latency'], marker='o', markersize=5, linewidth=2, color='green')
    ax.set_xlabel("Number of Requests", fontsize=12)
    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_title("RL Agent Learning Curve", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curve plot to {output_path}")


def plot_latency_distribution(
    request_logs_df: pd.DataFrame,
    output_path: Path
):
    """
    Plot latency distribution comparison.
    
    Args:
        request_logs_df: DataFrame with request logs
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy in ["round_robin", "rl_agent"]:
        df_strategy = request_logs_df[request_logs_df['routing_strategy'] == strategy]
        if not df_strategy.empty:
            ax.hist(
                df_strategy['response_time_ms'],
                bins=50,
                alpha=0.6,
                label=f"{strategy.replace('_', ' ').title()}",
                density=True
            )
    
    ax.set_xlabel("Response Time (ms)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Response Time Distribution", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latency distribution plot to {output_path}")


def plot_comparison_summary(
    comparison: Dict[str, Any],
    output_path: Path
):
    """
    Create a summary comparison plot with multiple metrics.
    
    Args:
        comparison: Comparison dictionary from MetricsAnalyzer
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    rr = comparison['round_robin']
    rl = comparison['rl_agent']
    
    # Latency comparison
    metrics = ['mean', 'p50', 'p95', 'p99']
    rr_values = [rr['latency_stats'][m] for m in metrics]
    rl_values = [rl['latency_stats'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, rr_values, width, label='Round Robin', alpha=0.8)
    axes[0, 0].bar(x + width/2, rl_values, width, label='RL Agent', alpha=0.8)
    axes[0, 0].set_xlabel('Metric', fontsize=11)
    axes[0, 0].set_ylabel('Latency (ms)', fontsize=11)
    axes[0, 0].set_title('Latency Statistics Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Success rate
    strategies = ['Round Robin', 'RL Agent']
    success_rates = [rr['success_rate'] * 100, rl['success_rate'] * 100]
    axes[0, 1].bar(strategies, success_rates, color=['steelblue', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('Success Rate (%)', fontsize=11)
    axes[0, 1].set_title('Success Rate Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Coefficient of variation (stability)
    cv_values = [rr['coefficient_of_variation'], rl['coefficient_of_variation']]
    axes[1, 0].bar(strategies, cv_values, color=['steelblue', 'coral'], alpha=0.8)
    axes[1, 0].set_ylabel('Coefficient of Variation', fontsize=11)
    axes[1, 0].set_title('Stability Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Improvement metrics
    improvement = comparison['improvement']
    improvements = [
        improvement['latency_reduction_percent'],
        improvement['stability_improvement'] * 100  # Scale for visibility
    ]
    improvement_labels = ['Latency\nReduction (%)', 'Stability\nImprovement (scaled)']
    axes[1, 1].bar(improvement_labels, improvements, color='green', alpha=0.8)
    axes[1, 1].set_ylabel('Improvement', fontsize=11)
    axes[1, 1].set_title('RL Agent Improvements', fontsize=12, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison summary plot to {output_path}")


def create_all_visualizations(
    request_logs_df: pd.DataFrame,
    comparison: Dict[str, Any],
    output_dir: Path
):
    """
    Create all visualization plots.
    
    Args:
        request_logs_df: DataFrame with request logs
        comparison: Comparison dictionary from MetricsAnalyzer
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Latency over time
    plot_latency_over_time(
        request_logs_df,
        output_dir / "latency_over_time.png"
    )
    
    # Throughput comparison
    plot_throughput_comparison(
        comparison['round_robin']['throughput_data'],
        comparison['rl_agent']['throughput_data'],
        output_dir / "throughput_comparison.png"
    )
    
    # Server utilization
    plot_server_utilization(
        comparison['round_robin']['server_distribution'],
        comparison['rl_agent']['server_distribution'],
        output_dir / "server_utilization.png"
    )
    
    # Learning curve
    if comparison['rl_agent']['learning_curve']:
        plot_learning_curve(
            comparison['rl_agent']['learning_curve'],
            output_dir / "learning_curve.png"
        )
    
    # Latency distribution
    plot_latency_distribution(
        request_logs_df,
        output_dir / "latency_distribution.png"
    )
    
    # Summary comparison
    plot_comparison_summary(
        comparison,
        output_dir / "comparison_summary.png"
    )
    
    print(f"\nAll visualizations saved to {output_dir}")
