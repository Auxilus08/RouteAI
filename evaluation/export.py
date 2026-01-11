"""
Export evaluation results to files.
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any
import pandas as pd


class ResultsExporter:
    """Exports evaluation results to various formats."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_json(self, data: Dict[str, Any], filename: str = "results.json"):
        """Export data to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Exported JSON to {filepath}")
    
    def export_csv(self, df: pd.DataFrame, filename: str):
        """Export DataFrame to CSV file."""
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Exported CSV to {filepath}")
    
    def export_summary_report(self, comparison: Dict[str, Any], filename: str = "summary_report.md"):
        """
        Export a markdown summary report.
        
        Args:
            comparison: Comparison dictionary from MetricsAnalyzer.compare_strategies()
            filename: Output filename
        """
        filepath = self.output_dir / filename
        
        rr = comparison['round_robin']
        rl = comparison['rl_agent']
        improvement = comparison['improvement']
        
        report = f"""# Load Balancing Performance Comparison Report

## Executive Summary

This report compares Round Robin and Reinforcement Learning (RL) based load balancing strategies.

## Latency Statistics

### Round Robin
- **Mean Latency**: {rr['latency_stats']['mean']:.2f} ms
- **P50 (Median)**: {rr['latency_stats']['p50']:.2f} ms
- **P95**: {rr['latency_stats']['p95']:.2f} ms
- **P99**: {rr['latency_stats']['p99']:.2f} ms
- **Standard Deviation**: {rr['latency_stats']['std']:.2f} ms

### RL Agent
- **Mean Latency**: {rl['latency_stats']['mean']:.2f} ms
- **P50 (Median)**: {rl['latency_stats']['p50']:.2f} ms
- **P95**: {rl['latency_stats']['p95']:.2f} ms
- **P99**: {rl['latency_stats']['p99']:.2f} ms
- **Standard Deviation**: {rl['latency_stats']['std']:.2f} ms

### Improvement
- **Latency Reduction**: {improvement['latency_reduction_percent']:.2f}%
- **Stability Improvement (CV reduction)**: {improvement['stability_improvement']:.4f}

## Success Rates

- **Round Robin**: {rr['success_rate']*100:.2f}%
- **RL Agent**: {rl['success_rate']*100:.2f}%

## Request Distribution

### Round Robin Server Distribution
"""
        
        for server, count in sorted(rr['server_distribution'].items()):
            report += f"- **{server}**: {count} requests\n"
        
        report += "\n### RL Agent Server Distribution\n"
        for server, count in sorted(rl['server_distribution'].items()):
            report += f"- **{server}**: {count} requests\n"
        
        report += f"""
## System Stability

- **Round Robin Coefficient of Variation**: {rr['coefficient_of_variation']:.4f}
- **RL Agent Coefficient of Variation**: {rl['coefficient_of_variation']:.4f}

## Learning Behavior

The RL agent showed learning behavior over time. See the learning curve visualization for details.

## Conclusion

"""
        
        if improvement['latency_reduction_percent'] > 0:
            report += f"The RL agent achieved a **{improvement['latency_reduction_percent']:.2f}% reduction** in average latency compared to Round Robin.\n\n"
        else:
            report += "The RL agent did not achieve latency reduction in this experiment.\n\n"
        
        if improvement['stability_improvement'] > 0:
            report += f"The RL agent demonstrated **better stability** with a lower coefficient of variation.\n\n"
        else:
            report += "The RL agent did not demonstrate improved stability in this experiment.\n\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Exported summary report to {filepath}")
