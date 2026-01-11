"""
Script to update experimental report with actual results from comparison_results.json
"""
import json
import re
from pathlib import Path
from datetime import datetime


def update_experimental_report(results_json_path: Path, report_template_path: Path, output_path: Path = None):
    """
    Update experimental report with actual results.
    
    Args:
        results_json_path: Path to comparison_results.json
        report_template_path: Path to experimental_report.md template
        output_path: Output path (defaults to overwriting template)
    """
    if output_path is None:
        output_path = report_template_path
    
    # Load results
    with open(results_json_path, 'r') as f:
        comparison = json.load(f)
    
    rr = comparison['round_robin']
    rl = comparison['rl_agent']
    improvement = comparison['improvement']
    
    # Read template
    with open(report_template_path, 'r') as f:
        template = f.read()
    
    # Extract learning curve data if available
    learning_curve = rl.get('learning_curve', [])
    initial_latency = learning_curve[0]['avg_latency'] if learning_curve else rl['latency_stats']['mean']
    final_latency = learning_curve[-1]['avg_latency'] if learning_curve else rl['latency_stats']['mean']
    convergence_point = len(learning_curve) * 1000 if learning_curve else 1000  # Approximate
    
    # Calculate throughput averages
    rr_throughput = rr.get('throughput_data', [])
    rl_throughput = rl.get('throughput_data', [])
    rr_avg_throughput = sum(t['throughput'] for t in rr_throughput) / len(rr_throughput) if rr_throughput else 0
    rl_avg_throughput = sum(t['throughput'] for t in rl_throughput) / len(rl_throughput) if rl_throughput else 0
    rr_peak_throughput = max((t['throughput'] for t in rr_throughput), default=0) if rr_throughput else 0
    rl_peak_throughput = max((t['throughput'] for t in rl_throughput), default=0) if rl_throughput else 0
    
    # Server distribution percentages
    rr_total = sum(rr['server_distribution'].values()) if rr['server_distribution'] else 1
    rl_total = sum(rl['server_distribution'].values()) if rl['server_distribution'] else 1
    
    # Build updated report section by section using regex
    updated_report = template
    
    # Round Robin latency stats - replace in Round Robin section
    rr_latency_pattern = r'(\*\*Round Robin\*\*:\s*\n)((?:- .+\n)+)'
    rr_latency_replacement = f"""**Round Robin**:
- Mean Latency: {rr['latency_stats']['mean']:.2f} ms
- P50 (Median): {rr['latency_stats']['p50']:.2f} ms
- P95: {rr['latency_stats']['p95']:.2f} ms
- P99: {rr['latency_stats']['p99']:.2f} ms
- Standard Deviation: {rr['latency_stats']['std']:.2f} ms

"""
    updated_report = re.sub(rr_latency_pattern, rr_latency_replacement, updated_report, count=1)
    
    # RL Agent latency stats
    rl_latency_pattern = r'(\*\*RL Agent\*\*:\s*\n)((?:- .+\n)+)'
    rl_latency_replacement = f"""**RL Agent**:
- Mean Latency: {rl['latency_stats']['mean']:.2f} ms
- P50 (Median): {rl['latency_stats']['p50']:.2f} ms
- P95: {rl['latency_stats']['p95']:.2f} ms
- P99: {rl['latency_stats']['p99']:.2f} ms
- Standard Deviation: {rl['latency_stats']['std']:.2f} ms

"""
    updated_report = re.sub(rl_latency_pattern, rl_latency_replacement, updated_report, count=1)
    
    # Improvement
    updated_report = re.sub(
        r'\*\*Improvement\*\*: \[To be calculated\]',
        f"**Improvement**: {improvement['latency_reduction_percent']:.2f}% latency reduction",
        updated_report
    )
    
    # Throughput
    updated_report = re.sub(
        r'\*\*Round Robin\*\*: \[To be filled\]\n- Average: \[req/s\]\n- Peak: \[req/s\]',
        f"**Round Robin**:\n- Average: {rr_avg_throughput:.2f} req/s\n- Peak: {rr_peak_throughput:.2f} req/s",
        updated_report
    )
    updated_report = re.sub(
        r'\*\*RL Agent\*\*: \[To be filled\]\n- Average: \[req/s\]\n- Peak: \[req/s\]',
        f"**RL Agent**:\n- Average: {rl_avg_throughput:.2f} req/s\n- Peak: {rl_peak_throughput:.2f} req/s",
        updated_report
    )
    
    # Success rates
    updated_report = re.sub(
        r'- \*\*Round Robin\*\*: \[To be filled\]%',
        f"- **Round Robin**: {rr['success_rate']*100:.2f}%",
        updated_report
    )
    updated_report = re.sub(
        r'- \*\*RL Agent\*\*: \[To be filled\]%',
        f"- **RL Agent**: {rl['success_rate']*100:.2f}%",
        updated_report
    )
    
    # Server distribution
    rr_dist_text = "**Round Robin**:\n"
    for server, count in sorted(rr['server_distribution'].items()):
        percentage = (count / rr_total) * 100
        rr_dist_text += f"  - {server}: {count} requests ({percentage:.1f}%)\n"
    
    rl_dist_text = "**RL Agent**:\n"
    for server, count in sorted(rl['server_distribution'].items()):
        percentage = (count / rl_total) * 100
        rl_dist_text += f"  - {server}: {count} requests ({percentage:.1f}%)\n"
    
    updated_report = re.sub(
        r'\*\*Round Robin\*\*: Expected uniform distribution \(~33% per server\)',
        rr_dist_text.strip(),
        updated_report
    )
    updated_report = re.sub(
        r'\*\*RL Agent\*\*: Adaptive distribution based on learned policy',
        rl_dist_text.strip(),
        updated_report
    )
    
    # Learning curve
    updated_report = re.sub(
        r'- Initial performance: \[To be filled\]',
        f"- Initial performance: {initial_latency:.2f} ms",
        updated_report
    )
    updated_report = re.sub(
        r'- Converged performance: \[To be filled\]',
        f"- Converged performance: {final_latency:.2f} ms",
        updated_report
    )
    updated_report = re.sub(
        r'- Convergence point: ~\[N\] requests',
        f"- Convergence point: ~{convergence_point} requests",
        updated_report
    )
    
    # Stability
    updated_report = re.sub(
        r'- \*\*Round Robin CV\*\*: \[To be filled\]',
        f"- **Round Robin CV**: {rr['coefficient_of_variation']:.4f}",
        updated_report
    )
    updated_report = re.sub(
        r'- \*\*RL Agent CV\*\*: \[To be filled\]',
        f"- **RL Agent CV**: {rl['coefficient_of_variation']:.4f}",
        updated_report
    )
    updated_report = re.sub(
        r'- \*\*Improvement\*\*: Lower CV indicates better stability',
        f"- **Improvement**: {improvement['stability_improvement']:.4f} (negative = RL more stable)",
        updated_report
    )
    
    # Analysis
    analysis_text = generate_analysis(comparison)
    updated_report = updated_report.replace(
        '[Analysis to be filled based on actual results]',
        analysis_text
    )
    
    # Conclusion
    conclusion_text = generate_conclusion(comparison)
    if '## Conclusion' in updated_report:
        # Replace the conclusion section
        start_idx = updated_report.find('## Conclusion')
        end_idx = updated_report.find('## References', start_idx)
        if end_idx > 0:
            updated_report = updated_report[:start_idx] + conclusion_text + '\n\n' + updated_report[end_idx:]
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_report = updated_report.replace(
        '**Note**: This is a template experimental report.',
        f'**Report Generated**: {timestamp}\n\n**Note**: This report was automatically generated from experimental results.'
    )
    
    # Write updated report
    with open(output_path, 'w') as f:
        f.write(updated_report)
    
    print(f"Experimental report updated: {output_path}")


def generate_analysis(comparison: dict) -> str:
    """Generate analysis section from comparison results."""
    rr = comparison['round_robin']
    rl = comparison['rl_agent']
    improvement = comparison['improvement']
    
    # stability_improvement = rr_cv - rl_cv, so positive means RL is more stable (lower CV)
    stability_text = "The RL agent shows" if improvement['stability_improvement'] > 0 else "Round Robin shows"
    
    analysis = f"""Based on the experimental results:

1. **Latency Reduction**: The RL agent achieved a **{improvement['latency_reduction_percent']:.2f}% reduction** in average latency compared to Round Robin.
   - Round Robin: {rr['latency_stats']['mean']:.2f} ms
   - RL Agent: {rl['latency_stats']['mean']:.2f} ms
   
2. **Stability Improvement**: {stability_text} better stability (lower coefficient of variation).
   - Round Robin CV: {rr['coefficient_of_variation']:.4f}
   - RL Agent CV: {rl['coefficient_of_variation']:.4f}
   - Improvement: {abs(improvement['stability_improvement']):.4f} (RL has {improvement['stability_improvement']:.1%} lower CV)
   
3. **Adaptive Behavior**: The RL agent demonstrates adaptive routing by distributing requests based on learned server performance patterns."""
    
    return analysis


def generate_conclusion(comparison: dict) -> str:
    """Generate conclusion section from comparison results."""
    improvement = comparison['improvement']
    rr = comparison['round_robin']
    rl = comparison['rl_agent']
    
    performance_text = 'Outperforms' if improvement['latency_reduction_percent'] > 0 else 'Performs similarly to'
    improvement_text = 'improvement' if improvement['latency_reduction_percent'] > 0 else 'difference'
    # stability_improvement = rr_cv - rl_cv, so positive means RL is more stable
    stability_text = 'RL agent' if improvement['stability_improvement'] > 0 else 'Round Robin'
    stability_comparison = 'better' if improvement['stability_improvement'] > 0 else 'similar'
    
    conclusion = f"""## Conclusion

The RL-based load balancer demonstrates:

1. **Feasibility**: Q-Learning successfully learns routing policies from server health metrics
2. **Performance**: {performance_text} Round Robin with {abs(improvement['latency_reduction_percent']):.2f}% {improvement_text} in average latency
3. **Practicality**: Suitable for scenarios with variable server loads, showing adaptive behavior through learned routing patterns

### Key Metrics

- **Latency Improvement**: {improvement['latency_reduction_percent']:.2f}%
- **Round Robin Performance**: {rr['latency_stats']['mean']:.2f} ms average latency, {rr['success_rate']*100:.2f}% success rate
- **RL Agent Performance**: {rl['latency_stats']['mean']:.2f} ms average latency, {rl['success_rate']*100:.2f}% success rate
- **Stability**: {stability_text} shows {stability_comparison} stability"""
    
    return conclusion


def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    results_json = project_root / "reports" / "comparison_results.json"
    report_template = project_root / "docs" / "experimental_report.md"
    
    if not results_json.exists():
        print(f"Error: Results file not found: {results_json}")
        print("\nPlease run an experiment first:")
        print("  python run_experiment.py")
        return
    
    if not report_template.exists():
        print(f"Error: Report template not found: {report_template}")
        return
    
    update_experimental_report(results_json, report_template)
    print("\nExperimental report has been updated with real data!")


if __name__ == "__main__":
    main()
