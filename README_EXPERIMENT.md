# Running Experiments and Updating Reports

## Quick Start

1. **Run the experiment**:
   ```bash
   python run_experiment.py
   ```
   
   This will:
   - Start 3 backend servers
   - Run Round Robin baseline experiment (1000 requests)
   - Run RL Agent experiment (1000 requests)
   - Generate comparison results and visualizations
   - Save everything to `reports/` directory

2. **Update the experimental report**:
   ```bash
   python update_experimental_report.py
   ```
   
   This will automatically update `docs/experimental_report.md` with real data from the experiment.

## Experiment Configuration

You can modify experiment parameters in `run_experiment.py`:

```python
await runner.run_full_experiment(
    num_requests=1000,      # Number of requests per strategy
    request_rate=10.0,      # Requests per second
    output_dir=None         # Output directory (default: reports/)
)
```

## Results Files

After running an experiment, the `reports/` directory will contain:
- `comparison_results.json` - Complete comparison data (used by update script)
- `request_logs.csv` - Request-level performance data
- `summary_report.md` - Auto-generated summary
- `*.png` - Visualization plots

## Updating the Report

The `update_experimental_report.py` script will:
- Read `reports/comparison_results.json`
- Extract all metrics and statistics
- Update `docs/experimental_report.md` with real values
- Generate analysis and conclusions based on actual results
