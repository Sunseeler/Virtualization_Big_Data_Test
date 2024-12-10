# System Virtualization Benchmark Tool

A Python-based system benchmarking tool that measures virtualization performance, system resources, and generates detailed performance metrics with visualizations. The tool simulates virtual machine workloads and provides comprehensive analysis of system behavior under load.

## Features

- Detailed system specifications gathering
- Virtual machine workload simulation
- Resource usage monitoring (CPU, RAM, Disk)
- Automated graph generation
- Comprehensive logging
- JSON result export
- Multi-process support for parallel VM simulation

## Prerequisites

- Python 3.7+
- Required Python packages:
  ```bash
  pip install psutil numpy matplotlib
  ```

## Installation

1. Clone the repository or download the script
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python system_benchmark.py
```

2. Enter a project name when prompted (or press Enter for auto-generated name)

## Output

The tool generates several types of output in `~/Documents/<project_name>/`:

1. **Benchmark Results** (`benchmark_results_YYYYMMDD_HHMMSS.json`):
   - System specifications
   - Benchmark metrics
   - Resource usage data
   - Timing information

2. **Performance Graphs** (`<test_name>_graph_YYYYMMDD_HHMMSS.png`):
   - Performance metrics visualization
   - Resource usage over time
   - VM workload statistics

3. **Log File** (`<project_name>_benchmark.log`):
   - Detailed execution logs
   - Error messages
   - Progress updates

## Benchmark Metrics

### System Specifications
- CPU cores (physical and logical)
- RAM (total and available)
- CPU frequency range
- Disk space
- Current system resource usage

### Virtualization Metrics
- Number of simulated VMs
- CPU processes per VM
- RAM allocation per VM
- Total VM RAM usage
- Peak memory usage
- CPU utilization
- Individual VM performance metrics

## Error Handling

The tool includes comprehensive error handling for:
- Resource allocation failures
- System access issues
- Data collection errors
- Graph generation problems
- File I/O operations

## Visualization

Generates two types of graphs for each benchmark:
1. Performance Metrics Bar Chart
   - Key performance indicators
   - Resource allocation metrics
   - System utilization stats

2. Resource Usage Timeline
   - CPU usage over time
   - Memory utilization
   - Per-VM resource consumption

## Best Practices

1. Close other applications before running benchmarks
2. Ensure sufficient system resources are available
3. Run multiple times for consistent results
4. Monitor system temperature during extended benchmarks

## Technical Details

### Workload Simulation
- Allocates 2GB RAM per simulated VM
- Uses numpy for CPU-intensive operations
- Runs for 60 seconds per VM
- Utilizes multiprocessing for parallel execution

### Resource Management
- Targets 85% of total system RAM
- Distributes CPU threads across VMs
- Monitors system resource usage in real-time
- Cleans up resources after completion

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]
