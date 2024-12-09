import time
import psutil
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing
import logging
from typing import Dict, Any
import warnings
from pathlib import Path

class SystemBenchmark:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.setup_logging()
        self.specs = self.get_system_specs()
        
    def setup_logging(self) -> None:
        """Configure logging for the benchmark."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.project_name}_benchmark.log')
            ]
        )
        
    def get_system_specs(self) -> Dict[str, Any]:
        """Gather detailed system specifications."""
        specs = {}
        try:
            specs['cpu_cores'] = psutil.cpu_count(logical=False)
            specs['cpu_threads'] = psutil.cpu_count(logical=True)
            virtual_memory = psutil.virtual_memory()
            specs['total_ram'] = virtual_memory.total
            specs['available_ram'] = virtual_memory.available
            
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                specs['cpu_max_freq'] = cpu_freq.max
                specs['cpu_min_freq'] = cpu_freq.min
                
            # Additional system information
            specs['cpu_usage_percent'] = psutil.cpu_percent(interval=1)
            specs['memory_usage_percent'] = virtual_memory.percent
            specs['swap_usage'] = dict(psutil.swap_memory()._asdict())
            
            disk = psutil.disk_usage('/')
            specs['disk_total'] = disk.total
            specs['disk_free'] = disk.free
            
        except Exception as e:
            logging.error(f"Error gathering system specs: {e}")
            raise
            
        return specs

    def vm_workload(self, vm_id: int) -> Dict[str, float]:
        """Simulate VM workload with performance metrics."""
        metrics = {'vm_id': vm_id}
        start_time = time.time()
        
        try:
            vm_ram_bytes = 2 * 1024 * 1024 * 1024  # 2 GB per VM
            mem = np.ones((vm_ram_bytes // 8,), dtype=np.int64)
            
            while time.time() - start_time < 60:
                np.dot(mem[:1000], mem[:1000])
                mem[np.random.randint(0, len(mem), 1000)] += 1
                
            metrics['duration'] = time.time() - start_time
            metrics['cpu_usage'] = psutil.cpu_percent()
            metrics['memory_usage'] = psutil.virtual_memory().percent
            
        except Exception as e:
            logging.error(f"Error in VM workload {vm_id}: {e}")
            metrics['error'] = str(e)
            
        return metrics

    def benchmark_virtualization(self) -> Dict[str, Any]:
        """Run virtualization benchmark with enhanced error handling and metrics."""
        logging.info("Starting virtualization benchmark...")
        results = {}
        
        try:
            target_ram_usage = int(self.specs['total_ram'] * 0.85)
            vm_ram_bytes = 2 * 1024 * 1024 * 1024
            vm_count = max(1, target_ram_usage // vm_ram_bytes)
            
            cpu_processes_per_vm = max(1, self.specs['cpu_threads'] // vm_count)
            
            start_time = time.time()
            with multiprocessing.Pool(processes=min(vm_count * cpu_processes_per_vm, 
                                                  self.specs['cpu_threads'])) as pool:
                vm_results = pool.map(self.vm_workload, range(vm_count))
                
            results = {
                "duration": time.time() - start_time,
                "vm_count": vm_count,
                "cpu_processes_per_vm": cpu_processes_per_vm,
                "vm_ram_gb": vm_ram_bytes / (1024**3),
                "total_vm_ram_gb": (vm_count * vm_ram_bytes) / (1024**3),
                "peak_memory_percent": psutil.virtual_memory().percent,
                "cpu_usage_percent": psutil.cpu_percent(),
                "vm_metrics": vm_results
            }
            
        except Exception as e:
            logging.error(f"Virtualization benchmark failed: {e}")
            results['error'] = str(e)
            
        return results

    def create_graphs(self, results: Dict[str, Dict]) -> None:
        """Create enhanced visualizations of benchmark results."""
        for test_name, result in results.items():
            if 'error' in result:
                continue
                
            plt.figure(figsize=(15, 10))
            
            numeric_keys = [k for k, v in result.items() 
                          if isinstance(v, (int, float)) and k != 'error' 
                          and k != 'vm_metrics']
            numeric_values = [result[k] for k in numeric_keys]
            
            # Create main performance graph
            plt.subplot(2, 1, 1)
            bars = plt.bar(range(len(numeric_values)), numeric_values)
            plt.title(f'{test_name} Performance Metrics')
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.xticks(range(len(numeric_values)), 
                      [k.replace('_', ' ').title() for k in numeric_keys],
                      rotation=45, ha='right')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            # Add system resource usage over time if available
            if 'vm_metrics' in result:
                plt.subplot(2, 1, 2)
                vm_cpu_usage = [vm['cpu_usage'] for vm in result['vm_metrics'] 
                              if 'cpu_usage' in vm]
                vm_mem_usage = [vm['memory_usage'] for vm in result['vm_metrics'] 
                              if 'memory_usage' in vm]
                
                if vm_cpu_usage and vm_mem_usage:
                    plt.plot(vm_cpu_usage, label='CPU Usage %')
                    plt.plot(vm_mem_usage, label='Memory Usage %')
                    plt.title('Resource Usage Over Time')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Usage %')
                    plt.legend()
            
            plt.tight_layout()
            
            # Save the graph
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(os.path.expanduser("~/Documents")) / self.project_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            graph_path = output_dir / f"{test_name}_graph_{timestamp}.png"
            plt.savefig(graph_path)
            plt.close()
            logging.info(f"Saved graph to {graph_path}")

    def save_results(self, results: Dict[str, Dict]) -> None:
        """Save benchmark results with enhanced metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(os.path.expanduser("~/Documents")) / self.project_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata to results
        full_results = {
            "metadata": {
                "timestamp": timestamp,
                "project_name": self.project_name,
                "system_specs": self.specs
            },
            "benchmarks": results
        }
        
        # Save results as JSON
        results_path = output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(full_results, f, indent=4)
        logging.info(f"Saved results to {results_path}")
        
        # Create and save graphs
        self.create_graphs(results)

    def run(self) -> None:
        """Run the virtualization benchmark with progress tracking."""
        logging.info(f"Starting benchmark project: {self.project_name}")
        
        # Log system specifications
        logging.info("System Specifications:")
        for key, value in self.specs.items():
            if isinstance(value, (int, float)):
                if value > 1000000:
                    logging.info(f"{key}: {value / (1024**3):.2f} GB")
                else:
                    logging.info(f"{key}: {value}")
                    
        total_start_time = time.time()
        
        results = {
            "Virtualization": self.benchmark_virtualization()
        }
        
        total_duration = time.time() - total_start_time
        logging.info(f"Total benchmark duration: {timedelta(seconds=int(total_duration))}")
        
        self.save_results(results)
        logging.info("Benchmark completed successfully.")

def main():
    try:
        project_name = input("Enter a name for this benchmark project: ").strip()
        if not project_name:
            project_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        benchmark = SystemBenchmark(project_name)
        benchmark.run()
        
    except KeyboardInterrupt:
        logging.warning("Benchmark interrupted by user")
        
    except Exception as e:
        logging.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
