import time
import psutil
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import multiprocessing

def get_system_specs():
    specs = {}
    specs['cpu_cores'] = psutil.cpu_count(logical=False)
    specs['cpu_threads'] = psutil.cpu_count(logical=True)
    virtual_memory = psutil.virtual_memory()
    specs['total_ram'] = virtual_memory.total
    specs['available_ram'] = virtual_memory.available
    
    cpu_freq = psutil.cpu_freq()
    if cpu_freq:
        specs['cpu_max_freq'] = cpu_freq.max
        specs['cpu_min_freq'] = cpu_freq.min
    
    return specs

def run_for_duration(func, duration):
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < duration:
        func()
        iterations += 1
    return iterations

def vm_workload(vm_id):
    vm_ram_bytes = 2 * 1024 * 1024 * 1024  # 2 GB per VM
    mem = np.ones((vm_ram_bytes // 8,), dtype=np.int64)
    start_time = time.time()
    while time.time() - start_time < 60:  # Run for 60 seconds
        np.dot(mem[:1000], mem[:1000])
        mem[np.random.randint(0, len(mem), 1000)] += 1

def benchmark_virtualization(specs):
    print("Simulating Virtualization workload...")
    
    total_ram_bytes = specs['total_ram']
    target_ram_usage = int(total_ram_bytes * 0.85)
    vm_ram_bytes = 2 * 1024 * 1024 * 1024  # 2 GB per VM
    vm_count = max(1, target_ram_usage // vm_ram_bytes)
    
    total_cpu_threads = specs['cpu_threads']
    cpu_processes_per_vm = max(1, total_cpu_threads // vm_count)
    
    print(f"Simulating {vm_count} VMs with 2GB RAM each")
    print(f"Each VM will use {cpu_processes_per_vm} CPU processes")
    
    start_time = time.time()
    with multiprocessing.Pool(processes=min(vm_count * cpu_processes_per_vm, total_cpu_threads)) as pool:
        pool.map(vm_workload, range(vm_count * cpu_processes_per_vm))
    end_time = time.time()
    
    duration = end_time - start_time
    peak_memory = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    
    print(f"Virtualization simulation completed in {duration:.2f} seconds")
    print(f"Peak memory usage: {peak_memory:.2f}%")
    print(f"CPU usage: {cpu_usage:.2f}%")
    
    return {
        "duration": duration,
        "vm_count": vm_count,
        "cpu_processes_per_vm": cpu_processes_per_vm,
        "vm_ram_gb": vm_ram_bytes / (1024**3),
        "total_vm_ram_gb": (vm_count * vm_ram_bytes) / (1024**3),
        "peak_memory_percent": peak_memory,
        "cpu_usage_percent": cpu_usage
    }

def benchmark_big_data_analytics(specs):
    print("Simulating Big Data Analytics workload...")
    spark = SparkSession.builder.appName("BigDataBenchmark").getOrCreate()
    data_size = min(50000000, int(specs['available_ram'] / 100))
    rdd = spark.sparkContext.parallelize(range(data_size))
    
    def analytics_operation():
        rdd.map(lambda x: (x % 100, x)) \
           .groupByKey() \
           .mapValues(lambda x: sum(x) / len(x)) \
           .sortByKey() \
           .collect()
    
    start_time = time.time()
    iterations = run_for_duration(analytics_operation, 60)  # 60 seconds
    end_time = time.time()
    duration = end_time - start_time
    print(f"Big Data Analytics simulation completed in {duration:.2f} seconds")
    spark.stop()
    return {"duration": duration, "data_size": data_size, "iterations": iterations}

def create_graph(name, result):
    plt.figure(figsize=(14, 8))
    
    numeric_keys = ['duration', 'vm_count', 'cpu_processes_per_vm', 'vm_ram_gb', 'total_vm_ram_gb', 'peak_memory_percent', 'cpu_usage_percent']
    numeric_values = [result[key] for key in numeric_keys if key in result]
    
    bars = plt.bar(range(len(numeric_values)), numeric_values)
    plt.title(f'{name} Benchmark Results')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(range(len(numeric_values)), [key.replace('_', ' ').title() for key in numeric_keys if key in result], rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    return plt

def save_results(project_name, results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    documents_folder = os.path.expanduser("~/Documents")
    project_folder = os.path.join(documents_folder, project_name)
    
    try:
        os.makedirs(project_folder, exist_ok=True)
        
        for test_name, result in results.items():
            json_file = os.path.join(project_folder, f"{test_name}_results_{timestamp}.json")
            with open(json_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"{test_name} results saved to {json_file}")
            
            plt = create_graph(test_name, result)
            graph_file = os.path.join(project_folder, f"{test_name}_graph_{timestamp}.png")
            plt.savefig(graph_file)
            plt.close()
            print(f"{test_name} graph saved to {graph_file}")
        
    except Exception as e:
        print(f"Error saving files: {e}")

def main():
    project_name = input("Enter a name for this benchmark project: ")
    print(f"Starting benchmark project: {project_name}")
    
    specs = get_system_specs()
    print("\nSystem Specifications:")
    for key, value in specs.items():
        if isinstance(value, int) and value > 1000000:
            print(f"{key}: {value / (1024**3):.2f} GB")
        elif isinstance(value, float):
            print(f"{key}: {value:.2f} MHz")
        else:
            print(f"{key}: {value}")
    print("\n")

    total_start_time = time.time()

    results = {
        "Virtualization": benchmark_virtualization(specs),
        "Big_Data_Analytics": benchmark_big_data_analytics(specs)
    }

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\nTotal benchmark duration: {timedelta(seconds=int(total_duration))}")

    save_results(project_name, results)

    print("Benchmark completed.")

if __name__ == "__main__":
    main()
