"""This file has the context manager for logging time metrics per batch using the cuda events library"""
import time
import torch
import wandb
import io
import numpy as np
import os
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import Union
from datetime import datetime
from transformers import MixtralConfig

# define a constant for plotting dir
PLOTTING_DIR = "./profiling_output"


def if_enabled(func):
    def wrapper(self, *args, **kwargs):
        if not self.disable and self.profile_complete:
            return func(self, *args, **kwargs)
 
    return wrapper


class TaskType(Enum):
   MERGE_TOTAL_GPU = auto()
   MERGE_TOTAL_CPU = auto()
   MERGE_MEMORY = auto()
   MERGE_TOTAL = auto()
   MERGE_PLANNER = auto()
   MERGE_STEP = auto()



class TaskLoggingContextManagerGPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        MetricStore.get_instance().add_metric_gpu(self.task_type, self.start_event, self.end_event)
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        

class TaskLoggingContextManagerCPU:
    def __init__(self, task_type: TaskType) -> None:
        self.task_type = task_type
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.perf_counter()
        elapsed_time = self.end_time - self.start_time
        MetricStore.get_instance().add_metric_cpu(self.task_type, elapsed_time)
        print(f"{self.task_type} took {elapsed_time} s")
        if exc_type is not None:
            print(f"An error occurred: {exc_value}")
        #print(f"{self.task_type} took {elapsed_time} s")

# Define context manager using torch.cuda.memory_stats for obtaining peak memory usage
class MemoryLoggingContextManager:
    def __init__(self) -> None:
        self.start_memory = None
        self.end_memory = None

    def __enter__(self):
        self.start_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        memory_usage = self.end_memory - self.start_memory
        # let's convert the memory usage to GB
        memory_usage = memory_usage / 1024**3
        print(f"Peak memory usage: {memory_usage} GB")
        torch.cuda.reset_peak_memory_stats()
        

"""This class is a singleton. It stores the metrics for each task type."""
class MetricStore:
    _instance = None
    def __init__(self) -> None:
        super(MetricStore, self).__init__()
        self.metrics = {}
        self.raw_metrics = {}
        self.disable = True
        self.memory_usage = 0
        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Define the full path including the base directory and the new subdirectory with date and time
        self.full_path = os.path.join(PLOTTING_DIR, f"plots_{current_datetime}")
        # Use os.makedirs to create the directory, including all necessary parent directories
        os.makedirs(self.full_path, exist_ok=True)

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @if_enabled
    def write_to_file(self, metrics: dict) -> None:
        with open("metrics.txt", "a") as f:
            f.write(str(metrics))
            f.write("\n")
        self.metrics = {}

    @if_enabled
    def after_merge(self):
        self.calculate_elapsed_times()


    @if_enabled
    def add_metric_gpu(self, task_type: TaskType, start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> None:
        if task_type in self.metrics:
            self.raw_metrics[task_type].append((start_event, end_event))
        else:
            self.raw_metrics[task_type] = [(start_event, end_event)]
       
    @if_enabled
    def add_metric_cpu(self, task_type: TaskType, elapsed_time: float) -> None:
        if task_type not in self.metrics:
            self.metrics[task_type] = elapsed_time
        self.metrics[task_type].put(elapsed_time)
    
    @if_enabled
    def get_metrics(self) -> dict:
        return self.metrics
    
    @if_enabled
    def set_memory_usage(self, memory_usage: int) -> None:
        self.memory_usage = memory_usage
    

    ### replace this with cpu task value
    @if_enabled
    def calculate_elapsed_times(self):
        torch.cuda.synchronize()  # Ensure all prior operations on the default stream are complete before calculations

        for task_type, measurements in list(self.raw_metrics.items()):
            if task_type == TaskType.MERGE_TOTAL: ## if using CPU
                continue  # Skip processing for specific task types if needed
            
            for measurement in measurements:
                start_event, end_event = measurement  # Unpack the tuple
                try:
                    # Calculate elapsed time. Ensure both are CUDA events; otherwise, an exception will be thrown
                    elapsed_time = start_event.elapsed_time(end_event)
                    print(f"{task_type} took {elapsed_time} ms")
                    self.add_metric_cpu(task_type, elapsed_time)
                except AttributeError as e:
                    print(f"Error processing measurement for {task_type}: {e}")
                    # Handle error, e.g., by skipping this measurement or logging the issue
            
            # Update the metrics dictionary with processed measurements (either original values or calculated elapsed times)
            self.raw_metrics[task_type] = []


    def set_disable(self):
        self.disable = True


    @if_enabled
    def plot_metrics(self):
        wandb.init(project="model-merging", group="across tasks")
        wandb.log(self.metrics)
        wandb.log("Memory use",self.memory_usage)
        for task_type, measurements in self.metrics.items():
            print("Logging metrics")
            #measurements.plot_cdf(self.full_path, f"{task_type}","size")
        wandb.finish()
