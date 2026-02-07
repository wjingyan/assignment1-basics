import time
import psutil
import os
from typing import Optional


def format_bytes(b: int) -> str:
    """Return the given bytes as a human-readable string, e.g. 1.2 GB or 512.3 MB."""
    gb = b / (1024**3)
    if gb >= 1:
        return f"{gb:.2f} GB"
    mb = b / (1024**2)
    return f"{mb:.2f} MB"


def find_top_process(name: str) -> Optional[psutil.Process]:
    """Find the process with the given name that is using the most memory."""
    top_process = None
    max_mem = 0
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            # Check if the process name contains the target name (more robust)
            if name.lower() in proc.name().lower():
                mem_info = proc.memory_info()
                # If this process uses more memory, it's our new top process
                if mem_info.rss > max_mem:
                    max_mem = mem_info.rss
                    top_process = proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Ignore processes that have terminated or we can't access
            pass
    return top_process


def monitor_system(interval_seconds: int = 5, log_file: str = "resource_log.txt", process_name: str = "python"):
    """
    Continuously monitors and prints system-wide RAM and Swap memory usage.
    Also tracks the highest memory-consuming process with a given name.

    Args:
        interval_seconds: The refresh interval in seconds.
        log_file: The file to write logs to.
        process_name: The name of the process to monitor (e.g., 'python').
    """
    print(f"--- System Memory Monitor ---")
    print(f"Logging to {log_file}. Press Ctrl+C to stop.")

    # Write a header to the log file each time the script starts
    with open(log_file, "w") as f:
        f.write("Timestamp | System RAM Usage | System Swap Usage | Top Process (PID, Name, RAM)\n")
        f.write("-" * 100 + "\n")

    try:
        while True:
            # Get system-wide memory stats
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            # Find the top memory-consuming process by name (e.g., "python")
            top_proc = find_top_process(process_name)
            
            proc_info_str = "Process not found"
            if top_proc:
                proc_mem = format_bytes(top_proc.memory_info().rss)
                proc_info_str = f"PID: {top_proc.pid}, Name: {top_proc.name()}, RAM: {proc_mem}"

            # Format for printing and logging
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            ram_str = f"{format_bytes(virtual_mem.used)} / {format_bytes(virtual_mem.total)} ({virtual_mem.percent}%)"
            swap_str = f"{format_bytes(swap_mem.used)} / {format_bytes(swap_mem.total)} ({swap_mem.percent}%)"
            
            log_line = f"{timestamp} | RAM: {ram_str} | Swap: {swap_str} | Top Process: {proc_info_str}"

            print(log_line)
            with open(log_file, "a") as f:
                f.write(log_line + "\n")
                f.flush()  # Ensure data is written to disk immediately

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopping memory monitor.")


if __name__ == "__main__":
    # This script can be run from the command line to monitor system resources.
    # It will log to a file and also track the python process using the most memory.
    # Example: `uv run python cs336_basics/resource_monitor.py`
    monitor_system(interval_seconds=5)
