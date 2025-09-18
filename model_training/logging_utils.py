# model_training/logging_utils.py

import os
import time
import threading
from datetime import datetime
import psutil
import logging
from model_training.debug_utils import save_checkpoint

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

def setup_main_logger(log_file_path, level=logging.INFO):
    """
    Setup the main logger to log both to console and to a file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Formatter
    fmt = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def get_gpu_stats():
    stats = []
    if not NVML_AVAILABLE:
        return stats

    count = pynvml.nvmlDeviceGetCount()
    for i in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats.append({
            "gpu_index": i,
            "gpu_mem_used_MB": mem.used / (1024 ** 2),
            "gpu_mem_total_MB": mem.total / (1024 ** 2),
            "gpu_util_percent": util.gpu,
        })
    return stats


def get_cpu_ram_stats():
    vm = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=None)
    return {
        "cpu_percent": cpu_percent,
        "ram_used_MB": (vm.total - vm.available) / (1024 ** 2),
        "ram_total_MB": vm.total / (1024 ** 2),
    }


def hardware_logger(log_path, interval=5, stop_event=None, logger=None):
    """
    Periodically logs hardware usage stats.
    If `logger` is provided, uses that; else writes to CSV file at log_path.
    """
    write_header = not os.path.exists(log_path)
    if logger is None:
        # fallback to writing raw CSV
        with open(log_path, "a") as f:
            if write_header:
                parts = [
                    "timestamp",
                    "cpu_percent", "ram_used_MB", "ram_total_MB"
                ]
                if NVML_AVAILABLE:
                    parts += [
                        "gpu_index", "gpu_util_percent",
                        "gpu_mem_used_MB", "gpu_mem_total_MB"
                    ]
                f.write(",".join(parts) + "\n")

            while (stop_event is None or not stop_event.is_set()):
                ts = datetime.now().isoformat()
                cpu_ram = get_cpu_ram_stats()
                gpu_stats = get_gpu_stats()

                if gpu_stats:
                    for gs in gpu_stats:
                        row = [
                            ts,
                            f"{cpu_ram['cpu_percent']:.2f}",
                            f"{cpu_ram['ram_used_MB']:.2f}",
                            f"{cpu_ram['ram_total_MB']:.2f}",
                            str(gs["gpu_index"]),
                            str(gs["gpu_util_percent"]),
                            f"{gs['gpu_mem_used_MB']:.2f}",
                            f"{gs['gpu_mem_total_MB']:.2f}"
                        ]
                        f.write(",".join(row) + "\n")
                else:
                    row = [
                        ts,
                        f"{cpu_ram['cpu_percent']:.2f}",
                        f"{cpu_ram['ram_used_MB']:.2f}",
                        f"{cpu_ram['ram_total_MB']:.2f}"
                    ]
                    f.write(",".join(row) + "\n")
                f.flush()
                time.sleep(interval)
    else:
        # If logger provided, log via logger.info
        while (stop_event is None or not stop_event.is_set()):
            ts = datetime.now().isoformat()
            cpu_ram = get_cpu_ram_stats()
            gpu_stats = get_gpu_stats()

            if gpu_stats:
                for gs in gpu_stats:
                    msg = (f"HW {ts} | CPU {cpu_ram['cpu_percent']:.2f}% | "
                           f"RAM {cpu_ram['ram_used_MB']:.2f}/{cpu_ram['ram_total_MB']:.2f} MB | "
                           f"GPU{gs['gpu_index']} {gs['gpu_util_percent']}% | "
                           f"GPU_mem {gs['gpu_mem_used_MB']:.2f}/{gs['gpu_mem_total_MB']:.2f} MB")
                    logger.info(msg)
            else:
                msg = (f"HW {ts} | CPU {cpu_ram['cpu_percent']:.2f}% | "
                       f"RAM {cpu_ram['ram_used_MB']:.2f}/{cpu_ram['ram_total_MB']:.2f} MB")
                logger.info(msg)

            time.sleep(interval)


def start_hardware_logging(output_dir, interval=5):
    """
    Helper to start the hardware logger in a background thread, returns (stop_event, thread).
    """
    log_path = os.path.join(output_dir, "hardware_usage.csv")
    os.makedirs(output_dir, exist_ok=True)

    stop_event = threading.Event()
    t = threading.Thread(target=hardware_logger, args=(log_path, interval, stop_event))
    t.start()
    return stop_event, t

