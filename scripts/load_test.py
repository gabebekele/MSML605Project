import os
import sys
import csv
import time
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# make sure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.embedding import embed_pair
from m1.similarity import cosine_similarity


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pair(row, config):
    start_time = time.perf_counter()

    left_vector, right_vector = embed_pair(
        row["left_path"],
        row["right_path"],
        config,
        normalization=None
    )

    _ = cosine_similarity(
        left_vector.reshape(1, -1),
        right_vector.reshape(1, -1)
    )[0]

    latency = time.perf_counter() - start_time
    return latency


def main():
    config = load_config("configs/m1.yaml")

    with open(config["pairs_paths"]["test"], "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)[:50]

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=4) as executor:
        latencies = list(executor.map(lambda row: run_pair(row, config), rows))

    wall_time = time.perf_counter() - wall_start

    print(f"total_requests: {len(latencies)}")
    print(f"total_wall_time_seconds: {float(wall_time):.6f}")
    print(f"throughput_requests_per_second: {float(len(latencies) / wall_time):.6f}")
    print(f"mean_latency_seconds: {float(np.mean(latencies)):.6f}")
    print(f"p95_latency_seconds: {float(np.percentile(latencies, 95)):.6f}")


if __name__ == "__main__":
    main()