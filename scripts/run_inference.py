import os
import sys
import time
import yaml
import argparse
from pathlib import Path

# make sure project root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.embedding import embed_pair
from m1.similarity import cosine_similarity
from src.thresholding import apply_threshold


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1.yaml")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = 0.30612244897959173

    start_time = time.perf_counter()

    left_vector, right_vector = embed_pair(
        args.left,
        args.right,
        config,
        normalization=None
    )

    score = cosine_similarity(
        left_vector.reshape(1, -1),
        right_vector.reshape(1, -1)
    )[0]

    predicted_label = apply_threshold(
        scores=[score],
        threshold=threshold,
        higher_is_same=True
    )[0]

    confidence = abs(float(score) - float(threshold))
    latency = time.perf_counter() - start_time

    print(f"left_path: {args.left}")
    print(f"right_path: {args.right}")
    print(f"score: {float(score):.6f}")
    print(f"threshold: {float(threshold):.6f}")
    print(f"prediction: {int(predicted_label)}")
    print(f"confidence: {float(confidence):.6f}")
    print(f"latency_seconds: {float(latency):.6f}")


if __name__ == "__main__":
    main()
