import time
import numpy as np
import os
import json

from similarity import cosine_similarity, euclidean_distance


def cosine_loop(a, b, eps=1e-12):
    n = a.shape[0]
    out = np.empty(n, dtype=float)

    for i in range(n):
        dot = float(np.dot(a[i], b[i]))
        denominator = np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + eps
        out[i] = dot / denominator

    return out


def euclidean_loop(a, b):
    n = a.shape[0]
    out = np.empty(n, dtype=float)

    for i in range(n):
        out[i] = np.linalg.norm(a[i] - b[i])

    return out


def main():
    n = 15000
    d = 300

    rng = np.random.default_rng(123)
    a = rng.normal(size=(n, d))
    b = rng.normal(size=(n, d))

    # Cos
    t0 = time.perf_counter()
    c_loop = cosine_loop(a, b)
    loop_time_cos = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_vec = cosine_similarity(a, b)
    vec_time_cos = time.perf_counter() - t0

    c_diff = float(np.max(np.abs(c_loop - c_vec)))

    # Eucl
    t0 = time.perf_counter()
    e_loop = euclidean_loop(a, b)
    loop_time_euc = time.perf_counter() - t0

    t0 = time.perf_counter()
    e_vec = euclidean_distance(a, b)
    vec_time_euc = time.perf_counter() - t0

    e_diff = float(np.max(np.abs(e_loop - e_vec)))

    print("Cosine loop time:", loop_time_cos)
    print("Cosine vectorized time:", vec_time_cos)
    print("Cosine max diff:", c_diff)
    print("Euclidean loop time:", loop_time_euc)
    print("Euclidean vectorized time:", vec_time_euc)
    print("Euclidean max diff:", e_diff)

    os.makedirs("outputs", exist_ok=True)

    results = {
        "cosine": {
            "loop_time_sec": loop_time_cos,
            "vectorized_time_sec": vec_time_cos,
            "max_diff": c_diff,
        },
        "euclidean": {
            "loop_time_sec": loop_time_euc,
            "vectorized_time_sec": vec_time_euc,
            "max_diff": e_diff,
        }
    }

    outpath = os.path.join("outputs", "bench_similarity.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark results -> {outpath}")

if __name__ == "__main__":
    main()
