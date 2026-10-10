import os
import time
import json
import numpy as np
import tensorflow as tf

from preprocess import build_dataset


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "keyword_cnn_fp32.keras"
)

RESULTS_DIR = os.path.join(
    "results",
    "metrics"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 1

WARMUP_RUNS = 20
BENCHMARK_RUNS = 200


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FP32 INFERENCE BENCHMARK")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# LOAD ONE TEST SAMPLE
# ============================================================

print("\nLoading one test sample...")

test_ds = build_dataset(
    split="test",
    batch_size=BATCH_SIZE,
    training=False
)

features, labels = next(iter(test_ds))

print("Feature shape:", features.shape)
print("Label:", labels.numpy())


# ============================================================
# WARM-UP
# ============================================================

print("\nRunning warm-up inference...")

for _ in range(WARMUP_RUNS):
    _ = model(
        features,
        training=False
    )

print(
    f"Warm-up complete: {WARMUP_RUNS} runs"
)


# ============================================================
# BENCHMARK
# ============================================================

print("\nRunning benchmark...")

latencies_ms = []

for i in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    _ = model(
        features,
        training=False
    )

    # Force TensorFlow result synchronization
    # by converting the output to NumPy.
    _.numpy()

    end = time.perf_counter()

    latency_ms = (
        end - start
    ) * 1000.0

    latencies_ms.append(
        latency_ms
    )


latencies_ms = np.array(
    latencies_ms
)


# ============================================================
# STATISTICS
# ============================================================

mean_latency = np.mean(
    latencies_ms
)

median_latency = np.median(
    latencies_ms
)

p95_latency = np.percentile(
    latencies_ms,
    95
)

min_latency = np.min(
    latencies_ms
)

max_latency = np.max(
    latencies_ms
)

throughput = (
    1000.0 / mean_latency
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FP32 BENCHMARK RESULTS")
print("=" * 60)

print(
    f"Warm-up runs       : {WARMUP_RUNS}"
)

print(
    f"Benchmark runs     : {BENCHMARK_RUNS}"
)

print(
    f"Mean latency       : {mean_latency:.3f} ms"
)

print(
    f"Median latency     : {median_latency:.3f} ms"
)

print(
    f"P95 latency        : {p95_latency:.3f} ms"
)

print(
    f"Minimum latency    : {min_latency:.3f} ms"
)

print(
    f"Maximum latency    : {max_latency:.3f} ms"
)

print(
    f"Throughput         : {throughput:.2f} inferences/sec"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "keyword_cnn_fp32",

    "model_path": MODEL_PATH,

    "batch_size": BATCH_SIZE,

    "warmup_runs": WARMUP_RUNS,

    "benchmark_runs": BENCHMARK_RUNS,

    "mean_latency_ms": float(
        mean_latency
    ),

    "median_latency_ms": float(
        median_latency
    ),

    "p95_latency_ms": float(
        p95_latency
    ),

    "min_latency_ms": float(
        min_latency
    ),

    "max_latency_ms": float(
        max_latency
    ),

    "throughput_inferences_per_second": float(
        throughput
    )
}


output_path = os.path.join(
    RESULTS_DIR,
    "fp32_benchmark.json"
)

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nBenchmark results saved:")
print(output_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FP32 BENCHMARK COMPLETE")
print("=" * 60)