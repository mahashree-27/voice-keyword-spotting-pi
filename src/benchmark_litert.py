import os
import sys
import json
import time
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess_audio


# ============================================================
# CONFIGURATION
# ============================================================

FP32_MODEL = "deployment/keyword_cnn_fp32.tflite"
INT8_MODEL = "deployment/keyword_cnn_int8.tflite"

TEST_DIR = "data/split/test"

OUTPUT_FILE = "results/metrics/litert_benchmark.json"

WARMUP_RUNS = 20
BENCHMARK_RUNS = 200


# ============================================================
# FIND ONE TEST AUDIO FILE
# ============================================================

print("=" * 60)
print("LITERT FP32 vs INT8 BENCHMARK")
print("=" * 60)

print("\nSearching for a test audio sample...")

test_audio = None

for root, dirs, files in os.walk(TEST_DIR):
    for file in files:
        if file.lower().endswith(".wav"):
            test_audio = os.path.join(root, file)
            break

    if test_audio is not None:
        break


if test_audio is None:
    raise FileNotFoundError(
        "No WAV files found in data/split/test"
    )


print("Test sample:", test_audio)


# ============================================================
# PREPROCESS AUDIO
# ============================================================

print("\nPreprocessing test audio...")

# preprocess_audio() expects the WAV FILE PATH,
# not the decoded waveform.

feature, _ = preprocess_audio(
    tf.constant(test_audio),
    tf.constant(0)
)

feature_np = feature.numpy()

# Add batch dimension
feature_np = np.expand_dims(
    feature_np,
    axis=0
)

print("Feature shape:", feature_np.shape)
print("Feature dtype:", feature_np.dtype)


# ============================================================
# BENCHMARK FUNCTION
# ============================================================

def benchmark_model(model_path, model_type):

    print("\n" + "=" * 60)
    print(f"BENCHMARKING {model_type}")
    print("=" * 60)

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model_size_bytes = os.path.getsize(
        model_path
    )

    model_size_mb = (
        model_size_bytes /
        (1024 * 1024)
    )

    print("Model:", model_path)
    print(
        f"Model size: "
        f"{model_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Load LiteRT model
    # --------------------------------------------------------

    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=1
    )

    interpreter.allocate_tensors()

    input_details = (
        interpreter.get_input_details()
    )

    output_details = (
        interpreter.get_output_details()
    )

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    input_dtype = input_details[0]["dtype"]
    output_dtype = output_details[0]["dtype"]

    print(
        "Input shape:",
        input_details[0]["shape"]
    )

    print(
        "Input dtype:",
        input_dtype
    )

    print(
        "Output shape:",
        output_details[0]["shape"]
    )

    print(
        "Output dtype:",
        output_dtype
    )

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    if input_dtype == np.int8:

        input_scale, input_zero_point = (
            input_details[0]["quantization"]
        )

        print(
            "Input scale:",
            input_scale
        )

        print(
            "Input zero point:",
            input_zero_point
        )

        # Float feature -> INT8
        input_data = np.round(
            feature_np / input_scale
            + input_zero_point
        ).astype(np.int8)

    else:

        # FP32 input
        input_data = feature_np.astype(
            np.float32
        )

    # --------------------------------------------------------
    # Verify input shape
    # --------------------------------------------------------

    expected_shape = tuple(
        input_details[0]["shape"]
    )

    if tuple(input_data.shape) != expected_shape:

        # Some LiteRT models have a dynamic batch
        # dimension. Resize only if necessary.

        if (
            len(expected_shape) == 4
            and expected_shape[1:] ==
            input_data.shape[1:]
        ):

            interpreter.resize_tensor_input(
                input_index,
                input_data.shape
            )

            interpreter.allocate_tensors()

            input_details = (
                interpreter.get_input_details()
            )

            output_details = (
                interpreter.get_output_details()
            )

            input_index = (
                input_details[0]["index"]
            )

            output_index = (
                output_details[0]["index"]
            )

        else:

            raise ValueError(
                f"Input shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {input_data.shape}"
            )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"\nWarm-up runs: "
        f"{WARMUP_RUNS}"
    )

    for _ in range(WARMUP_RUNS):

        interpreter.set_tensor(
            input_index,
            input_data
        )

        interpreter.invoke()

        # Read output to ensure inference completes
        interpreter.get_tensor(
            output_index
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        f"Benchmark runs: "
        f"{BENCHMARK_RUNS}"
    )

    latencies_ms = []

    for _ in range(BENCHMARK_RUNS):

        # Set input before starting timer
        interpreter.set_tensor(
            input_index,
            input_data
        )

        # Measure MODEL INFERENCE only
        start = time.perf_counter()

        interpreter.invoke()

        end = time.perf_counter()

        # Retrieve output after timing
        interpreter.get_tensor(
            output_index
        )

        latency_ms = (
            end - start
        ) * 1000.0

        latencies_ms.append(
            latency_ms
        )

    latencies_ms = np.array(
        latencies_ms,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mean_latency = float(
        np.mean(latencies_ms)
    )

    median_latency = float(
        np.median(latencies_ms)
    )

    p95_latency = float(
        np.percentile(
            latencies_ms,
            95
        )
    )

    min_latency = float(
        np.min(latencies_ms)
    )

    max_latency = float(
        np.max(latencies_ms)
    )

    throughput = (
        1000.0 /
        mean_latency
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nResults:")

    print(
        f"Mean latency   : "
        f"{mean_latency:.4f} ms"
    )

    print(
        f"Median latency : "
        f"{median_latency:.4f} ms"
    )

    print(
        f"P95 latency    : "
        f"{p95_latency:.4f} ms"
    )

    print(
        f"Minimum        : "
        f"{min_latency:.4f} ms"
    )

    print(
        f"Maximum        : "
        f"{max_latency:.4f} ms"
    )

    print(
        f"Throughput     : "
        f"{throughput:.2f} "
        f"inferences/sec"
    )

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return {

        "model_type": model_type,

        "model_path": model_path,

        "model_size_bytes":
            model_size_bytes,

        "model_size_mb":
            model_size_mb,

        "input_dtype":
            str(input_dtype),

        "output_dtype":
            str(output_dtype),

        "input_shape":
            list(input_data.shape),

        "num_threads": 1,

        "warmup_runs":
            WARMUP_RUNS,

        "benchmark_runs":
            BENCHMARK_RUNS,

        "mean_latency_ms":
            mean_latency,

        "median_latency_ms":
            median_latency,

        "p95_latency_ms":
            p95_latency,

        "min_latency_ms":
            min_latency,

        "max_latency_ms":
            max_latency,

        "throughput_inferences_per_second":
            throughput
    }


# ============================================================
# RUN BENCHMARKS
# ============================================================

fp32_results = benchmark_model(
    FP32_MODEL,
    "LiteRT FP32"
)

int8_results = benchmark_model(
    INT8_MODEL,
    "LiteRT INT8"
)


# ============================================================
# COMPARISON
# ============================================================

latency_reduction = (
    1.0 -
    (
        int8_results["mean_latency_ms"] /
        fp32_results["mean_latency_ms"]
    )
) * 100.0


speedup = (
    fp32_results["mean_latency_ms"] /
    int8_results["mean_latency_ms"]
)


size_reduction = (
    1.0 -
    (
        int8_results["model_size_mb"] /
        fp32_results["model_size_mb"]
    )
) * 100.0


comparison = {

    "fp32": fp32_results,

    "int8": int8_results,

    "comparison": {

        "mean_latency_reduction_percent":
            latency_reduction,

        "speedup_factor":
            speedup,

        "model_size_reduction_percent":
            size_reduction
    }
}


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(
    f"LiteRT FP32 mean latency : "
    f"{fp32_results['mean_latency_ms']:.4f} ms"
)

print(
    f"LiteRT INT8 mean latency : "
    f"{int8_results['mean_latency_ms']:.4f} ms"
)

print(
    f"\nLatency reduction        : "
    f"{latency_reduction:.2f}%"
)

print(
    f"Speedup                  : "
    f"{speedup:.2f}x"
)

print(
    f"\nFP32 model size          : "
    f"{fp32_results['model_size_mb']:.2f} MB"
)

print(
    f"INT8 model size          : "
    f"{int8_results['model_size_mb']:.2f} MB"
)

print(
    f"Model size reduction     : "
    f"{size_reduction:.2f}%"
)

print("\nResults saved:")
print(OUTPUT_FILE)

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)