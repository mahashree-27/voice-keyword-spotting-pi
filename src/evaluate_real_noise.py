import os
import json
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import preprocess_audio


# ============================================================
# Configuration
# ============================================================

REAL_NOISE_DIR = "data/real_noise"

FP32_MODEL = "deployment/keyword_cnn_fp32.tflite"
INT8_MODEL = "deployment/keyword_cnn_int8.tflite"

RESULTS_DIR = "results/metrics"

COMMANDS = [
    "down",
    "left",
    "no",
    "off",
    "on",
    "right",
    "up",
    "yes",
]

NUM_CLASSES = len(COMMANDS)


# ============================================================
# Utility functions
# ============================================================

def load_interpreter(model_path):
    """Load a LiteRT/TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    return interpreter, input_details, output_details


def quantize_input(feature, input_details):
    """Convert float32 feature to INT8 using model quantization parameters."""

    scale = input_details["quantization"][0]
    zero_point = input_details["quantization"][1]

    if scale == 0:
        raise ValueError("Invalid input quantization scale: 0")

    quantized = np.round(feature / scale + zero_point)

    quantized = np.clip(
        quantized,
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max,
    ).astype(np.int8)

    return quantized


def dequantize_output(output, output_details):
    """Convert INT8 output back to float32."""

    scale = output_details["quantization"][0]
    zero_point = output_details["quantization"][1]

    return (output.astype(np.float32) - zero_point) * scale


def run_inference(
    interpreter,
    input_details,
    output_details,
    feature,
    is_int8=False,
):
    """Run one inference and return prediction + latency."""

    if is_int8:
        input_data = quantize_input(feature, input_details)
    else:
        input_data = feature.astype(np.float32)

    # Make sure batch dimension exists
    if input_data.ndim == 3:
        input_data = np.expand_dims(input_data, axis=0)

    # Set input BEFORE timing because we want model inference latency.
    interpreter.set_tensor(
        input_details["index"],
        input_data,
    )

    start = time.perf_counter()

    interpreter.invoke()

    end = time.perf_counter()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    if is_int8:
        output = dequantize_output(
            output,
            output_details,
        )

    prediction = int(np.argmax(output[0]))

    latency_ms = (end - start) * 1000.0

    return prediction, latency_ms


# ============================================================
# Main evaluation
# ============================================================

def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("REAL AMBIENT NOISE EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading FP32 LiteRT model...")
    fp32_interpreter, fp32_input, fp32_output = load_interpreter(
        FP32_MODEL
    )

    print("Loading INT8 LiteRT model...")
    int8_interpreter, int8_input, int8_output = load_interpreter(
        INT8_MODEL
    )

    print("\nFP32 input dtype:", fp32_input["dtype"])
    print("FP32 output dtype:", fp32_output["dtype"])

    print("INT8 input dtype:", int8_input["dtype"])
    print("INT8 output dtype:", int8_output["dtype"])

    print(
        "INT8 input scale:",
        int8_input["quantization"][0],
    )

    print(
        "INT8 input zero point:",
        int8_input["quantization"][1],
    )

    # --------------------------------------------------------
    # Collect WAV files
    # --------------------------------------------------------

    samples = []

    for command_index, command in enumerate(COMMANDS):

        command_dir = os.path.join(
            REAL_NOISE_DIR,
            command,
        )

        if not os.path.isdir(command_dir):
            raise FileNotFoundError(
                f"Missing directory: {command_dir}"
            )

        wav_files = sorted(
            [
                os.path.join(command_dir, f)
                for f in os.listdir(command_dir)
                if f.lower().endswith(".wav")
            ]
        )

        print(
            f"{command:>6}: {len(wav_files)} recordings"
        )

        for wav_path in wav_files:
            samples.append(
                (wav_path, command_index, command)
            )

    print("\nTotal recordings:", len(samples))

    if len(samples) == 0:
        raise RuntimeError(
            "No WAV files were found."
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    y_true = []

    fp32_predictions = []
    int8_predictions = []

    fp32_latencies = []
    int8_latencies = []

    print("\nEvaluating...")
    print("-" * 70)

    for index, (wav_path, true_label, command) in enumerate(
        samples,
        start=1,
    ):

        # ----------------------------------------------
        # Same preprocessing used during training
        # ----------------------------------------------

        feature, _ = preprocess_audio(
            tf.constant(wav_path),
            tf.constant(true_label),
        )

        feature_np = feature.numpy()

        # ----------------------------------------------
        # FP32 inference
        # ----------------------------------------------

        fp32_prediction, fp32_latency = run_inference(
            fp32_interpreter,
            fp32_input,
            fp32_output,
            feature_np,
            is_int8=False,
        )

        # ----------------------------------------------
        # INT8 inference
        # ----------------------------------------------

        int8_prediction, int8_latency = run_inference(
            int8_interpreter,
            int8_input,
            int8_output,
            feature_np,
            is_int8=True,
        )

        # ----------------------------------------------
        # Store results
        # ----------------------------------------------

        y_true.append(true_label)

        fp32_predictions.append(fp32_prediction)
        int8_predictions.append(int8_prediction)

        fp32_latencies.append(fp32_latency)
        int8_latencies.append(int8_latency)

        fp32_correct = fp32_prediction == true_label
        int8_correct = int8_prediction == true_label

        print(
            f"[{index:02d}/{len(samples)}] "
            f"True: {command:<5} | "
            f"FP32: {COMMANDS[fp32_prediction]:<5} "
            f"{'✓' if fp32_correct else '✗'} | "
            f"INT8: {COMMANDS[int8_prediction]:<5} "
            f"{'✓' if int8_correct else '✗'}"
        )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    y_true = np.array(y_true)

    fp32_predictions = np.array(
        fp32_predictions
    )

    int8_predictions = np.array(
        int8_predictions
    )

    fp32_latencies = np.array(
        fp32_latencies
    )

    int8_latencies = np.array(
        int8_latencies
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    fp32_accuracy = accuracy_score(
        y_true,
        fp32_predictions,
    )

    int8_accuracy = accuracy_score(
        y_true,
        int8_predictions,
    )

    # --------------------------------------------------------
    # Classification reports
    # --------------------------------------------------------

    fp32_report = classification_report(
        y_true,
        fp32_predictions,
        target_names=COMMANDS,
        output_dict=True,
        zero_division=0,
    )

    int8_report = classification_report(
        y_true,
        int8_predictions,
        target_names=COMMANDS,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    fp32_cm = confusion_matrix(
        y_true,
        fp32_predictions,
        labels=range(NUM_CLASSES),
    )

    int8_cm = confusion_matrix(
        y_true,
        int8_predictions,
        labels=range(NUM_CLASSES),
    )

    # --------------------------------------------------------
    # Latency statistics
    # --------------------------------------------------------

    fp32_mean = float(np.mean(fp32_latencies))
    fp32_median = float(np.median(fp32_latencies))
    fp32_p95 = float(np.percentile(fp32_latencies, 95))

    int8_mean = float(np.mean(int8_latencies))
    int8_median = float(np.median(int8_latencies))
    int8_p95 = float(np.percentile(int8_latencies, 95))

    # --------------------------------------------------------
    # Accuracy difference
    # --------------------------------------------------------

    accuracy_difference = (
        int8_accuracy - fp32_accuracy
    ) * 100

    # --------------------------------------------------------
    # Latency improvement
    # --------------------------------------------------------

    latency_reduction = (
        (fp32_mean - int8_mean)
        / fp32_mean
    ) * 100

    speedup = (
        fp32_mean / int8_mean
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL AMBIENT NOISE RESULTS")
    print("=" * 70)

    print(
        f"\nFP32 Accuracy : "
        f"{fp32_accuracy * 100:.2f}%"
    )

    print(
        f"INT8 Accuracy : "
        f"{int8_accuracy * 100:.2f}%"
    )

    print(
        f"\nAccuracy difference "
        f"(INT8 - FP32): "
        f"{accuracy_difference:+.2f} percentage points"
    )

    print("\nInference latency:")
    print(
        f"FP32 mean   : {fp32_mean:.4f} ms"
    )
    print(
        f"FP32 median : {fp32_median:.4f} ms"
    )
    print(
        f"FP32 p95    : {fp32_p95:.4f} ms"
    )

    print(
        f"\nINT8 mean   : {int8_mean:.4f} ms"
    )
    print(
        f"INT8 median : {int8_median:.4f} ms"
    )
    print(
        f"INT8 p95    : {int8_p95:.4f} ms"
    )

    print(
        f"\nLatency reduction : "
        f"{latency_reduction:.2f}%"
    )

    print(
        f"Speedup           : "
        f"{speedup:.2f}x"
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    results = {
        "dataset": "Real ambient-noise recordings",
        "total_samples": int(len(samples)),
        "commands": COMMANDS,

        "fp32": {
            "accuracy": float(fp32_accuracy),
            "accuracy_percent": float(
                fp32_accuracy * 100
            ),
            "mean_latency_ms": fp32_mean,
            "median_latency_ms": fp32_median,
            "p95_latency_ms": fp32_p95,
            "classification_report": fp32_report,
            "confusion_matrix": fp32_cm.tolist(),
        },

        "int8": {
            "accuracy": float(int8_accuracy),
            "accuracy_percent": float(
                int8_accuracy * 100
            ),
            "mean_latency_ms": int8_mean,
            "median_latency_ms": int8_median,
            "p95_latency_ms": int8_p95,
            "classification_report": int8_report,
            "confusion_matrix": int8_cm.tolist(),
        },

        "comparison": {
            "accuracy_difference_percentage_points":
                float(accuracy_difference),

            "latency_reduction_percent":
                float(latency_reduction),

            "speedup":
                float(speedup),
        },
    }

    output_file = os.path.join(
        RESULTS_DIR,
        "real_noise_evaluation.json",
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Save confusion matrices
    # --------------------------------------------------------

    np.savetxt(
        os.path.join(
            RESULTS_DIR,
            "real_noise_fp32_confusion_matrix.csv",
        ),
        fp32_cm,
        fmt="%d",
        delimiter=",",
    )

    np.savetxt(
        os.path.join(
            RESULTS_DIR,
            "real_noise_int8_confusion_matrix.csv",
        ),
        int8_cm,
        fmt="%d",
        delimiter=",",
    )

    print(
        f"\nResults saved to:\n{output_file}"
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()