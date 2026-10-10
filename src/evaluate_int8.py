import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from preprocess import build_dataset, COMMANDS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "deployment",
    "keyword_cnn_int8.tflite"
)

RESULTS_DIR = os.path.join(
    "results",
    "metrics"
)

FIGURES_DIR = os.path.join(
    "results",
    "figures"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

BATCH_SIZE = 1


# ============================================================
# LOAD INT8 MODEL
# ============================================================

print("=" * 60)
print("INT8 LITERT MODEL EVALUATION")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

input_scale = input_details[0]["quantization"][0]
input_zero_point = input_details[0]["quantization"][1]

output_scale = output_details[0]["quantization"][0]
output_zero_point = output_details[0]["quantization"][1]

print("INT8 model loaded successfully.")


# ============================================================
# BUILD TEST DATASET
# ============================================================

print("\nBuilding test dataset...")

test_ds = build_dataset(
    split="test",
    batch_size=BATCH_SIZE,
    training=False
)

print("Test dataset ready.")


# ============================================================
# EVALUATE
# ============================================================

print("\n" + "=" * 60)
print("RUNNING INT8 TEST EVALUATION")
print("=" * 60)

y_true = []
y_pred = []

total_samples = 0
correct_samples = 0

for features, labels in test_ds:

    features_np = features.numpy()

    # --------------------------------------------------------
    # FLOAT32 -> INT8
    # --------------------------------------------------------

    input_quantized = np.round(
        features_np / input_scale
        + input_zero_point
    ).astype(np.int8)

    # --------------------------------------------------------
    # INT8 INFERENCE
    # --------------------------------------------------------

    interpreter.set_tensor(
        input_index,
        input_quantized
    )

    interpreter.invoke()

    output_quantized = interpreter.get_tensor(
        output_index
    )

    # --------------------------------------------------------
    # INT8 -> FLOAT32
    # --------------------------------------------------------

    output_float = (
        output_quantized.astype(np.float32)
        - output_zero_point
    ) * output_scale

    prediction = int(
        np.argmax(output_float[0])
    )

    true_label = int(
        labels.numpy()[0]
    )

    y_true.append(true_label)
    y_pred.append(prediction)

    total_samples += 1

    if prediction == true_label:
        correct_samples += 1

    if total_samples % 500 == 0:
        print(
            f"Processed: "
            f"{total_samples}/3086"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# ACCURACY
# ============================================================

accuracy = (
    correct_samples /
    total_samples
)

print("\n" + "=" * 60)
print("INT8 ACCURACY")
print("=" * 60)

print(
    f"Correct predictions : "
    f"{correct_samples}"
)

print(
    f"Total samples       : "
    f"{total_samples}"
)

print(
    f"INT8 Accuracy        : "
    f"{accuracy:.4f}"
)

print(
    f"INT8 Accuracy        : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("INT8 CLASSIFICATION REPORT")
print("=" * 60)

report_text = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS)),
    target_names=COMMANDS,
    zero_division=0
)

print(report_text)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS)),
    target_names=COMMANDS,
    output_dict=True,
    zero_division=0
)


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(COMMANDS)),
        zero_division=0
    )
)

print("Per-class performance:")

for i, command in enumerate(COMMANDS):

    print(
        f"{command:>6s} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Support: {support[i]}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    MODEL_PATH
)

model_size_mb = (
    model_size_bytes /
    (1024 * 1024)
)

print("\nINT8 model size:")
print(
    f"{model_size_mb:.2f} MB"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "keyword_cnn_int8",

    "model_path": MODEL_PATH,

    "number_of_test_samples": int(
        total_samples
    ),

    "correct_predictions": int(
        correct_samples
    ),

    "accuracy": float(
        accuracy
    ),

    "accuracy_percent": float(
        accuracy * 100
    ),

    "model_size_bytes": int(
        model_size_bytes
    ),

    "model_size_mb": float(
        model_size_mb
    ),

    "input_scale": float(
        input_scale
    ),

    "input_zero_point": int(
        input_zero_point
    ),

    "output_scale": float(
        output_scale
    ),

    "output_zero_point": int(
        output_zero_point
    ),

    "classes": COMMANDS,

    "classification_report": report_dict,

    "confusion_matrix": cm.tolist()
}


json_path = os.path.join(
    RESULTS_DIR,
    "int8_evaluation.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "int8_classification_report.csv"
)

with open(
    csv_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "keyword,precision,recall,f1_score,support\n"
    )

    for i, command in enumerate(COMMANDS):

        f.write(
            f"{command},"
            f"{precision[i]:.6f},"
            f"{recall[i]:.6f},"
            f"{f1[i]:.6f},"
            f"{support[i]}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\nResults saved:")
print(json_path)
print(csv_path)

print("\n" + "=" * 60)
print("INT8 EVALUATION COMPLETE")
print("=" * 60)