import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from preprocess import build_dataset, COMMANDS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join("models", "keyword_cnn_fp32.keras")

RESULTS_DIR = os.path.join("results", "metrics")
FIGURES_DIR = os.path.join("results", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

BATCH_SIZE = 64


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FP32 MODEL EVALUATION")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


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
# BASIC MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print("\nFP32 Test Results:")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PREDICTIONS")
print("=" * 60)

y_true = []
y_pred = []

for features, labels in test_ds:

    logits = model(features, training=False)

    predictions = tf.argmax(
        logits,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predictions.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"Number of test samples: {len(y_true)}")


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS))
)

print("\nLabels:")
print(COMMANDS)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS)),
    target_names=COMMANDS,
    output_dict=True,
    zero_division=0
)

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(COMMANDS)),
        target_names=COMMANDS,
        zero_division=0
    )
)


# ============================================================
# PER-CLASS PRECISION / RECALL / F1
# ============================================================

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=np.arange(len(COMMANDS)),
    zero_division=0
)

print("\nPer-class performance:")

for i, command in enumerate(COMMANDS):
    print(
        f"{command:>6s} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Support: {support[i]}"
    )


# ============================================================
# SAVE CONFUSION MATRIX FIGURE
# ============================================================

plt.figure(figsize=(9, 8))

plt.imshow(cm)

plt.title("FP32 CNN Confusion Matrix")
plt.xlabel("Predicted Keyword")
plt.ylabel("True Keyword")

plt.xticks(
    ticks=np.arange(len(COMMANDS)),
    labels=COMMANDS,
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(COMMANDS)),
    labels=COMMANDS
)

# Write values inside cells
for i in range(len(COMMANDS)):
    for j in range(len(COMMANDS)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar(label="Number of Samples")
plt.tight_layout()

cm_path = os.path.join(
    FIGURES_DIR,
    "fp32_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved:")
print(cm_path)


# ============================================================
# MODEL FILE SIZE
# ============================================================

model_size_bytes = os.path.getsize(MODEL_PATH)
model_size_mb = model_size_bytes / (1024 * 1024)

print("\n" + "=" * 60)
print("MODEL SIZE")
print("=" * 60)

print(f"FP32 model size: {model_size_mb:.2f} MB")


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {
    "model": "keyword_cnn_fp32",
    "model_path": MODEL_PATH,

    "test_loss": float(test_loss),
    "test_accuracy": float(test_accuracy),

    "test_accuracy_percent": float(
        test_accuracy * 100
    ),

    "model_size_bytes": int(model_size_bytes),
    "model_size_mb": float(model_size_mb),

    "number_of_test_samples": int(len(y_true)),

    "classes": COMMANDS,

    "classification_report": report,

    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    RESULTS_DIR,
    "fp32_evaluation.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

print("\nEvaluation metrics saved:")
print(metrics_path)


# ============================================================
# SAVE CLASSIFICATION CSV
# ============================================================

csv_path = os.path.join(
    RESULTS_DIR,
    "fp32_classification_report.csv"
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

print("Classification report saved:")
print(csv_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FP32 EVALUATION COMPLETE")
print("=" * 60)