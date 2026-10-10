import os
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "keyword_cnn_fp32.keras"
)

DEPLOYMENT_DIR = "deployment"

OUTPUT_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "keyword_cnn_fp32.tflite"
)

os.makedirs(
    DEPLOYMENT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FP32 KERAS -> LITERT CONVERSION")
print("=" * 60)

print("\nLoading Keras model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# CONVERT TO LITERT
# ============================================================

print("\nConverting model...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    OUTPUT_PATH,
    "wb"
) as f:
    f.write(tflite_model)


# ============================================================
# MODEL SIZE
# ============================================================

model_size_bytes = os.path.getsize(
    OUTPUT_PATH
)

model_size_mb = (
    model_size_bytes /
    (1024 * 1024)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(
    f"Output model : {OUTPUT_PATH}"
)

print(
    f"Model size   : {model_size_mb:.2f} MB"
)

print(
    f"Model bytes  : {model_size_bytes}"
)


# ============================================================
# TEST LITERT MODEL
# ============================================================

print("\nTesting LiteRT model...")

interpreter = tf.lite.Interpreter(
    model_path=OUTPUT_PATH
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)


# ============================================================
# RUN ONE INFERENCE
# ============================================================

import numpy as np

input_shape = input_details[0]["shape"]

dummy_input = np.zeros(
    input_shape,
    dtype=input_details[0]["dtype"]
)

interpreter.set_tensor(
    input_details[0]["index"],
    dummy_input
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nTest inference successful.")

print(
    f"Input shape  : {input_shape}"
)

print(
    f"Input dtype  : {input_details[0]['dtype']}"
)

print(
    f"Output shape : {output.shape}"
)

print(
    f"Output dtype : {output.dtype}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("LITERT FP32 MODEL READY")
print("=" * 60)