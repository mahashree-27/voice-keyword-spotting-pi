import os
import json
import numpy as np
import tensorflow as tf

from preprocess import build_dataset


# ============================================================
# CONFIGURATION
# ============================================================

FP32_MODEL_PATH = os.path.join(
    "models",
    "keyword_cnn_fp32.keras"
)

DEPLOYMENT_DIR = "deployment"

OUTPUT_PATH = os.path.join(
    DEPLOYMENT_DIR,
    "keyword_cnn_int8.tflite"
)

RESULTS_DIR = os.path.join(
    "results",
    "metrics"
)

os.makedirs(
    DEPLOYMENT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

BATCH_SIZE = 1

# Number of representative samples
REPRESENTATIVE_SAMPLES = 300


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("INT8 LITERT QUANTIZATION")
print("=" * 60)

print("\nLoading FP32 Keras model:")
print(FP32_MODEL_PATH)

model = tf.keras.models.load_model(
    FP32_MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# BUILD REPRESENTATIVE DATASET
# ============================================================

print("\nBuilding representative dataset...")

representative_ds = build_dataset(
    split="train",
    batch_size=BATCH_SIZE,
    training=False
)

print("Training dataset ready.")


# ============================================================
# REPRESENTATIVE DATA GENERATOR
# ============================================================

def representative_dataset():
    count = 0

    for features, _ in representative_ds:

        # Convert to NumPy
        features_np = features.numpy()

        yield [features_np]

        count += features_np.shape[0]

        if count >= REPRESENTATIVE_SAMPLES:
            break


print(
    f"Representative samples: "
    f"{REPRESENTATIVE_SAMPLES}"
)


# ============================================================
# CREATE CONVERTER
# ============================================================

print("\nCreating LiteRT converter...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)


# ============================================================
# ENABLE FULL INTEGER QUANTIZATION
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# ============================================================
# CONVERT
# ============================================================

print("\nConverting to full INT8...")

tflite_int8_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_PATH,
    "wb"
) as f:

    f.write(
        tflite_int8_model
    )


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
# LOAD INT8 INTERPRETER
# ============================================================

print("\nLoading INT8 LiteRT model...")

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


# ============================================================
# PRINT MODEL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("INT8 MODEL INFORMATION")
print("=" * 60)

print("\nInput details:")
print(input_details)

print("\nOutput details:")
print(output_details)

print(
    f"\nModel size: {model_size_mb:.2f} MB"
)


# ============================================================
# CHECK DATA TYPES
# ============================================================

input_dtype = input_details[0]["dtype"]
output_dtype = output_details[0]["dtype"]

print(
    f"\nInput dtype : {input_dtype}"
)

print(
    f"Output dtype: {output_dtype}"
)


# ============================================================
# CHECK QUANTIZATION PARAMETERS
# ============================================================

input_scale = (
    input_details[0]
    ["quantization"]
    [0]
)

input_zero_point = (
    input_details[0]
    ["quantization"]
    [1]
)

output_scale = (
    output_details[0]
    ["quantization"]
    [0]
)

output_zero_point = (
    output_details[0]
    ["quantization"]
    [1]
)

print("\nInput quantization:")
print(
    f"Scale       : {input_scale}"
)

print(
    f"Zero point  : {input_zero_point}"
)

print("\nOutput quantization:")
print(
    f"Scale       : {output_scale}"
)

print(
    f"Zero point  : {output_zero_point}"
)


# ============================================================
# TEST ONE INFERENCE
# ============================================================

print("\nTesting INT8 inference...")

test_ds = build_dataset(
    split="test",
    batch_size=1,
    training=False
)

features, labels = next(
    iter(test_ds)
)

features_np = features.numpy()

# Quantize input from float32 to int8
input_quantized = np.round(
    features_np / input_scale
    + input_zero_point
).astype(
    np.int8
)

interpreter.set_tensor(
    input_details[0]["index"],
    input_quantized
)

interpreter.invoke()

output_quantized = (
    interpreter.get_tensor(
        output_details[0]["index"]
    )
)

# Dequantize output back to float
output_float = (
    output_quantized.astype(
        np.float32
    ) - output_zero_point
) * output_scale

predicted_class = int(
    np.argmax(output_float)
)

true_class = int(
    labels.numpy()[0]
)


print("\nTest inference successful.")

print(
    f"True class      : {true_class}"
)

print(
    f"Predicted class  : {predicted_class}"
)

print(
    f"Input dtype      : {input_quantized.dtype}"
)

print(
    f"Output dtype     : {output_quantized.dtype}"
)


# ============================================================
# SAVE QUANTIZATION INFORMATION
# ============================================================

quantization_info = {

    "model": "keyword_cnn_int8",

    "source_model": FP32_MODEL_PATH,

    "representative_samples": (
        REPRESENTATIVE_SAMPLES
    ),

    "model_size_bytes": int(
        model_size_bytes
    ),

    "model_size_mb": float(
        model_size_mb
    ),

    "input_dtype": str(
        input_dtype
    ),

    "output_dtype": str(
        output_dtype
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
    )
}


json_path = os.path.join(
    RESULTS_DIR,
    "int8_quantization.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        quantization_info,
        f,
        indent=4
    )


print("\nQuantization information saved:")
print(json_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("INT8 QUANTIZATION COMPLETE")
print("=" * 60)

print(
    f"INT8 model: {OUTPUT_PATH}"
)

print(
    f"INT8 size : {model_size_mb:.2f} MB"
)