import os
import numpy as np
import tensorflow as tf
import sounddevice as sd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR, "deployment", "keyword_cnn_int8.tflite"
)

# Headset Microphone on the user's system
MIC_DEVICE = 1

SAMPLE_RATE = 16000
CHANNELS = 1

# Listen for exactly 1 second after pressing ENTER
RECORD_SECONDS = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * RECORD_SECONDS)

# Lower threshold for microphone activity
ENERGY_THRESHOLD = 0.003

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


# ============================================================
# Same preprocessing used during training/evaluation
# ============================================================

def preprocess_waveform(waveform):
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Exactly 16000 samples
    waveform = waveform[:SAMPLE_RATE]

    current_length = tf.shape(waveform)[0]
    padding = tf.maximum(0, SAMPLE_RATE - current_length)
    waveform = tf.pad(waveform, [[0, padding]])

    # STFT
    stft = tf.signal.stft(
        waveform,
        frame_length=640,
        frame_step=320,
        fft_length=1024,
    )

    magnitude = tf.abs(stft)
    power = tf.square(magnitude)

    # 64-bin Mel filterbank
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=1024 // 2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=80.0,
        upper_edge_hertz=7600.0,
    )

    mel = tf.matmul(power, mel_filterbank)

    # Log-Mel
    log_mel = tf.math.log(mel + 1e-6)

    # Per-utterance z-score normalization
    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)
    log_mel = (log_mel - mean) / (std + 1e-6)

    # Channel dimension -> (49, 64, 1)
    return tf.expand_dims(log_mel, axis=-1)


# ============================================================
# Load INT8 LiteRT model
# ============================================================

print("Loading INT8 LiteRT model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_index = input_details["index"]
output_index = output_details["index"]

input_scale, input_zero_point = input_details["quantization"]
output_scale, output_zero_point = output_details["quantization"]

print("Model loaded.")
print("Input:", input_details["shape"], input_details["dtype"])
print("Output:", output_details["shape"], output_details["dtype"])
print("Commands:", ", ".join(COMMANDS))


# ============================================================
# Check microphone
# ============================================================

try:
    device_info = sd.query_devices(MIC_DEVICE)

    if device_info["max_input_channels"] < 1:
        raise RuntimeError("Selected device has no input channel.")

    print(f"Microphone: {device_info['name']}")

except Exception as e:
    print("\nERROR: Could not access microphone device 1.")
    print("Check devices using:")
    print('python -c "import sounddevice as sd; print(sd.query_devices())"')
    print("\nDetails:", e)
    raise SystemExit(1)


# ============================================================
# Prediction
# ============================================================

def predict_from_audio(audio):

    feature = preprocess_waveform(audio)

    feature_np = feature.numpy().astype(np.float32)
    feature_np = np.expand_dims(feature_np, axis=0)

    # Float32 -> INT8
    quantized = np.round(
        feature_np / input_scale + input_zero_point
    )

    quantized = np.clip(
        quantized, -128, 127
    ).astype(np.int8)

    interpreter.set_tensor(input_index, quantized)
    interpreter.invoke()

    output = interpreter.get_tensor(output_index)

    # INT8 -> float32
    output_float = (
        output.astype(np.float32) - output_zero_point
    ) * output_scale

    # Softmax
    exp_logits = np.exp(
        output_float - np.max(output_float, axis=1, keepdims=True)
    )

    probabilities = exp_logits / np.sum(
        exp_logits, axis=1, keepdims=True
    )

    probabilities = probabilities[0]

    indices = np.argsort(probabilities)[::-1]

    top1 = indices[0]
    top2 = indices[1]

    return (
        COMMANDS[top1],
        float(probabilities[top1]),
        COMMANDS[top2],
        float(probabilities[top2]),
    )


# ============================================================
# Record ONE command after ENTER
# ============================================================

def listen_once():

    print("\nListening for 1 second... SPEAK NOW!")

    audio = sd.rec(
        NUM_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=MIC_DEVICE,
    )

    sd.wait()

    audio = audio[:, 0]

    # RMS of complete recording
    rms = float(
        np.sqrt(np.mean(np.square(audio)) + 1e-12)
    )

    print(f"RMS: {rms:.5f}")

    if rms < ENERGY_THRESHOLD:
        print("No strong speech detected.")
        print(f"(RMS {rms:.5f} < threshold {ENERGY_THRESHOLD})")
        return

    prediction, confidence, second_prediction, second_confidence = \
        predict_from_audio(audio)

    print("\nResult")
    print("-" * 40)
    print(f"Prediction : {prediction}")
    print(f"Confidence : {confidence:.2f}")
    print(f"Second     : {second_prediction} ({second_confidence:.2f})")
    print("-" * 40)


# ============================================================
# Main interactive loop
# ============================================================

print("\n" + "=" * 60)
print("INTERACTIVE LIVE KEYWORD SPOTTING")
print("=" * 60)
print("Commands:", ", ".join(COMMANDS))
print("")
print("Press ENTER -> listen for 1 second -> predict -> wait.")
print("Type q + ENTER -> quit.")
print("=" * 60)

while True:

    try:
        command = input("\nPress ENTER to speak (q to quit): ")

        if command.strip().lower() == "q":
            print("Exiting.")
            break

        listen_once()

    except KeyboardInterrupt:
        print("\nExiting.")
        break

    except Exception as e:
        print("\nError:", e)
        print("Check microphone device 1 and try again.")
