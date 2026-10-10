import os
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000

FRAME_LENGTH = 640
FRAME_STEP = 320
FFT_LENGTH = 1024

NUM_MEL_BINS = 64
LOWER_HZ = 80.0
UPPER_HZ = 7600.0

EPSILON = 1e-6

COMMANDS = [
    "down",
    "left",
    "no",
    "off",
    "on",
    "right",
    "up",
    "yes"
]


# ============================================================
# Preprocessing functions
# ============================================================

def load_audio(file_path):
    """
    Load a WAV file as a fixed-length mono waveform.
    """

    audio_binary = tf.io.read_file(file_path)

    waveform, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=NUM_SAMPLES
    )

    waveform = tf.squeeze(waveform, axis=-1)

    # The dataset has already been verified to contain
    # 16 kHz audio.
    tf.debugging.assert_equal(
        sample_rate,
        SAMPLE_RATE,
        message="Audio sample rate must be 16 kHz."
    )

    return waveform


def waveform_to_log_mel(waveform):
    """
    Convert waveform into a normalized log-Mel spectrogram.
    """

    # --------------------------------------------------------
    # 1. Short-Time Fourier Transform
    # --------------------------------------------------------

    stft = tf.signal.stft(
        waveform,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        pad_end=False
    )

    # Magnitude
    magnitude = tf.abs(stft)

    # Power spectrogram
    power_spectrogram = tf.square(magnitude)

    # --------------------------------------------------------
    # 2. Convert frequency bins to Mel bins
    # --------------------------------------------------------

    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MEL_BINS,
        num_spectrogram_bins=(FFT_LENGTH // 2) + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=LOWER_HZ,
        upper_edge_hertz=UPPER_HZ
    )

    mel_spectrogram = tf.matmul(
        power_spectrogram,
        mel_matrix
    )

    # --------------------------------------------------------
    # 3. Log transformation
    # --------------------------------------------------------

    log_mel = tf.math.log(
        mel_spectrogram + EPSILON
    )

    # --------------------------------------------------------
    # 4. Per-utterance normalization
    # --------------------------------------------------------

    mean = tf.reduce_mean(log_mel)
    std = tf.math.reduce_std(log_mel)

    normalized_log_mel = (
        log_mel - mean
    ) / (std + EPSILON)

    # --------------------------------------------------------
    # 5. Add CNN channel dimension
    # --------------------------------------------------------

    normalized_log_mel = tf.expand_dims(
        normalized_log_mel,
        axis=-1
    )

    return normalized_log_mel


def preprocess_audio(file_path, label):
    """
    Complete preprocessing pipeline.

    Input:
        WAV file path + integer label

    Output:
        Log-Mel spectrogram + label
    """

    waveform = load_audio(file_path)

    features = waveform_to_log_mel(waveform)

    return features, label


# ============================================================
# Dataset creation
# ============================================================

def get_file_paths_and_labels(split):
    """
    Get file paths and integer labels for a dataset split.
    """

    base_dir = os.path.join(
        "data",
        "split",
        split
    )

    file_paths = []
    labels = []

    for label, command in enumerate(COMMANDS):

        command_dir = os.path.join(
            base_dir,
            command
        )

        if not os.path.exists(command_dir):
            raise FileNotFoundError(
                f"Directory not found: {command_dir}"
            )

        for filename in sorted(os.listdir(command_dir)):

            if filename.lower().endswith(".wav"):

                file_paths.append(
                    os.path.join(
                        command_dir,
                        filename
                    )
                )

                labels.append(label)

    return file_paths, labels


def build_dataset(
    split,
    batch_size=64,
    training=False
):
    """
    Build a TensorFlow dataset.

    split:
        train / validation / test

    training:
        If True, shuffle the dataset.
    """

    file_paths, labels = get_file_paths_and_labels(split)

    print(
        f"{split.capitalize()} files: "
        f"{len(file_paths)}"
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            file_paths,
            labels
        )
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        preprocess_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# Test the complete pipeline
# ============================================================

def main():

    print("=" * 60)
    print("Testing complete preprocessing pipeline")
    print("=" * 60)

    train_dataset = build_dataset(
        "train",
        batch_size=64,
        training=True
    )

    validation_dataset = build_dataset(
        "validation",
        batch_size=64,
        training=False
    )

    test_dataset = build_dataset(
        "test",
        batch_size=64,
        training=False
    )

    # --------------------------------------------------------
    # Inspect one training batch
    # --------------------------------------------------------

    features, labels = next(
        iter(train_dataset)
    )

    print("\nTraining batch:")
    print("Feature shape :", features.shape)
    print("Feature dtype :", features.dtype)
    print("Label shape   :", labels.shape)
    print("Label dtype   :", labels.dtype)

    print(
        "Feature range :",
        float(tf.reduce_min(features)),
        "to",
        float(tf.reduce_max(features))
    )

    print("\nExpected:")
    print("Feature shape : (64, 49, 64, 1)")
    print("Number classes:", len(COMMANDS))

    # --------------------------------------------------------
    # Check validation
    # --------------------------------------------------------

    val_features, val_labels = next(
        iter(validation_dataset)
    )

    print("\nValidation batch:")
    print("Feature shape :", val_features.shape)
    print("Label shape   :", val_labels.shape)

    # --------------------------------------------------------
    # Check test
    # --------------------------------------------------------

    test_features, test_labels = next(
        iter(test_dataset)
    )

    print("\nTest batch:")
    print("Feature shape :", test_features.shape)
    print("Label shape   :", test_labels.shape)

    print("\nClass mapping:")

    for index, command in enumerate(COMMANDS):
        print(f"{index}: {command}")

    print("\nPreprocessing pipeline test successful.")


if __name__ == "__main__":
    main()