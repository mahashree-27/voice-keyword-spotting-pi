from pathlib import Path
import tensorflow as tf

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed"

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

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000


# --------------------------------------------------
# Find all audio files
# --------------------------------------------------

def get_audio_files():

    audio_files = []
    labels = []

    for label, command in enumerate(COMMANDS):

        command_dir = DATASET_DIR / command

        files = sorted(command_dir.glob("*.wav"))

        for file in files:
            audio_files.append(str(file))
            labels.append(label)

    return audio_files, labels


# --------------------------------------------------
# Load one WAV file
# --------------------------------------------------

def load_audio(file_path, label):

    audio_binary = tf.io.read_file(file_path)

    audio, sample_rate = tf.audio.decode_wav(
        audio_binary,
        desired_channels=1,
        desired_samples=NUM_SAMPLES
    )

    audio = tf.squeeze(audio, axis=-1)

    # Convert from int-style WAV range to float
    audio = tf.cast(audio, tf.float32)

    return audio, label


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("LOADING VOICE KEYWORD DATASET")
    print("=" * 60)

    audio_files, labels = get_audio_files()

    print(f"\nTotal audio files: {len(audio_files)}")
    print(f"Number of classes: {len(COMMANDS)}")

    print("\nClass mapping:")

    for index, command in enumerate(COMMANDS):
        count = labels.count(index)
        print(f"{index}: {command:>5} -> {count} files")

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (audio_files, labels)
    )

    dataset = dataset.map(
        load_audio,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    print("\nDataset created successfully.")

    # Inspect first sample
    audio, label = next(iter(dataset))

    print("\nFirst sample:")
    print(f"Audio shape : {audio.shape}")
    print(f"Audio dtype : {audio.dtype}")
    print(f"Label       : {label.numpy()}")
    print(f"Keyword     : {COMMANDS[label.numpy()]}")

    print("\nFirst 10 audio samples:")
    print(audio[:10].numpy())

    print("\nDataset loading test successful.")


if __name__ == "__main__":
    main()