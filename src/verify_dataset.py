from pathlib import Path
import wave
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
]


def inspect_wav(file_path):
    with wave.open(str(file_path), "rb") as wav:
        return {
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "samples": wav.getnframes(),
            "duration": wav.getnframes() / wav.getframerate(),
        }


def main():

    print("=" * 60)
    print("VOICE KEYWORD SPOTTING DATASET VERIFICATION")
    print("=" * 60)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    print(f"\nDataset directory:\n{DATASET_DIR}\n")

    total_files = 0
    corrupted_files = []
    stats = defaultdict(list)

    for command in TARGET_COMMANDS:

        command_dir = DATASET_DIR / command

        if not command_dir.exists():
            print(f"[ERROR] Missing folder: {command}")
            continue

        wav_files = list(command_dir.glob("*.wav"))

        print(f"{command:>5}: {len(wav_files):>5} files")

        total_files += len(wav_files)

        for wav_file in wav_files:

            try:
                info = inspect_wav(wav_file)

                stats["sample_rate"].append(info["sample_rate"])
                stats["channels"].append(info["channels"])
                stats["samples"].append(info["samples"])
                stats["duration"].append(info["duration"])

            except Exception as e:
                corrupted_files.append((wav_file, str(e)))

    print("\n" + "-" * 60)
    print("SUMMARY")
    print("-" * 60)

    print(f"Total WAV files : {total_files}")

    if stats["sample_rate"]:
        print(
            f"Sample rates   : "
            f"{sorted(set(stats['sample_rate']))}"
        )

        print(
            f"Channels       : "
            f"{sorted(set(stats['channels']))}"
        )

        print(
            f"Sample counts  : "
            f"min={min(stats['samples'])}, "
            f"max={max(stats['samples'])}"
        )

        print(
            f"Duration       : "
            f"min={min(stats['duration']):.3f}s, "
            f"max={max(stats['duration']):.3f}s"
        )

    print(f"Corrupted files: {len(corrupted_files)}")

    if corrupted_files:
        print("\nCorrupted files:")
        for file_path, error in corrupted_files[:20]:
            print(f"  {file_path}")
            print(f"    {error}")

    print("\nVerification complete.")


if __name__ == "__main__":
    main()