import tarfile
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw" / "speech_commands_v0.02.tar.gz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Required keywords
TARGET_COMMANDS = {
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
}


def main():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset archive not found:\n{ARCHIVE_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Opening dataset archive...")
    print(f"Archive: {ARCHIVE_PATH}")
    print()
    print("Extracting only these commands:")
    for command in sorted(TARGET_COMMANDS):
        print(f"  - {command}")

    extracted = {command: 0 for command in TARGET_COMMANDS}

    with tarfile.open(ARCHIVE_PATH, mode="r:gz") as tar:
        members = tar.getmembers()

        for member in members:
            if not member.isfile():
                continue

            parts = Path(member.name).parts

            if len(parts) < 2:
                continue

            command = parts[0]

            if command not in TARGET_COMMANDS:
                continue

            if not member.name.lower().endswith(".wav"):
                continue

            # Extract into data/processed/<command>/
            destination = OUTPUT_DIR / command / Path(*parts[1:])

            destination.parent.mkdir(parents=True, exist_ok=True)

            with tar.extractfile(member) as source:
                if source is None:
                    continue

                with open(destination, "wb") as target:
                    target.write(source.read())

            extracted[command] += 1

            total = sum(extracted.values())

            if total % 1000 == 0:
                print(f"Extracted {total} audio files...")

    print("\nExtraction complete!")
    print("\nFiles extracted per command:")

    total_files = 0

    for command in sorted(TARGET_COMMANDS):
        count = extracted[command]
        total_files += count
        print(f"{command:>5}: {count}")

    print(f"\nTotal WAV files: {total_files}")
    print(f"Dataset location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()