import os
import shutil
import zipfile
import urllib.request

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATASET_DIR = os.path.join(RAW_DIR, "speech_commands")

# Speech Commands v0.02
DATASET_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/speech_commands_v0.02.tar.gz"
)

ARCHIVE_PATH = os.path.join(RAW_DIR, "speech_commands_v0.02.tar.gz")

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


# --------------------------------------------------
# Download dataset
# --------------------------------------------------

def download_dataset():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(ARCHIVE_PATH):
        print("Dataset archive already exists.")
        return

    print("Downloading Speech Commands dataset...")
    print("This is a large download, so please be patient.")

    urllib.request.urlretrieve(
        DATASET_URL,
        ARCHIVE_PATH,
        reporthook=download_progress
    )

    print("\nDownload completed.")


def download_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return

    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)

    print(
        f"\rDownloaded: {percent:.1f}%",
        end=""
    )


# --------------------------------------------------
# Extract required commands
# --------------------------------------------------

def extract_required_commands():
    print("\n\nExtracting required command folders...")

    os.makedirs(DATASET_DIR, exist_ok=True)

    # The archive is a .tar.gz file, so Python's tarfile module
    # is used instead of zipfile.
    import tarfile

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        members = tar.getmembers()

        for command in TARGET_COMMANDS:
            prefix = f"speech_commands_v0.02/{command}/"

            command_members = [
                member
                for member in members
                if member.name.startswith(prefix)
            ]

            if not command_members:
                print(f"WARNING: Could not find '{command}'")
                continue

            # Extract each command into data/raw/speech_commands/
            tar.extractall(
                path=RAW_DIR,
                members=command_members
            )

    # The extraction creates:
    #
    # data/raw/speech_commands_v0.02/yes
    # data/raw/speech_commands_v0.02/no
    # etc.
    #
    # Move the required folders into a cleaner location.

    extracted_root = os.path.join(
        RAW_DIR,
        "speech_commands_v0.02"
    )

    for command in TARGET_COMMANDS:
        source = os.path.join(extracted_root, command)
        destination = os.path.join(DATASET_DIR, command)

        if os.path.exists(source):
            if os.path.exists(destination):
                shutil.rmtree(destination)

            shutil.move(source, destination)

    # Remove the temporary extracted directory if empty
    if os.path.exists(extracted_root):
        try:
            os.rmdir(extracted_root)
        except OSError:
            pass

    print("Extraction completed.")


# --------------------------------------------------
# Display dataset statistics
# --------------------------------------------------

def show_statistics():
    print("\nDataset statistics")
    print("=" * 40)

    total_files = 0

    for command in TARGET_COMMANDS:
        command_dir = os.path.join(DATASET_DIR, command)

        if os.path.exists(command_dir):
            files = [
                f for f in os.listdir(command_dir)
                if f.lower().endswith(".wav")
            ]

            count = len(files)
            total_files += count

            print(f"{command:>5}: {count} audio files")
        else:
            print(f"{command:>5}: NOT FOUND")

    print("-" * 40)
    print(f"Total : {total_files} audio files")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("VOICE KEYWORD SPOTTING - DATASET PREPARATION")
    print("=" * 60)

    download_dataset()
    extract_required_commands()
    show_statistics()

    print("\nDataset preparation completed.")