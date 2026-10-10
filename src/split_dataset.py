from pathlib import Path
import random
import shutil

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "split"

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

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("CREATING TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 60)

    random.seed(RANDOM_SEED)

    # Check ratios
    assert abs(
        TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
    ) < 1e-6

    # Create output directories
    for split in ["train", "validation", "test"]:
        for command in COMMANDS:
            (SPLIT_DIR / split / command).mkdir(
                parents=True,
                exist_ok=True
            )

    total_train = 0
    total_validation = 0
    total_test = 0

    print("\nSplitting files...\n")

    for command in COMMANDS:

        source_command_dir = SOURCE_DIR / command

        files = list(source_command_dir.glob("*.wav"))

        # Shuffle deterministically
        random.shuffle(files)

        total = len(files)

        train_count = int(total * TRAIN_RATIO)
        validation_count = int(total * VALIDATION_RATIO)

        train_files = files[:train_count]

        validation_files = files[
            train_count:
            train_count + validation_count
        ]

        test_files = files[
            train_count + validation_count:
        ]

        # Copy files
        for file in train_files:
            shutil.copy2(
                file,
                SPLIT_DIR / "train" / command / file.name
            )

        for file in validation_files:
            shutil.copy2(
                file,
                SPLIT_DIR / "validation" / command / file.name
            )

        for file in test_files:
            shutil.copy2(
                file,
                SPLIT_DIR / "test" / command / file.name
            )

        total_train += len(train_files)
        total_validation += len(validation_files)
        total_test += len(test_files)

        print(
            f"{command:>5}: "
            f"train={len(train_files):4d}, "
            f"validation={len(validation_files):4d}, "
            f"test={len(test_files):4d}"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    total_files = total_train + total_validation + total_test

    print("\n" + "-" * 60)
    print("SPLIT SUMMARY")
    print("-" * 60)

    print(f"Training   : {total_train}")
    print(f"Validation : {total_validation}")
    print(f"Testing    : {total_test}")
    print(f"Total      : {total_files}")

    print("\nActual percentages:")

    print(
        f"Training   : "
        f"{100 * total_train / total_files:.2f}%"
    )

    print(
        f"Validation : "
        f"{100 * total_validation / total_files:.2f}%"
    )

    print(
        f"Testing    : "
        f"{100 * total_test / total_files:.2f}%"
    )

    print(f"\nRandom seed: {RANDOM_SEED}")

    print(f"\nSplit dataset location:")
    print(SPLIT_DIR)

    print("\nDataset split completed successfully.")


if __name__ == "__main__":
    main()