import os
import time
import sounddevice as sd
import soundfile as sf

# =========================
# Configuration
# =========================
SAMPLE_RATE = 16000
DURATION = 1.0
MIC_DEVICE = 1

COMMANDS = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
]

RECORDINGS_PER_COMMAND = 5
OUTPUT_DIR = "data/real_noise"


def record_one(command, recording_number):
    """Record one 1-second audio sample."""

    output_dir = os.path.join(OUTPUT_DIR, command)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{command}_{recording_number:02d}.wav"
    filepath = os.path.join(output_dir, filename)

    print()
    print(f"Recording: {command.upper()} - {recording_number}/{RECORDINGS_PER_COMMAND}")
    print("Get ready...")

    for countdown in [3, 2, 1]:
        print(countdown)
        time.sleep(1)

    print(">>> SPEAK NOW <<<")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
    )

    sd.wait()

    sf.write(filepath, audio, SAMPLE_RATE)

    print(f"Saved: {filepath}")


def main():
    print("=" * 60)
    print("REAL AMBIENT NOISE DATA COLLECTION")
    print("=" * 60)

    print(f"Microphone device: {MIC_DEVICE}")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} second")
    print(f"Commands: {len(COMMANDS)}")
    print(f"Recordings per command: {RECORDINGS_PER_COMMAND}")
    print(f"Total recordings: {len(COMMANDS) * RECORDINGS_PER_COMMAND}")
    print()

    print("IMPORTANT:")
    print("- Record in a normal environment with real background noise.")
    print("- Speak the command naturally.")
    print("- Do not deliberately create extremely loud noise.")
    print("- Stay reasonably close to the earphone microphone.")
    print()

    input("Press ENTER when you are ready to start...")

    for command in COMMANDS:
        print()
        print("=" * 60)
        print(f"COMMAND: {command.upper()}")
        print("=" * 60)

        for recording_number in range(1, RECORDINGS_PER_COMMAND + 1):
            record_one(command, recording_number)

            if not (
                command == COMMANDS[-1]
                and recording_number == RECORDINGS_PER_COMMAND
            ):
                print("Next recording in 2 seconds...")
                time.sleep(2)

    print()
    print("=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"Saved recordings to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()