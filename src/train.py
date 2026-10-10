import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess import build_dataset, COMMANDS


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

MODEL_DIR = "models"
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "keyword_cnn_fp32.keras"
)


# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# Create directories
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# Build CNN
# ============================================================

def build_model():

    model = tf.keras.Sequential([
        
        tf.keras.layers.Input(
            shape=(49, 64, 1)
        ),

        # Feature extraction
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Dropout(0.25),

        # Classification
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        # 8 keyword classes
        tf.keras.layers.Dense(
            len(COMMANDS),
            activation=None
        )
    ])

    return model


# ============================================================
# Plot training history
# ============================================================

def plot_training_history(history):

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    accuracy_path = os.path.join(
        FIGURES_DIR,
        "training_accuracy.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    loss_path = os.path.join(
        FIGURES_DIR,
        "training_loss.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTraining plots saved:")
    print(accuracy_path)
    print(loss_path)


# ============================================================
# Save training history
# ============================================================

def save_history(history):

    history_path = os.path.join(
        METRICS_DIR,
        "training_history.json"
    )

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print("Training history saved:")
    print(history_path)


# ============================================================
# Main training procedure
# ============================================================

def main():

    print("=" * 60)
    print("FP32 CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = build_dataset(
        "train",
        batch_size=BATCH_SIZE,
        training=True
    )

    validation_dataset = build_dataset(
        "validation",
        batch_size=BATCH_SIZE,
        training=False
    )

    test_dataset = build_dataset(
        "test",
        batch_size=BATCH_SIZE,
        training=False
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    print("\nModel architecture:")
    model.summary()

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),

        metrics=[
            "accuracy"
        ]
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        # Save the model with the best validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        # Stop if validation accuracy stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when validation performance
        # stops improving
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Evaluate test set
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print("\nFP32 Test Results:")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model.save(BEST_MODEL_PATH)

    print("\nBest FP32 model saved:")
    print(BEST_MODEL_PATH)

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    plot_training_history(history)
    save_history(history)

    # --------------------------------------------------------
    # Save model architecture
    # --------------------------------------------------------

    architecture_path = os.path.join(
        METRICS_DIR,
        "model_summary.txt"
    )

    with open(
        architecture_path,
        "w",
        encoding="utf-8"
    ) as file:

        model.summary(
            print_fn=lambda line: file.write(line + "\n")
        )

    print("Model summary saved:")
    print(architecture_path)

    print("\n" + "=" * 60)
    print("FP32 TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()