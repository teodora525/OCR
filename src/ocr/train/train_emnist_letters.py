# src/ocr/train/train_emnist_letters.py

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from ..models.cnn import build_letters_cnn  # koristi tvoj CNN za slova

# ===== putanje (kao i do sada) =====
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ARTIFACTS = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS, "letters_cnn.h5")

# ===== konstante =====
NUM_CLASSES = 26
BATCH_SIZE = 256
EPOCHS = 20
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

def load_emnist_letters_tfds(batch_size=BATCH_SIZE, seed=SEED):
    """
    Učitava EMNIST Letters preko tensorflow_datasets.
    Ne treba ručno skidanje fajlova – TFDS sve odradi i kešira.
    Vraća: ds_train, ds_val, ds_test (tf.data.Dataset)
    """
    ds_train_full, ds_test = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,   # (image, label)
        with_info=False
    )

    def _prep(image, label):
        # EMNIST letters: labele su 1..26 → pomeri na 0..25 i one-hot
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, -1)            # (28,28,1)
        label = tf.one_hot(label - 1, NUM_CLASSES)   # (26,)
        return image, label

    # podela train na train/val (90/10). Shuffle pre split-a da ne bude sistematske podele.
    AUTOTUNE = tf.data.AUTOTUNE
    train_size = tf.data.experimental.cardinality(ds_train_full).numpy()
    val_count = max(1, int(0.1 * train_size))

    ds_train_full = ds_train_full.shuffle(10000, seed=seed, reshuffle_each_iteration=False)

    ds_val = ds_train_full.take(val_count).map(_prep, num_parallel_calls=AUTOTUNE).batch(batch_size).cache().prefetch(AUTOTUNE)
    ds_train = ds_train_full.skip(val_count).map(_prep, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)
    ds_test  = ds_test.map(_prep, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test


def train():
    ds_train, ds_val, ds_test = load_emnist_letters_tfds()

    # tvoj model za slova (28x28x1, 26 klasa)
    model = build_letters_cnn(input_shape=(28, 28, 1), num_classes=NUM_CLASSES)

    cbs = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=2
    )

    loss, acc = model.evaluate(ds_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    model.save(MODEL_PATH)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    train()
