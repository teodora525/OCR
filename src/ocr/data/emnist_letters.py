# src/ocr/data/emnist_letters.py
import os, gzip, struct
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import tensorflow as tf

NUM_CLASSES = 26
IMG_H = 28
IMG_W = 28

def _load_idx_images(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad magic(images): {magic}"
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data

def _load_idx_labels(path_gz):
    with gzip.open(path_gz, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad magic(labels): {magic}"
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

def _fix_emnist_orientation(imgs):
    # EMNIST je transponovan u odnosu na MNIST.
    # Tipična korekcija: transpose pa horizontalni flip.
    imgs = np.transpose(imgs, (0, 2, 1))
    imgs = np.flip(imgs, axis=2)
    return imgs


def load_emnist_letters_tfds():
    # ovo automatski skida i kešira dataset (~500MB prvi put)
    ds_train, ds_test = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True  # vraća (img, label)
    )

    def _prep(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.expand_dims(image, -1)  # (28,28,1)
        label = tf.one_hot(label - 1, 26)  # EMNIST letters labela je 1–26
        return image, label

    ds_train = ds_train.map(_prep).batch(256).shuffle(10_000)
    ds_test  = ds_test.map(_prep).batch(256)

    return ds_train, ds_test
