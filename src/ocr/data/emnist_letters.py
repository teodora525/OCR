# src/ocr/data/emnist_letters.py
import os, gzip, struct
import numpy as np
from keras.utils import to_categorical
import tensorflow_datasets as tfds
import tensorflow as tf

NUM_CLASSES = 26
IMG_H = 28
IMG_W = 28

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
