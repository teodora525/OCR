from __future__ import annotations
import numpy as np
from tensorflow.keras.datasets import mnist
from typing import Tuple

def load_mnist(normalize: bool = True, expand_dims: bool = True,
               val_split: float = 0.1, seed: int = 42
               ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                          Tuple[np.ndarray, np.ndarray],
                          Tuple[np.ndarray, np.ndarray]]:
    """
    Učitava MNIST, vraća (x_train, y_train), (x_val, y_val), (x_test, y_test).
    - normalize: skalira piksele u [0,1]
    - expand_dims: dodaje kanal (H, W) -> (H, W, 1)
    - val_split: procenat train skupa koji ide u validaciju
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    if normalize:
        x_train /= 255.0
        x_test /= 255.0

    if expand_dims:
        x_train = np.expand_dims(x_train, -1)
        x_test  = np.expand_dims(x_test, -1)

    # kreiramo validation split
    rng = np.random.default_rng(seed)
    idx = np.arange(x_train.shape[0])
    rng.shuffle(idx)

    val_size = int(len(idx) * val_split)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    x_val, y_val = x_train[val_idx], y_train[val_idx]
    x_train, y_train = x_train[train_idx], y_train[train_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
