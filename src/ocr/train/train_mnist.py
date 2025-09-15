import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

from src.ocr.data.mnist import load_mnist
from src.ocr.models.cnn import build_cnn

def main():
    # 1) Učitavanje podataka
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    
    # 2) Kreiranje modela
    model = build_cnn(input_shape=x_train.shape[1:], num_classes=10)
    
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # 3) Callback-ovi
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ModelCheckpoint("models/mnist_cnn.keras", monitor="val_accuracy", save_best_only=True),
        CSVLogger("artifacts/training_log.csv")
    ]
    
    # 4) Trening
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=3,           # probni trening – kasnije možeš staviti 10-20
        batch_size=128,
        callbacks=callbacks,
        verbose=2
    )
    
    # 5) Evaluacija na test skupu
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
