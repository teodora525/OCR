# src/ocr/eval/evaluate_letters.py
import os, numpy as np
from tensorflow import keras
from ..data.emnist_letters import load_emnist_letters, NUM_CLASSES

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS = os.path.join(PROJECT_ROOT, "..", "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS, "letters_cnn.h5")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def main():
    (_, _, _), (x_test, y_test_raw, y_test) = load_emnist_letters(DATA_DIR)
    model = keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Letters test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
