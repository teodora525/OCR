import tensorflow as tf
import numpy as np
from src.ocr.preprocess import prepare_image

def predict_single(image_path: str):
    model = tf.keras.models.load_model("models/mnist_cnn.keras")
    x = prepare_image(image_path)
    y_prob = model.predict(x, verbose=0)[0]
    top3_idx = np.argsort(y_prob)[::-1][:3]
    return [(int(i), float(y_prob[i])) for i in top3_idx]

if __name__ == "__main__":
    preds = predict_single("samples/cetiri.png")
    for label, prob in preds:
        print(f"Predikcija: {label}, verovatnoća: {prob:.4f}")
