import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.ocr.data.mnist import load_mnist

def main():
    # 1) Učitavanje test skupa
    (_, _), (_, _), (x_test, y_test) = load_mnist()

    # 2) Učitavanje modela
    model = tf.keras.models.load_model("models/mnist_cnn.keras")

    # 3) Predikcija
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # 4) Matrica konfuzije i izveštaj
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    os.makedirs("artifacts", exist_ok=True)
    np.savetxt("artifacts/confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    with open("artifacts/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # 5) Vizuelizacija matrice konfuzije
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=150)
    plt.close()

    # 6) Prikaz nekoliko pogrešno klasifikovanih primera
    wrong_idx = np.where(y_pred != y_test)[0]
    sample_idx = wrong_idx[:25]  # prvih 25 grešaka

    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(sample_idx):
        plt.subplot(5, 5, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap='gray')
        plt.title(f"T:{y_test[idx]} P:{y_pred[idx]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("artifacts/misclassified_samples.png", dpi=150)
    plt.close()

    print("Classification report:\n")
    print(report)
    print("\nMatrica konfuzije i izveštaj sačuvani su u artifacts \n")

if __name__ == "__main__":
    main()
