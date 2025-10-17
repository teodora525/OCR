import os
import matplotlib.pyplot as plt
from src.ocr.data.mnist import load_mnist

def main():
    os.makedirs("artifacts", exist_ok=True)
    (x_train, y_train), _, _ = load_mnist()

    # uzmi prvih 10 primera
    images = x_train[:10]
    labels = y_train[:10]

    # nacrtaj 2x5 grid
    plt.figure(figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("artifacts/sample_images.png", dpi=150)
    plt.close()
    print("Sačuvano: artifacts/sample_images.png")

if __name__ == "__main__":
    main()
