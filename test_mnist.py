from src.ocr.data.mnist import load_mnist

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()

print(f"Train shape: {x_train.shape}, labels: {y_train.shape}")
print(f"Validation shape: {x_val.shape}, labels: {y_val.shape}")
print(f"Test shape: {x_test.shape}, labels: {y_test.shape}")
print(f"Pixel value range: min={x_train.min()}, max={x_train.max()}")

