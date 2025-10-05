# from src.ocr.data.mnist import load_mnist

# (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()

# print(f"Train shape: {x_train.shape}, labels: {y_train.shape}")
# print(f"Validation shape: {x_val.shape}, labels: {y_val.shape}")
# print(f"Test shape: {x_test.shape}, labels: {y_test.shape}")
# print(f"Pixel value range: min={x_train.min()}, max={x_train.max()}")

from src.ocr.data.mnist import load_mnist

def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()

    print("Train:", x_train.shape, y_train.shape)
    print("Val  :", x_val.shape, y_val.shape)
    print("Test :", x_test.shape, y_test.shape)

    # Opseg piksela (treba 0.0 do 1.0)
    print(f"Pixel range (train): min={x_train.min():.3f}, max={x_train.max():.3f}")

    # Brzi sanity check labela
    print(f"Labels train unique: {sorted(set(y_train.tolist()))}")

if __name__ == "__main__":
    main()
