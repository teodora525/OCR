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
