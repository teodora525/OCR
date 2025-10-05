import os, csv
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("artifacts", "training_log.csv")
OUT_ACC  = os.path.join("artifacts", "accuracy_curve.png")
OUT_LOSS = os.path.join("artifacts", "loss_curve.png")

def read_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError("CSV je prazan.")
    return rows

def get_col(rows, names):
    """Vrati prvu postojeću kolonu iz liste mogućih imena."""
    for n in names:
        if n in rows[0]:
            vals = []
            for row in rows:
                v = row.get(n)
                if v is None or v == "":
                    continue
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
            if vals:
                return vals, n
    return None, None

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Nema {CSV_PATH}. Pokreni trening da se CSV generiše.")
    os.makedirs("artifacts", exist_ok=True)

    rows = read_rows(CSV_PATH)

    # Probaj više varijanti naziva koje Keras generiše
    acc, acc_name = get_col(rows, ["accuracy", "acc"])
    val_acc, val_acc_name = get_col(rows, ["val_accuracy", "val_acc"])
    loss, loss_name = get_col(rows, ["loss"])
    val_loss, val_loss_name = get_col(rows, ["val_loss"])
    lr, lr_name = get_col(rows, ["learning_rate", "lr"])

    # --- Accuracy plot ---
    plt.figure()
    if acc:     plt.plot(acc, label=acc_name)
    if val_acc: plt.plot(val_acc, label=val_acc_name)
    plt.title("Accuracy per epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    if acc or val_acc: plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_ACC, dpi=150)
    plt.close()

    # --- Loss plot ---
    plt.figure()
    if loss:     plt.plot(loss, label=loss_name)
    if val_loss: plt.plot(val_loss, label=val_loss_name)
    plt.title("Loss per epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if loss or val_loss: plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_LOSS, dpi=150)
    plt.close()

    # kratki ispis poslednjih vrednosti
    def last(v): return f"{v[-1]:.4f}" if v else "n/a"
    print(f"Saved: {OUT_ACC}, {OUT_LOSS}")
    print(f"Final metrics -> acc:{last(acc)} | val_acc:{last(val_acc)} | loss:{last(loss)} | val_loss:{last(val_loss)} | lr:{last(lr)}")

if __name__ == "__main__":
    main()
