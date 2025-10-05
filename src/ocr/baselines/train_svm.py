import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from src.ocr.data.mnist import load_mnist
import os, joblib

def main():
    (x_tr, y_tr), (x_val, y_val), (x_te, y_te) = load_mnist()
    # flatten 28x28 -> 784
    Xtr = x_tr.reshape(len(x_tr), -1)
    Xte = x_te.reshape(len(x_te), -1)

    clf = LinearSVC(random_state=42)
    clf.fit(Xtr, y_tr)
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/baseline_svm.joblib")

    y_pred = clf.predict(Xte)
    rep = classification_report(y_te, y_pred, digits=4)
    cm  = confusion_matrix(y_te, y_pred)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/baseline_svm_report.txt", "w", encoding="utf-8") as f:
        f.write(rep)
    np.savetxt("artifacts/baseline_svm_cm.csv", cm, fmt="%d", delimiter=",")
    print(rep)

if __name__ == "__main__":
    main()
