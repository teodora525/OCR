import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from src.ocr.preprocess import _to_binary_inv, _tight_bbox, _center_mass  # već ih imaš u preprocess.py

MODEL_PATH = "models/mnist_cnn.keras"

def _preprocess_digit_roi(roi: np.ndarray) -> np.ndarray | None:
    """Pretvori ROI jedne cifre u MNIST format (1,28,28,1). Vraća None ako ROI nema 'tinta'."""
    if roi is None:
        return None
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    th = _to_binary_inv(roi)
    bbox = _tight_bbox(th)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    digit = th[y1:y2+1, x1:x2+1]

    # skala veće dimenzije na 20 px, očuvana proporcija
    h, w = digit.shape
    if h > w:
        new_h, new_w = 20, int(round(w * (20.0 / h)))
    else:
        new_w, new_h = 20, int(round(h * (20.0 / w)))
    digit20 = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad na 28x28 i centriranje po težištu (MNIST stil)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit20
    canvas = _center_mass(canvas)

    x = canvas.astype("float32") / 255.0
    x = np.expand_dims(x, (-1, 0))  # (1,28,28,1)
    return x

def _segment_boxes(gray_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Vrati listu bbox-ova (x, y, w, h) s leva nadesno za sve cifre na slici."""
    th = _to_binary_inv(gray_img)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)  # malo zadebljaj potez

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > 100]  # odbaci sitne fleke
    boxes.sort(key=lambda b: b[0])  # s leva nadesno
    return boxes

def predict_line_from_array(img_bgr: np.ndarray) -> List[Tuple[int, float]]:
    """Primaj BGR (npr. iz cv2.imdecode). Vraća [(label, conf), ...] za svaku segmentisanu cifru."""
    if img_bgr is None:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    model = tf.keras.models.load_model(MODEL_PATH)

    preds: List[Tuple[int, float]] = []
    for (x, y, w, h) in _segment_boxes(gray):
        roi = gray[y:y+h, x:x+w]
        xin = _preprocess_digit_roi(roi)
        if xin is None:
            preds.append(("?", 0.0))  # fallback
            continue
        prob = model.predict(xin, verbose=0)[0]
        lab = int(np.argmax(prob)); conf = float(prob[lab])
        preds.append((lab, conf))
    return preds

def predict_line_string_from_array(img_bgr: np.ndarray) -> str:
    items = predict_line_from_array(img_bgr)
    return "".join(str(l) if isinstance(l, int) else "?" for (l, _) in items)
