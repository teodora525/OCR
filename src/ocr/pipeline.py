import os
import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from functools import lru_cache
from tensorflow import keras
from pathlib import Path 

from src.ocr.preprocess import _to_binary_inv, _tight_bbox, _center_mass

# ====== PATHS ======
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent.parent
MODELS = PROJECT_ROOT / "models"
ARTIFACTS = PROJECT_ROOT / "artifacts"

# VAŽNO: ime fajla za cifarski model da bude isto kao u Streamlit-u
MODEL_PATH = str(MODELS / "mnist_cnn.keras")             # npr. OCR/models/mnist.keras
LETTERS_MODEL_PATH = str(ARTIFACTS / "letters_cnn.h5")

# ====== DIGITS  ======
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
    """Vrati listu bbox-ova (x, y, w, h) s leva nadesno za sve cifre/slova na slici."""
    th = _to_binary_inv(gray_img)
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)  # blago zadebljanje

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > 100]  # odbaci sitno
    boxes.sort(key=lambda b: b[0])  # s leva nadesno
    return boxes


def predict_line_from_array(img_bgr: np.ndarray) -> List[Tuple[int | str, float]]:
    """Primaj BGR. Vraća [(label, conf), ...] za svaku segmentisanu cifru (ili '?' ako ne uspe)."""
    if img_bgr is None:
        return []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ako trenutno "radimo kao da nema cifara", slobodno preskoči load ispod.
    model = tf.keras.models.load_model(MODEL_PATH)

    preds: List[Tuple[int | str, float]] = []
    for (x, y, w, h) in _segment_boxes(gray):
        roi = gray[y:y+h, x:x+w]
        xin = _preprocess_digit_roi(roi)
        if xin is None:
            preds.append(("?", 0.0))
            continue
        prob = model.predict(xin, verbose=0)[0]
        lab = int(np.argmax(prob))
        conf = float(prob[lab])
        preds.append((lab, conf))
    return preds


def predict_line_string_from_array(img_bgr: np.ndarray) -> str:
    items = predict_line_from_array(img_bgr)
    return "".join(str(l) if isinstance(l, int) else "?" for (l, _) in items)


# ====== LETTERS (novi deo) ======
@lru_cache(maxsize=1)
def _get_letters_model():
    if not os.path.exists(LETTERS_MODEL_PATH):
        raise FileNotFoundError(f"Letters model not found: {LETTERS_MODEL_PATH}")
    return keras.models.load_model(LETTERS_MODEL_PATH)


def _prep_28x28(gray28: np.ndarray) -> np.ndarray:
    """
    Ulaz: (28,28) gray [0..255] ili [0..1]. Izlaz: float32 (28,28,1) u [0,1].
    Ako je pozadina svetla, invertuj.
    """
    g = gray28.astype("float32")
    if g.max() > 1.5:  # pretpostavi 0..255
        g /= 255.0
    # heuristika za invert
    if g.mean() > 0.5:
        g = 1.0 - g
    g = g[..., None]
    return g


def predict_letter28x28(gray28: np.ndarray) -> Tuple[str, float]:
    """Prima (28,28) ili (28,28,1) sliku slova; vraća (letter, confidence)."""
    model = _get_letters_model()
    if gray28.ndim == 3 and gray28.shape[-1] == 1:
        x = gray28.astype("float32")
        if x.max() > 1.5:
            x /= 255.0
    else:
        x = _prep_28x28(gray28)
    probs = model.predict(x[None, ...], verbose=0)[0]
    cls = int(np.argmax(probs))
    conf = float(np.max(probs))
    letter = chr(ord('a') + cls)
    return letter, conf


def predict_letters_from_array(img_bgr: np.ndarray,
                               boxes: List[Tuple[int, int, int, int]] | None = None
                               ) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """
    img_bgr: originalna slika (BGR, OpenCV).
    boxes: opcioni list [(x,y,w,h), ...] ako već imaš segmentaciju.
    vraća: list[(letter, conf, (x,y,w,h))]
    """
    if boxes is None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean() > 127:
            th = 255 - th

        # spoji sitne prekide pa malo zadebljaj potez
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
        k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.dilate(th, k_dil, iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        H, W = gray.shape[:2]
        boxes = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 20:
                continue
            pad = max(1, int(0.15 * max(w, h)))
            x0 = max(0, x - pad); y0 = max(0, y - pad)
            x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
            boxes.append((x0, y0, x1 - x0, y1 - y0))
        boxes.sort(key=lambda b: b[0])

    out: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    for (x, y, w, h) in boxes:
        roi = img_bgr[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # centriranje u 28x28 (letterboxing)
        scale = 20.0 / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        xs = (28 - nw) // 2
        ys = (28 - nh) // 2
        canvas[ys:ys+nh, xs:xs+nw] = small
        canvas = _center_mass(canvas)

        letter, conf = predict_letter28x28(canvas)
        out.append((letter, conf, (x, y, w, h)))
    return out

def predict_text_from_array(img_bgr: np.ndarray,
                            digit_strict: float = 0.97,
                            digit_lo: float = 0.85,
                            letter_min: float = 0.45,
                            margin: float = 0.05):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:
        th = 255 - th

    # isti tretman kao za letters: close + dilate
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k_close, iterations=1)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.dilate(th, k_dil, iterations=1)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

    digit_model = tf.keras.models.load_model(MODEL_PATH)

    ambiguous = {('o', 0), ('i', 1), ('z', 2), ('s', 5), ('b', 8), ('g', 6), ('j', 3)}

    items = []
    H, W = gray.shape[:2]
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        # padding bbox-a
        pad = max(1, int(0.15 * max(w, h)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)

        roi = gray[y0:y1, x0:x1]

        # 28x28
        Hc, Wc = roi.shape[:2]
        scale = 20.0 / max(Wc, Hc)
        nw, nh = max(1, int(Wc * scale)), max(1, int(Hc * scale))
        small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((28, 28), dtype=np.uint8)
        xs, ys = (28 - nw)//2, (28 - nh)//2
        canvas[ys:ys+nh, xs:xs+nw] = small
        canvas = _center_mass(canvas)

        # digit
        xd = (canvas.astype("float32") / 255.0)[..., None][None, ...]
        pd = digit_model.predict(xd, verbose=0)[0]
        d_lab = int(np.argmax(pd)); d_conf = float(pd[d_lab])

        # letter
        l_letter, l_conf = predict_letter28x28(canvas)

        # odluka (blago favorizuj slovo)
        if (l_letter, d_lab) in ambiguous and l_conf >= 0.50 and d_conf < 0.997:
            sym, conf = l_letter, l_conf
        elif d_conf >= digit_strict and d_conf >= (l_conf + 0.12):
            sym, conf = str(d_lab), d_conf
        elif l_conf >= letter_min and (l_conf + margin) >= d_conf:
            sym, conf = l_letter, l_conf
        elif d_conf >= digit_lo:
            sym, conf = str(d_lab), d_conf
        elif l_conf >= letter_min:
            sym, conf = l_letter, l_conf
        else:
            sym, conf = "?", max(d_conf, l_conf)

        items.append((sym, conf, (x0, y0, x1 - x0, y1 - y0)))

    items.sort(key=lambda t: t[2][0])
    text = "".join(s for s, _, _ in items)
    return text, items
