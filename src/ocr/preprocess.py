import cv2
import numpy as np

import cv2, numpy as np

def _to_binary_inv(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ako je pozadina svetla → invertuj da mastilo bude belo
    if th.mean() > 127:
        th = 255 - th
    return th

def _tight_bbox(bin_img: np.ndarray):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def _center_mass(img28: np.ndarray) -> np.ndarray:
    # img28: 28x28, mastilo belo (255), pozadina crna
    m = cv2.moments(img28)
    if abs(m["m00"]) < 1e-3:
        return img28
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(img28, M, (28, 28), flags=cv2.INTER_NEAREST, borderValue=0)


def prepare_image(image_path: str):
    """
    Pretvori proizvoljnu sliku cifre u MNIST format:
    - grayscale -> binarizacija (bela cifra na crnom)
    - crop na bounding box
    - skaliranje veće dimenzije na 20px, uz očuvanje proporcije
    - pad na 28x28
    - centriranje po težištu
    - normalizacija [0,1] i reshape (1,28,28,1)
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)

    # 1) binarizacija i inverzija po potrebi
    th = _to_binary_inv(gray)

    # 2) ukloni šum i tanko zgusni linije
    th = cv2.medianBlur(th, 3)
    th = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=1)

    # 3) crop na tesan bbox
    bbox = _tight_bbox(th)
    if bbox is None:
        raise ValueError("Nema vidljive cifre.")
    x1,y1,x2,y2 = bbox
    digit = th[y1:y2+1, x1:x2+1]

    # 4) skaliraj veću dimenziju na 20 px (MNIST stil)
    h, w = digit.shape
    if h > w:
        new_h, new_w = 20, int(round(w * (20.0 / h)))
    else:
        new_w, new_h = 20, int(round(h * (20.0 / w)))
    digit20 = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5) pad na 28×28 (centar)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = digit20

    # 6) fino centriranje po težištu
    canvas = _center_mass(canvas)

    # 7) u [0,1] i (1,28,28,1)
    x = canvas.astype("float32") / 255.0
    x = np.expand_dims(x, (-1, 0))
    return x
