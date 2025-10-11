# src/ocr/pipeline.py
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tensorflow import keras

from src.ocr.preprocess import _to_binary_inv, _tight_bbox, _center_mass


class OCRPipeline:
    """
    Jedinstven OCR pipeline za cifre i slova (mešovito).
    - Stabilna segmentacija sa split-ovanjem širokih boxeva i spajanjem tačke/stuba (i/j)
    - Jedinstven preprocess (MNIST/EMNIST stil): binarno, belo mastilo, 28x28, centriranje
    - Softmax normalizacija (sprečava conf > 1.0)
    - Heuristike za j↔1, c↔o/0, a↔o
    """

    # ===== PATHS =====
    FILE_DIR = Path(__file__).resolve().parent      # .../OCR/src/ocr
    PROJECT_ROOT = FILE_DIR.parent.parent           # .../OCR
    MODELS = PROJECT_ROOT / "models"
    ARTIFACTS = PROJECT_ROOT / "artifacts"

    DEFAULT_DIGIT_MODEL = str(MODELS / "mnist_cnn.keras")
    DEFAULT_LETTER_MODEL = str(ARTIFACTS / "letters_cnn.h5")

    def __init__(
        self,
        digit_model_path: Optional[str] = None,
        letter_model_path: Optional[str] = None
    ) -> None:
        self.digit_model_path = digit_model_path or self.DEFAULT_DIGIT_MODEL
        self.letter_model_path = letter_model_path or self.DEFAULT_LETTER_MODEL

        if not os.path.exists(self.digit_model_path):
            raise FileNotFoundError(f"Digit model not found: {self.digit_model_path}")
        if not os.path.exists(self.letter_model_path):
            raise FileNotFoundError(f"Letters model not found: {self.letter_model_path}")

        # cache loaderi – compile=False
        self._get_digit_model = lru_cache(maxsize=1)(self._load_digit_model)
        self._get_letters_model = lru_cache(maxsize=1)(self._load_letters_model)

    # ========= Model loading =========
    def _load_digit_model(self):
        return keras.models.load_model(self.digit_model_path, compile=False)

    def _load_letters_model(self):
        return keras.models.load_model(self.letter_model_path, compile=False)

    # ========= Math helpers =========
    @staticmethod
    def _softmax_safe(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        v = v - np.max(v)
        e = np.exp(v)
        s = e.sum() + 1e-9
        return e / s

    def _predict_digit_probs(self, xin: np.ndarray) -> np.ndarray:
        p = self._get_digit_model().predict(xin, verbose=0)[0]
        if (p.max() > 1.0) or (abs(p.sum() - 1.0) > 1e-3):
            p = self._softmax_safe(p)
        return p

    def _predict_letter_probs(self, xin: np.ndarray) -> np.ndarray:
        p = self._get_letters_model().predict(xin, verbose=0)[0]
        if (p.max() > 1.0) or (abs(p.sum() - 1.0) > 1e-3):
            p = self._softmax_safe(p)
        return p

    # ========= Preprocess =========
    @staticmethod
    def _preprocess_roi_28x28(roi: np.ndarray) -> Optional[np.ndarray]:
        """
        ROI -> binarno belo mastilo, tight crop, resize ~20px, letterbox 28x28, centroid centering.
        Output: (1,28,28,1) float32 u [0,1]
        """
        if roi is None:
            return None
        if roi.ndim == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        th = _to_binary_inv(roi)
        th = cv2.bilateralFilter(th, d=3, sigmaColor=25, sigmaSpace=25)

        bbox = _tight_bbox(th)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = th[y1:y2+1, x1:x2+1]

        h, w = crop.shape
        scale = 20.0 / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((28, 28), dtype=np.uint8)
        ys = (28 - nh) // 2
        xs = (28 - nw) // 2
        canvas[ys:ys+nh, xs:xs+nw] = small

        canvas = _center_mass(canvas)
        x = (canvas.astype("float32") / 255.0)[..., None][None, ...]
        return x

    # ========= Segmentacija =========
    @staticmethod
    def _vertical_split_if_needed(gray: np.ndarray, x, y, w, h) -> List[Tuple[int,int,int,int]]:
        """Ako je box preširok, podeli ga po glatkom projekcionom profilu."""
        roi = gray[y:y+h, x:x+w]
        th = _to_binary_inv(roi)

        col_sum = th.sum(axis=0).astype(np.float32)
        col_sum = cv2.GaussianBlur(col_sum[:, None], (9, 1), 0).ravel()
        m = col_sum.max() if col_sum.size else 1.0
        col_sum = col_sum / (m + 1e-6)

        valley = (col_sum < 0.08).astype(np.uint8)

        boxes = []
        in_gap = True
        start = 0
        for i in range(len(valley)):
            if in_gap and valley[i] == 0:
                start = i; in_gap = False
            if not in_gap and valley[i] == 1:
                end = i
                ww = end - start
                if ww >= max(3, int(0.14 * w)):
                    boxes.append((x + start, y, ww, h))
                in_gap = True
        if not in_gap:
            end = len(valley) - 1
            ww = end - start + 1
            if ww >= max(3, int(0.14 * w)):
                boxes.append((x + start, y, ww, h))

        return boxes if len(boxes) > 1 else [(x, y, w, h)]

    @staticmethod
    def _segment_symbols(gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        """Open -> dilate -> CC -> split širokih -> merge tačka+stub (i/j)."""
        th = _to_binary_inv(gray)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)
        th = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)), iterations=1)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        raw = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area < 25 or h < 6 or w < 3:
                continue
            raw.append((x, y, w, h))
        raw.sort(key=lambda b: b[0])

        # 1) split preširokih
        boxes: List[Tuple[int,int,int,int]] = []
        for (x, y, w, h) in raw:
            if w > 1.55 * h:
                boxes.extend(OCRPipeline._vertical_split_if_needed(gray, x, y, w, h))
            else:
                boxes.append((x, y, w, h))

        # 2) spoji tačku sa stubom (i/j)
        merged: List[Tuple[int,int,int,int]] = []
        skip = set()
        med_h = np.median([bh for _,_,_,bh in boxes]) if boxes else 0.0
        for i, (x1, y1, w1, h1) in enumerate(boxes):
            if i in skip:
                continue
            # mala gornja komponenta (tačka)
            if h1 < 0.65 * med_h and w1 <= int(0.9 * h1):
                cx1 = x1 + w1 // 2
                best = None
                best_score = 1e9
                for j, (x2, y2, w2, h2) in enumerate(boxes):
                    if j == i or j in skip:
                        continue
                    if y2 <= y1:
                        continue
                    cx2 = x2 + w2 // 2
                    dx = abs(cx1 - cx2)
                    dy = y2 - (y1 + h1)
                    if dy <= int(0.55 * h2) and dx <= max(5, int(0.6 * w2)):
                        if h2 >= 1.15 * med_h and (w2 < 0.85 * h2):
                            score = dx + 0.4 * dy
                            if score < best_score:
                                best_score = score
                                best = j
                if best is not None:
                    x2, y2, w2, h2 = boxes[best]
                    x0 = min(x1, x2); y0 = min(y1, y2)
                    x3 = max(x1+w1, x2+w2); y3 = max(y1+h1, y2+h2)
                    merged.append((x0, y0, x3-x0, y3-y0))
                    skip.add(best); skip.add(i)
                    continue
            if i not in skip:
                merged.append((x1, y1, w1, h1))

        merged.sort(key=lambda b: b[0])
        return merged

    # ========= Oblici/feature-i =========
    @staticmethod
    def _holes_and_solidity(bin28: np.ndarray) -> Tuple[int, float]:
        cnts, hier = cv2.findContours(bin28, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0
        if hier is not None:
            for i in range(len(cnts)):
                if hier[0][i][3] != -1:
                    holes += 1
        if not cnts:
            return holes, 1.0
        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull) if hull is not None else area
        solidity = (area / hull_area) if hull_area > 1e-3 else 1.0
        return holes, float(solidity)

    @staticmethod
    def _estimate_baseline(boxes: List[Tuple[int,int,int,int]]) -> float:
        bottoms = [y + h for (_, y, _, h) in boxes] or [0.0]
        return float(np.median(bottoms))

    @staticmethod
    def _descender_amount(y: int, h: int, baseline: float) -> float:
        bottom = y + h
        return max(0.0, (bottom - baseline) / max(1.0, h))

    # ========= DIGITS-ONLY API (backward compat) =========
    @staticmethod
    def _segment_boxes_digits(gray_img: np.ndarray) -> List[Tuple[int,int,int,int]]:
        th = _to_binary_inv(gray_img)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [b for b in boxes if b[2] * b[3] > 100]
        boxes.sort(key=lambda b: b[0])
        return boxes

    def predict_line_from_array(self, img_bgr: np.ndarray) -> List[Tuple[int | str, float]]:
        if img_bgr is None:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        preds: List[Tuple[int | str, float]] = []
        for (x, y, w, h) in self._segment_boxes_digits(gray):
            roi = gray[y:y+h, x:x+w]
            xin = self._preprocess_roi_28x28(roi)
            if xin is None:
                preds.append(("?", 0.0))
                continue
            prob = self._predict_digit_probs(xin)
            lab = int(np.argmax(prob))
            conf = float(prob[lab])
            preds.append((lab, conf))
        return preds

    # ========= MIXED (letters + digits) =========
    def predict_letter28x28(self, gray28: np.ndarray) -> Tuple[str, float]:
        if gray28.ndim == 3 and gray28.shape[-1] == 1:
            x = gray28.astype("float32")
            if x.max() > 1.5:
                x /= 255.0
        else:
            g = gray28.astype("float32")
            if g.max() > 1.5:
                g /= 255.0
            if g.mean() > 0.5:
                g = 1.0 - g
            x = g[..., None]
        probs = self._predict_letter_probs(x[None, ...])
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        letter = chr(ord('a') + cls)  # EMNIST Letters: 26 klasa (a..z)
        return letter, conf

    def predict_letters_from_array(
        self,
        img_bgr: np.ndarray,
        boxes: Optional[List[Tuple[int,int,int,int]]] = None
    ) -> List[Tuple[str, float, Tuple[int,int,int,int]]]:
        if boxes is None:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            boxes = self._segment_symbols(gray)
        out: List[Tuple[str, float, Tuple[int,int,int,int]]] = []
        for (x, y, w, h) in boxes:
            roi = img_bgr[y:y+h, x:x+w]
            xin = self._preprocess_roi_28x28(roi)
            if xin is None:
                out.append(("?", 0.0, (x, y, w, h)))
                continue
            probs = self._predict_letter_probs(xin)
            cls = int(np.argmax(probs))
            conf = float(probs[cls])
            letter = chr(ord('a') + cls)
            out.append((letter, conf, (x, y, w, h)))
        return out

    def predict_text_from_array(
        self,
        img_bgr: np.ndarray,
        digit_strict: float = 0.996,
        digit_lo: float = 0.90,
        letter_min: float = 0.58,
        margin: float = 0.10,
    ):
        if img_bgr is None:
            return "", []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._segment_symbols(gray)
        baseline = self._estimate_baseline(boxes)

        ambiguous = {('o', 0), ('i', 1), ('z', 2), ('s', 5), ('b', 8), ('g', 9), ('j', 1)}
        heights = [h for (_,_,_,h) in boxes] or [1]
        med_h = float(np.median(heights))

        items = []
        for (x, y, w, h) in boxes:
            roi_gray = gray[y:y+h, x:x+w]
            xin = self._preprocess_roi_28x28(roi_gray)
            if xin is None:
                items.append(("?", 0.0, (x, y, w, h)))
                continue

            # predikcije
            pd = self._predict_digit_probs(xin)
            d_lab = int(np.argmax(pd)); d_conf = float(pd[d_lab])

            pl = self._predict_letter_probs(xin)
            l_lab = int(np.argmax(pl)); l_conf = float(pl[l_lab])
            l_letter = chr(ord('a') + l_lab)

            # bin 28×28 za oblike
            canvas = (xin[0, ..., 0] * 255).astype(np.uint8)
            bin28 = (canvas > 127).astype(np.uint8) * 255

            holes, solidity = self._holes_and_solidity(bin28)
            aspect = w / max(1, h)
            uppercase_hint = (h >= 1.25 * med_h) or (solidity > 0.92)
            rep = self._descender_amount(y, h, baseline)

            # heuristike
            if rep > 0.12 and aspect < 0.55:   # j vs 1
                if d_lab == 1:
                    d_conf *= 0.92
                if l_letter == 'j':
                    l_conf *= 1.08

            if holes == 0 and 0.85 <= aspect <= 1.25 and solidity < 0.86:  # c vs o/0
                if l_letter == 'c':
                    l_conf *= 1.10
                if d_lab == 0:
                    d_conf *= 0.95

            bin_cols = bin28.sum(axis=0).astype(np.float32)   # a vs o
            w28 = bin_cols.shape[0]
            mid = int(w28 * 0.55); right = int(w28 * 0.85)
            mid_val = bin_cols[mid] / (bin28.shape[0] * 255 + 1e-6)
            right_val = bin_cols[right] / (bin28.shape[0] * 255 + 1e-6)
            notch = (mid_val - right_val) > 0.08
            if holes >= 1 and 0.75 <= aspect <= 1.25:
                if notch and l_letter == 'a':
                    l_conf *= 1.07
                if notch and d_lab == 0:
                    d_conf *= 0.97

            if uppercase_hint:
                l_conf *= 1.02

            # odluka
            if (l_letter, d_lab) in ambiguous and l_conf >= 0.56 and d_conf < 0.998:
                sym, conf = l_letter, l_conf
            elif l_letter == 'j' and rep > 0.12 and l_conf >= 0.52 and (l_conf + 0.05) >= d_conf:
                sym, conf = l_letter, l_conf
            elif d_conf >= digit_strict and d_conf >= (l_conf + 0.13):
                sym, conf = str(d_lab), d_conf
            elif l_conf >= letter_min and (l_conf + margin) >= d_conf:
                sym, conf = l_letter, l_conf
            elif d_conf >= digit_lo:
                sym, conf = str(d_lab), d_conf
            elif l_conf >= letter_min:
                sym, conf = l_letter, l_conf
            else:
                sym, conf = "?", max(d_conf, l_conf)

            items.append((sym, conf, (x, y, w, h)))

        items.sort(key=lambda t: t[2][0])
        text = "".join(s for s, _, _ in items)
        return text, items


# ===== Back-compat globalne funkcije (da stari importi rade) =====
__all__ = [
    "OCRPipeline",
    "predict_line_from_array",
    "predict_text_from_array",
    "predict_letters_from_array",
    "predict_letter28x28",
]

_PIPE = None
def _get_pipe():
    global _PIPE
    if _PIPE is None:
        _PIPE = OCRPipeline()  # koristi podrazumevane putanje modela
    return _PIPE

def predict_line_from_array(img_bgr):
    return _get_pipe().predict_line_from_array(img_bgr)

def predict_text_from_array(img_bgr,
                            digit_strict: float = 0.996,
                            digit_lo: float = 0.90,
                            letter_min: float = 0.58,
                            margin: float = 0.10):
    return _get_pipe().predict_text_from_array(
        img_bgr,
        digit_strict=digit_strict,
        digit_lo=digit_lo,
        letter_min=letter_min,
        margin=margin,
    )

def predict_letters_from_array(img_bgr, boxes=None):
    return _get_pipe().predict_letters_from_array(img_bgr, boxes)

def predict_letter28x28(gray28):
    return _get_pipe().predict_letter28x28(gray28)
