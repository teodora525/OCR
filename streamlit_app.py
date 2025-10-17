import io
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf

from src.ocr.preprocess import prepare_image
from src.ocr.pipeline import (
    predict_line_from_array,          # cifre (multi)
    predict_line_string_from_array,   # cifre (string) - ako ti zatreba
    _segment_boxes,                   # crtanje bbox-ova
    predict_letters_from_array,       # SLOVA (multi)
    predict_letter28x28,           # SLOVA (single)
    predict_text_from_array,
)

MODEL_PATH = "models/mnist_cnn.keras"

@st.cache_resource  # učitaj model jednom (za CIFRE)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def read_image_to_bgr(file) -> np.ndarray | None:
    """Streamlit UploadedFile -> BGR numpy (kompatibilno sa OpenCV)."""
    data = file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def top3_from_probs(probs: np.ndarray):
    idx = np.argsort(probs)[::-1][:3]
    return [(int(i), float(probs[i])) for i in idx]

st.set_page_config(page_title="OCR Demo", page_icon="🔢", layout="centered")
st.title("OCR – MNIST / EMNIST demo")
st.write("Uploaduj sliku cifre/slova ili niza. Modeli: cifre (MNIST), slova (EMNIST Letters).")

# NOVO: ako je čekirano – radi se sa slovima; inače ostaje tvoj stari flow za cifre
letters_mode = st.checkbox("Režim slova (A–Z)", value=False)
mode_multi   = st.checkbox("Višesimbolni mod (segmentacija & niz)", value=False)
mixed_mode   = st.checkbox("Mešovito (A–Z + 0–9)", value=True)  # podrazumevano uključen


uploaded = st.file_uploader("Izaberi sliku (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded:
    st.image(uploaded, caption="Ulazna slika", use_column_width=True)
    img_bgr = read_image_to_bgr(uploaded)
    if img_bgr is None:
        st.error("Ne mogu da učitam sliku.")
        st.stop()

    # --- MEŠOVITO (A–Z + 0–9) ima prioritet ---
    if mixed_mode:
        text, details = predict_text_from_array(img_bgr)
        st.subheader(f"Predikcija (mešovito): **{text}**")
        st.write("Confidence po simbolu (sleva nadesno):")
        for i, (sym, conf, _) in enumerate(details, 1):
            st.write(f"{i}. {sym} — {conf:.3f}")

        img_draw = img_bgr.copy()
        for (_, _, (x, y, w, h)) in details:
            cv2.rectangle(img_draw, (x, y), (x+w, y+h), (0,255,0), 2)
        st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                 caption="Segmentacija (mešovito)", use_column_width=True)
        st.stop()  # ne nastavljaj na cifre/slova grane

    # ====== CIFRE (ostaje identično kao pre) ======
    if not letters_mode:
        model = load_model()  # tvoj postojeći cifarski model

        if mode_multi:
            preds = predict_line_from_array(img_bgr)
            out_str = "".join(str(l) if isinstance(l, int) else "?" for (l, _) in preds)
            st.subheader(f"Predikcija niza (cifre): **{out_str}**")

            st.write("Confidence po cifri (sleva nadesno):")
            for i, (lab, conf) in enumerate(preds, 1):
                st.write(f"{i}. {lab}  —  {conf:.3f}")

            # Prikaz segmentacije (bbox-ovi)
            img_draw = img_bgr.copy()
            gray = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in _segment_boxes(gray):
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

            st.image(
                cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                caption="Segmentacija (bbox)",
                use_column_width=True
            )
        else:
            # --- single digit mod (tvoj postojeći kod) ---
            ok, buff = cv2.imencode(".png", img_bgr)
            if not ok:
                st.error("Ne mogu da obradim sliku.")
                st.stop()
            tmp_bytes = io.BytesIO(buff.tobytes())

            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(tmp_bytes.getvalue())
                tmp_path = tmp.name

            x = prepare_image(tmp_path)  # (1,28,28,1)
            os.remove(tmp_path)

            probs = model.predict(x, verbose=0)[0]
            top3 = top3_from_probs(probs)

            st.subheader(f"Predikcija (cifra): **{top3[0][0]}**  (p={top3[0][1]:.3f})")
            st.write("Top-3:")
            for lab, p in top3:
                st.write(f"- {lab}: {p:.3f}")

    # ====== SLOVA ======
    else:
        if mode_multi:
            preds = predict_letters_from_array(img_bgr)  # [(letter, conf, (x,y,w,h)), ...]
            out_str = "".join([l for (l, _, _) in preds]) if preds else ""
            st.subheader(f"Predikcija niza (slova): **{out_str}**")

            st.write("Confidence po slovu (sleva nadesno):")
            for i, (letter, conf, _) in enumerate(preds, 1):
                st.write(f"{i}. {letter}  —  {conf:.3f}")

            img_draw = img_bgr.copy()
            for (_, _, (x, y, w, h)) in preds:
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

            st.image(
                cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                caption="Segmentacija (bbox)",
                use_column_width=True
            )
        else:
            gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray_full, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if th.mean() > 127:
                th = 255 - th
            num, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

            candidate = None
            best_area = -1
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area > best_area:
                    best_area = area
                    candidate = (x, y, w, h)

            if candidate is None:
                st.error("Nisam pronašao slovo na slici.")
                st.stop()

            x, y, w, h = candidate
            roi = gray_full[y:y+h, x:x+w]

            scale = 20.0 / max(w, h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((28, 28), dtype=np.uint8)
            xs = (28 - nw) // 2
            ys = (28 - nh) // 2
            canvas[ys:ys+nh, xs:xs+nw] = small

            letter, conf = predict_letter28x28(canvas)
            st.subheader(f"Predikcija (slovo): **{letter}**  (p={conf:.3f})")