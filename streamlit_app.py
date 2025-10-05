import io
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf

from src.ocr.preprocess import prepare_image
from src.ocr.pipeline import predict_line_from_array, predict_line_string_from_array, _segment_boxes


MODEL_PATH = "models/mnist_cnn.keras"

@st.cache_resource  # učitaj model jednom
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
st.title("OCR – MNIST demo")
st.write("Uploaduj sliku cifre ili niza cifara. Model: CNN (MNIST) sa augmentacijom.")

mode_multi = st.checkbox("Višecifarski mod (segmentacija & niz cifara)", value=False)
uploaded = st.file_uploader("Izaberi sliku (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded:
    st.image(uploaded, caption="Ulazna slika", use_column_width=True)
    img_bgr = read_image_to_bgr(uploaded)
    if img_bgr is None:
        st.error("Ne mogu da učitam sliku.")
        st.stop()

    model = load_model()

    if mode_multi:
        preds = predict_line_from_array(img_bgr)
        out_str = "".join(str(l) if isinstance(l, int) else "?" for (l, _) in preds)
        st.subheader(f"Predikcija niza: **{out_str}**")

        st.write("Confidence po cifri (sleva nadesno):")
        for i, (lab, conf) in enumerate(preds, 1):
            st.write(f"{i}. {lab}  —  {conf:.3f}")
        # Prikaz segmentacije (bounding box-ovi)
        img_draw = img_bgr.copy()
        gray = cv2.cvtColor(img_draw, cv2.COLOR_BGR2GRAY)
        for (x, y, w, h) in _segment_boxes(gray):
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB),
                caption="Segmentacija (bbox)", use_column_width=True)
    else:
        # --- single digit mod ---
        # Sačuvaj u privremeni fajl u memoriji → cv preprocess već očekuje path.
        # Da ne pišemo na disk, kreiramo privremeni PNG iz img_bgr:
        ok, buff = cv2.imencode(".png", img_bgr)
        if not ok:
            st.error("Ne mogu da obradim sliku.")
            st.stop()
        tmp_bytes = io.BytesIO(buff.tobytes())

        # OpenCV čita sa puta, mi ćemo napraviti privremen fajl na RAM disku Streamlita
        # ali jednostavnije: upotrebimo varijantu preprocess-a direktno iz array-a:
        # (Ako želiš isključivo path-based, upiši u /tmp; na Windowsu može i NamedTemporaryFile)
        # Ovde koristimo direktan MNIST-style pipeline: reimplement kratko iz prepare_image:
        # lakše: snimi u privremeni path
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(tmp_bytes.getvalue())
            tmp_path = tmp.name

        x = prepare_image(tmp_path)  # (1,28,28,1)
        os.remove(tmp_path)

        probs = model.predict(x, verbose=0)[0]
        top3 = top3_from_probs(probs)

        st.subheader(f"Predikcija: **{top3[0][0]}**  (p={top3[0][1]:.3f})")
        st.write("Top-3:")
        for lab, p in top3:
            st.write(f"- {lab}: {p:.3f}")
