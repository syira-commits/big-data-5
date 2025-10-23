import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time

st.set_page_config(page_title="Smart YOLO Filter", page_icon="🧠", layout="centered")
st.title("🩷 YOLO Smart Filter (Cocopham & Sprite)")

@st.cache_resource
def load_model():
    return YOLO("model/Mulya Syira_Laporan 4.pt")

model = load_model()

# --- fungsi cek apakah gambar mirip gambar sel ---
def is_cell_image(image, edge_threshold=0.015):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < edge_threshold:
        # kalau terlalu polos, kemungkinan gambar mikroskop
        return True
    return False

# --- fungsi utama deteksi ---
def detect_objects(img):
    results = model(img, conf=0.7, iou=0.5)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return pd.DataFrame(columns=["Class", "Confidence"])

    names = model.names
    df = pd.DataFrame({
        "Class": [names[int(c)] for c in boxes.cls.cpu().numpy()],
        "Confidence": [round(float(x), 2) for x in boxes.conf.cpu().numpy()]
    })
    return df.sort_values("Confidence", ascending=False)

# --- upload gambar ---
uploaded_file = st.file_uploader("📸 Upload gambar untuk deteksi:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Gambar diunggah", use_container_width=True)

    if st.button("🌸 Jalankan Deteksi"):
        with st.spinner("Menganalisis gambar..."):
            time.sleep(1)

            # Filter domain otomatis
            if is_cell_image(img):
                st.warning("🧫 Gambar terlalu polos — kemungkinan besar gambar sel mikroskop, bukan produk Cocopham/Sprite.")
            else:
                df = detect_objects(img)
                if len(df) > 0:
                    result_img = model(img, conf=0.7)[0].plot()
                    st.image(result_img, caption="🎀 Hasil Deteksi 🎀", use_container_width=True)
                    st.dataframe(df)
                    st.success("✅ Deteksi selesai!")
                else:
                    st.info("Tidak ada objek yang cocok terdeteksi.")
