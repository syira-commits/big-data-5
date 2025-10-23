import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch
import pandas as pd
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Gambar", page_icon="🩷", layout="centered")

st.title("🩷 Aplikasi Deteksi Gambar (YOLO + Filter Otomatis)")

# === Load model YOLO ===
@st.cache_resource
def load_yolo_model():
    return YOLO("model_yolo.pt")  # ganti dengan path modelmu

yolo_model = load_yolo_model()

# === Fungsi filter otomatis untuk mendeteksi gambar sel ===
def is_homogeneous(image, edge_threshold=0.025):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    st.write(f"🔬 Edge density: {edge_density:.5f}")  # bisa dihapus kalau mau tanpa debug

    # Ambang batas fleksibel — semakin kecil semakin ketat
    if edge_density < edge_threshold:
        return True  # gambar terlalu polos → kemungkinan gambar sel
    return False

# === Upload Gambar ===
uploaded_files = st.file_uploader("📸 Upload Gambar", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"✨ Gambar: {uploaded_file.name}", use_container_width=True)

        if st.button(f"🌸 Jalankan Deteksi ({uploaded_file.name})"):
            with st.spinner("🪄 Sedang memproses..."):
                time.sleep(1.0)

                # 🔹 Langkah 1: cek apakah gambar mirip sel
                if is_homogeneous(img):
                    st.warning("🧫 Gambar terlalu homogen — kemungkinan gambar sel, bukan Cocopham/Sprite.")
                    continue

                # 🔹 Langkah 2: jalankan YOLO hanya jika bukan gambar sel
                results = yolo_model(img, conf=0.7, iou=0.5)
                det = results[0].boxes

                if det is not None and len(det) > 0:
                    boxes = det.xyxy.cpu().numpy()
                    confs = det.conf.cpu().numpy()
                    classes = det.cls.cpu().numpy().astype(int)
                    class_names = [yolo_model.names[c] for c in classes]

                    df = pd.DataFrame({
                        "Class": class_names,
                        "Confidence": [round(float(c), 2) for c in confs]
                    })

                    # 🔹 Filter hasil YOLO biar nggak spam
                    df = df[df["Confidence"] > 0.8]
                    df = df.drop_duplicates(subset=["Class"])
                    df = df.sort_values(by="Confidence", ascending=False).head(3)

                    if len(df) == 0:
                        st.warning("🤔 Tidak ada objek dengan confidence tinggi.")
                    else:
                        result_img = results[0].plot()
                        st.image(result_img, caption="🎀 Hasil Deteksi YOLO 🎀", use_container_width=True)
                        st.dataframe(df, use_container_width=True)
                        st.success("✅ Deteksi selesai!")
                else:
                    st.warning("🚫 Tidak ada objek yang terdeteksi.")
