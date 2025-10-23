import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import torch
import pandas as pd
from ultralytics import YOLO
from collections import Counter

st.set_page_config(page_title="PinkVision Adaptive Filter", page_icon="🩷", layout="centered")
st.title("🩷 PinkVision Adaptive Filter (YOLO + Smart Cell Filter)")

@st.cache_resource
def load_yolo_model():
    return YOLO("model/Mulya Syira_Laporan 4.pt")

yolo_model = load_yolo_model()

# === Fungsi bantu: cek apakah gambar terlalu homogen ===
def is_homogeneous(image, edge_threshold=0.025):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density < edge_threshold

# === Fungsi bantu: cek warna dominan ===
def get_dominant_color(image, k=4):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img.reshape((-1, 3))
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    return dominant  # BGR

# === Fungsi bantu: tentukan apakah warna dominan mirip warna mikroskop ===
def looks_like_cell_color(bgr_color):
    b, g, r = bgr_color
    # Warna khas mikroskop: kebiruan / abu-abu pucat / ungu muda
    if (b > 100 and r < 140 and g < 140) or (abs(r-g) < 20 and abs(g-b) < 20 and r < 160):
        return True
    return False

# === Upload Gambar ===
uploaded_files = st.file_uploader("📸 Upload Gambar", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

if uploaded_files:
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption=f"✨ Gambar: {uploaded_file.name}", use_container_width=True)

        if st.button(f"🌷 Jalankan Deteksi ({uploaded_file.name})"):
            with st.spinner("🪄 Sedang memproses..."):
                time.sleep(1.0)

                # 🔹 Langkah 1: cek homogenitas
                homogen = is_homogeneous(img)

                # 🔹 Langkah 2: cek warna dominan
                dom_color = get_dominant_color(img)
                cell_color_like = looks_like_cell_color(dom_color)

                # 🔹 Debug info
                st.markdown(f"🎨 Warna dominan (BGR): `{dom_color}`")

                # 🔹 Logika adaptif
                if homogen and cell_color_like:
                    st.warning("🧫 Gambar terlalu homogen & warnanya mirip mikroskop — kemungkinan gambar sel.")
                    continue
                elif homogen and not cell_color_like:
                    st.info("🌈 Gambar agak polos tapi warnanya bukan khas mikroskop — tetap dilanjutkan.")

                # 🔹 Jalankan YOLO
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
