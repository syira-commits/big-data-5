import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ========================== CUSTOM CSS ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Quicksand', sans-serif;
            background-color: #faf7f2;
            color: #2e2a26;
        }

        .main-header {
            font-size: 32px;
            font-weight: bold;
            color: #2f2a27;
            margin-bottom: -5px;
        }

        .sub-header {
            color: #6d635a;
            font-size: 16px;
            margin-bottom: 40px;
        }

        .stButton>button {
            background-color: #f0d8e0;
            color: #2f2a27;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
            font-weight: 600;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #e4c1ce;
            transform: scale(1.05);
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: #4b3f3a;
            margin-bottom: 10px;
        }

        .info-card {
            background-color: #fff7f0;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        .result-box {
            background-color: #f7e9ff;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: #4b2e83;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown('<div class="main-header">Good morning, Dr. Olivia</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI Vision Dashboard – Detect and Classify Images Effortlessly</div>', unsafe_allow_html=True)

# ========================== LAYOUT ==========================
col1, col2 = st.columns([1.2, 1])

# ========================== LEFT COLUMN ==========================
with col1:
    st.markdown('<div class="section-title">🔍 Object Detection</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="info-card">Unggah gambar untuk mendeteksi objek menggunakan model YOLOv8.</div>', unsafe_allow_html=True)
        uploaded_detection = st.file_uploader("Upload Gambar Deteksi", type=["jpg", "png", "jpeg"], key="detect")

        if uploaded_detection:
            image = Image.open(uploaded_detection)
            st.image(image, caption="Gambar yang Diupload", use_column_width=True)

            yolo_model = YOLO("yolov8n.pt")
            results = yolo_model.predict(np.array(image))

            st.markdown('<div class="section-title">Hasil Deteksi:</div>', unsafe_allow_html=True)
            st.write(results[0].boxes)
        else:
            st.info("Silakan unggah gambar untuk mendeteksi objek.")

    st.markdown('<hr style="margin:40px 0;">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧠 Image Classification</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="info-card">Unggah gambar sel untuk mengklasifikasikan antara <b>Parasitized</b> dan <b>Uninfected</b>.</div>', unsafe_allow_html=True)
        uploaded_class = st.file_uploader("Upload Gambar Klasifikasi", type=["jpg", "png", "jpeg"], key="classify")

        if uploaded_class:
            image = Image.open(uploaded_class)
            st.image(image, caption="Gambar yang Diupload", use_column_width=True)

            keras_model = tf.keras.models.load_model("model_klasifikasi.h5")
            img = image.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
            pred = keras_model.predict(img_array)
            kelas = np.argmax(pred)

            st.markdown('<div class="section-title">Hasil Klasifikasi:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">Kelas terdeteksi: Kelas {kelas}</div>', unsafe_allow_html=True)
        else:
            st.info("Silakan unggah gambar untuk mengklasifikasikan.")

# ========================== RIGHT COLUMN ==========================
with col2:
    st.markdown('<div class="section-title">📅 Activity Summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <b>Hari Ini:</b> Kamis, 15 Mei 2025<br><br>
        • 2 Gambar dideteksi<br>
        • 1 Klasifikasi berhasil<br>
        • Model aktif: <b>YOLOv8n</b> dan <b>Keras</b><br><br>
        <b>Catatan:</b> Pastikan gambar jelas dan tidak blur untuk hasil terbaik.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧾 Informasi Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <b>YOLOv8n:</b> Model deteksi objek ringan, akurat, cepat.<br>
        <b>Keras CNN:</b> Model klasifikasi berbasis konvolusi untuk gambar sel.<br><br>
        <i>Model telah dioptimasi untuk mendeteksi dan mengklasifikasikan objek sederhana.</i>
    </div>
    """, unsafe_allow_html=True)
