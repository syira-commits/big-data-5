import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="Deteksi Objek dan Klasifikasi Gambar",
    page_icon="📸",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Quicksand', sans-serif;
            background: linear-gradient(135deg, #d9b99b 0%, #f5e6ca 100%);
            color: #4b2e05;
        }

        .main-title {
            text-align: center;
            color: #4b2e05;
            font-size: 38px;
            font-weight: bold;
            margin-bottom: -10px;
        }

        .sub-text {
            text-align: center;
            color: #6b4226;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .stButton button {
            background-color: #a47c48;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background-color: #8b6333;
            transform: scale(1.05);
        }

        .mode-title {
            text-align: center;
            font-size: 26px;
            color: #3e2723;
            font-weight: bold;
            margin-bottom: 20px;
        }

        .info-box {
            background-color: #f3e5ab;
            padding: 15px;
            border-radius: 10px;
            color: #3e2723;
            font-size: 16px;
            margin-top: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== TITLE ==========================
st.markdown('<p class="main-title">📸 Deteksi Objek dan Klasifikasi Gambar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Pilih mode untuk mendeteksi atau mengklasifikasikan gambar dengan model AI</p>', unsafe_allow_html=True)

# ========================== MODE SELECTION ==========================
menu = st.radio("Pilih Mode:", ["🏠 Menu Utama", "🔍 Deteksi Objek", "🧠 Klasifikasi Gambar"], horizontal=True)

# ========================== MENU UTAMA ==========================
if menu == "🏠 Menu Utama":
    st.markdown('<div class="mode-title">Selamat Datang di Aplikasi Deteksi & Klasifikasi!</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 18px;">
        Aplikasi ini memiliki dua mode utama:<br><br>
        <b>🔍 Deteksi Objek:</b> Menggunakan model YOLO untuk mengenali objek dalam gambar.<br>
        <b>🧠 Klasifikasi Gambar:</b> Menggunakan model Keras (.h5) untuk mengklasifikasikan gambar sel.<br><br>
        Pilih mode di atas untuk memulai!
    </div>
    """, unsafe_allow_html=True)

# ========================== DETEKSI OBJEK ==========================
elif menu == "🔍 Deteksi Objek":
    st.markdown('<div class="mode-title">🔍 Mode Deteksi Objek</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Gunakan gambar seperti <b>Cocopham</b> atau <b>Sprite</b> untuk uji deteksi.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah Gambar untuk Deteksi", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)

        yolo_model = YOLO("yolov8n.pt")
        results = yolo_model.predict(np.array(image))

        st.subheader("Hasil Deteksi Objek:")
        st.write(results[0].boxes)
    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk mendeteksi objek.")

# ========================== KLASIFIKASI GAMBAR ==========================
elif menu == "🧠 Klasifikasi Gambar":
    st.markdown('<div class="mode-title">🧠 Mode Klasifikasi Gambar</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Gunakan gambar seperti <b>Sel Parasitized</b> atau <b>Uninfected</b> untuk uji klasifikasi.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah Gambar untuk Klasifikasi", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_column_width=True)

        keras_model = tf.keras.models.load_model("model_klasifikasi.h5")
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = keras_model.predict(img_array)
        kelas = np.argmax(pred)

        st.subheader("Hasil Klasifikasi:")
        st.write(f"Gambar terdeteksi sebagai: **Kelas {kelas}**")
    else:
        st.info("Silakan unggah gambar terlebih dahulu untuk mengklasifikasikan.")
