import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Konfigurasi Dashboard
# ==========================
st.set_page_config(page_title="CuteVision App", page_icon="🐾", layout="wide")

# ==========================
# CSS Kustom (Tema Pastel Ungu-Pink)
# ==========================
st.markdown("""
    <style>
    .main {
        background-color: #FAF7FF;
    }
    .stSidebar {
        background-color: #F4EAFB;
    }
    h1, h2, h3, h4, h5 {
        color: #4B2E83;
    }
    .stButton>button {
        background-color: #CDB4DB;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #B5838D;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # Ganti path sesuai model kamu
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Sidebar
# ==========================
st.sidebar.title("🌈 Pilih Mode:")
mode = st.sidebar.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

st.sidebar.write("---")
st.sidebar.info("Unggah gambar dan lihat keajaiban AI bekerja!")

# ==========================
# Judul Utama
# ==========================
st.title("🐾 CuteVision App — Deteksi & Klasifikasi Gambar Lucu")
st.write("Aplikasi ini menggunakan YOLO untuk mendeteksi objek dan CNN (TensorFlow) untuk mengklasifikasikan gambar. "
         "Cocok untuk kamu yang suka hal-hal lucu tapi tetap cerdas!")

# ==========================
# Upload Gambar
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Gambar Asli")
        st.image(image_pil, use_container_width=True)

    # ==========================
    # Progress Bar & Prediksi
    # ==========================
    with st.spinner("✨ AI sedang menganalisis gambar kamu..."):
        time.sleep(1.5)  # simulasi loading
        st.progress(100)

        if mode == "Deteksi Objek (YOLO)":
            results = yolo_model(image_pil)
            with col2:
                st.subheader("🎯 Hasil Deteksi Objek")
                st.image(results[0].plot(), use_container_width=True)
                st.success("AI berhasil mendeteksi objek di gambar kamu!")

        else:
            img = image_pil.resize((150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0
            preds = classifier.predict(x)
            label = np.argmax(preds, axis=1)[0]

            with col2:
                st.subheader("🧠 Hasil Klasifikasi")
                st.image(image_pil, use_container_width=True)
                st.success(f"Gambar ini termasuk ke dalam kelas **{label}**.")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk mulai.")

# ==========================
# Footer
# ==========================
st.write("---")
st.caption("🐾 Dibuat oleh Mulya Syira — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow.")
