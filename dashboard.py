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
        body {
            background-color: #f7e7ff;
        }
        .main-title {
            text-align: center;
            color: #7b2cbf;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-text {
            text-align: center;
            color: #5a189a;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== TITLE ==========================
st.markdown('<p class="main-title">Deteksi Objek dan Klasifikasi Gambar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Unggah gambar untuk mendeteksi dan mengklasifikasikan objek di dalamnya</p>', unsafe_allow_html=True)

# ========================== LOAD MODELS ==========================
yolo_model = YOLO('yolov8n.pt')  # Model YOLO untuk deteksi objek
keras_model = tf.keras.models.load_model('model_klasifikasi.h5')  # Model Keras untuk klasifikasi

# ========================== UPLOAD IMAGE ==========================
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    # ========================== YOLO DETECTION ==========================
    results = yolo_model.predict(np.array(image))

    st.subheader("Hasil Deteksi Objek:")
    st.write(results[0].boxes)

    # ========================== IMAGE CLASSIFICATION ==========================
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = keras_model.predict(img_array)
    kelas = np.argmax(pred)

    st.subheader("Hasil Klasifikasi Gambar:")
    st.write(f"Kelas: {kelas}")
else:
    st.info("Silakan unggah gambar terlebih dahulu untuk mendeteksi dan mengklasifikasikan objek.")
