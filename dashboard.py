import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==============================
# Load Models
# ==============================
@st.cache_resource
def load_models():
    # Model deteksi objek (YOLO)
    yolo_model = YOLO("model/best.pt")

    # Model klasifikasi gambar (TensorFlow)
    classifier = tf.keras.models.load_model("model/classifier_model.h5")

    return yolo_model, classifier


# ==============================
# UI
# ==============================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]
)

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # Load model sekali saja
    yolo_model, classifier = load_models()

    # ==============================
    # Mode Deteksi Objek (YOLO)
    # ==============================
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("Hasil Deteksi Objek (YOLO)")
        results = yolo_model.predict(np.array(img))
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    # ==============================
    # Mode Klasifikasi Gambar (TensorFlow)
    # ==============================
    elif menu == "Klasifikasi Gambar":
        st.subheader("Hasil Klasifikasi Gambar")

        # Preprocessing
        img_resized = img.resize((224, 224))  # sesuaikan dengan ukuran model
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("Prediksi kelas:", class_index)
        st.write("Probabilitas:", float(np.max(prediction)))
