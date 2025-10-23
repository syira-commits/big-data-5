import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "0"
import matplotlib
matplotlib.use('Agg')
import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Tampilan Dashboard
# ==========================
st.set_page_config(page_title="CuteVision App", page_icon="🐱", layout="centered")

st.title("🐾 CuteVision App — Deteksi & Klasifikasi Gambar Lucu")
st.markdown("Aplikasi ini menggunakan YOLO untuk mendeteksi objek dan CNN (TensorFlow) untuk mengklasifikasi gambar. Cocok untuk yang suka hal-hal lucu tapi tetap cerdas!")

menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar dan lihat keajaiban AI bekerja!")

uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("🐱 Sedang mendeteksi objek... tunggu sebentar ya!"):
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="🎉 Hasil Deteksi!", use_container_width=True)

            data = results[0].boxes.data.cpu().numpy()
            if len(data) > 0:
                st.subheader("📋 Detail Deteksi")
                st.dataframe({
                    "Class": [results[0].names[int(cls)] for cls in data[:, 5]],
                    "Confidence": [round(conf, 2) for conf in data[:, 4]],
                    "X": data[:, 0],
                    "Y": data[:, 1],
                    "Width": data[:, 2],
                    "Height": data[:, 3]
                })
            else:
                st.warning("Tidak ada objek yang terdeteksi 😿")

    elif menu == "🧩 Klasifikasi Gambar":
        with st.spinner("🐶 Sedang memprediksi jenis gambar..."):
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = classifier.predict(img_array)

            # ==== Bagian ini HARUS di dalam blok ====
            if prediction.shape[-1] == 1:
                prob = prediction[0][0]
                label = "Uninfected" if prob > 0.5 else "Parasitized"
                confidence = prob if prob > 0.5 else 1 - prob
            else:
                class_names = ["Parasitized", "Uninfected"]
                class_index = np.argmax(prediction)
                label = class_names[class_index]
                confidence = np.max(prediction)
            # ========================================

            st.success("🎊 Prediksi Berhasil!")
            st.write("Hasil Prediksi:", label)
            st.write("Probabilitas:", f"{confidence:.2f}")

else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

# ==========================
# Footer lucu
# ==========================
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow.")
