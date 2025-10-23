import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import cv2
import matplotlib.pyplot as plt

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="💖 PinkVision: Smart & Cute AI 💖",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM CSS (TEMA PINK)
# ==========================
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe4e9;
        color: #5c005c;
        font-family: "Poppins", sans-serif;
    }
    .stButton>button {
        background-color: #ff85a2;
        color: white;
        border-radius: 12px;
        padding: 8px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff4d79;
    }
    .stSidebar {
        background-color: #fff0f5;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model kamu... tunggu bentar ya 💕"):
    yolo_model, classifier = load_models()
st.success("Model berhasil dimuat! 🌸")

# ==========================
# HEADER
# ==========================
st.title("🌷 PinkVision: Cute Image & Object Detector 🌷")
st.markdown(
    "Selamat datang di *PinkVision*! 💖<br>"
    "Aplikasi ini bisa mendeteksi objek (YOLO), klasifikasi gambar, "
    "dan menampilkan grafik akurasi dengan gaya imut tapi cerdas 🧠✨",
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🎀 Pilihan Mode")
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    [
        "Deteksi Objek (YOLO)",
        "Klasifikasi Gambar",
        "Grafik Akurasi Model"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu klik tombol Mulai Prediksi! 🌸")

# ==========================
# MODE UTAMA: DETEKSI & KLASIFIKASI
# ==========================
if menu in ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]:
    uploaded_file = st.file_uploader(
        "📸 Unggah gambar kamu di sini:",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="✨ Gambar yang diunggah ✨", use_container_width=True)

        if st.button("🎀 Jalankan Deteksi"):
            with st.spinner("🪄 Sedang memproses..."):
                time.sleep(1)

                # ==========================
                # CEK HOMOGENITAS GAMBAR
                # ==========================
                img_cv = np.array(img.convert("RGB"))
                gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size

                st.write(f"🧮 Edge Density: {edge_density:.6f}")

                # Ambang batas empiris (bisa kamu ubah nanti)
                EDGE_THRESHOLD = 0.004

                if edge_density < EDGE_THRESHOLD:
                    st.warning("🩺 Gambar terlalu homogen — kemungkinan gambar sel, bukan Cocopham/Sprite.")
                    st.stop()

                # ==========================
                # MODE DETEKSI OBJEK
                # ==========================
                if menu == "Deteksi Objek (YOLO)":
                    results = yolo_model(img)
                    result_img = results[0].plot()
                    st.image(result_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)

                    st.success("✨ Deteksi selesai! ✨")

                # ==========================
                # MODE KLASIFIKASI GAMBAR
                # ==========================
                elif menu == "Klasifikasi Gambar":
                    img_resized = img.resize((128, 128))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    prediction = classifier.predict(img_array)
                    class_index = np.argmax(prediction)
                    confidence = np.max(prediction)

                    labels = ["Uninfected", "Parasitized"]  # ubah sesuai label kamu
                    predicted_label = labels[class_index]

                    st.write(f"🎯 *Hasil Prediksi:* {predicted_label}")
                    st.progress(float(confidence))

                    if confidence > 0.85:
                        st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                    elif confidence > 0.6:
                        st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                    else:
                        st.error("😅 Model kurang yakin. Coba gambar lain yang lebih jelas ya!")

# ==========================
# MODE: GRAFIK AKURASI
# ==========================
if menu == "Grafik Akurasi Model":
    st.subheader("📊 Grafik Akurasi Model")
    model_names = ["Model A", "Model B", "Model C"]
    accuracy = [0.91, 0.88, 0.93]

    fig, ax = plt.subplots()
    ax.bar(model_names, accuracy, color=["#ff85a2", "#ffa6c9", "#ffb6d9"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Akurasi")
    ax.set_title("💖 Perbandingan Akurasi Model 💖")
    st.pyplot(fig)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<center>Made with 💕 by Mulya Syira 🌸</center>", unsafe_allow_html=True)
