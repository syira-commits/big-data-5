import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt
import cv2

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
# CUSTOM CSS
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
    "Aplikasi ini bisa melakukan deteksi objek (YOLO), klasifikasi gambar, "
    "dan menampilkan grafik akurasi dengan gaya imut tapi cerdas 🧠✨",
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("🎀 Pilihan Mode")
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Grafik Akurasi Model"]
)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu klik tombol Mulai Prediksi! 🌸")

# ==========================
# UTILITAS FILTER GAMBAR SEL
# ==========================
def is_cell_image(pil_img):
    """Deteksi sederhana apakah gambar kemungkinan besar adalah gambar sel mikroskop."""
    img_np = np.array(pil_img.resize((128, 128)))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    std_color = np.std(img_np)
    edge_density = cv2.Canny(gray, 50, 150).mean()

    # Ambang batas — sesuaikan jika perlu
    EDGE_THRESHOLD = 25
    COLOR_STD_THRESHOLD = 18

    # Gambar sel biasanya punya tekstur halus & warna abu-abu, tidak terlalu kontras
    if edge_density < EDGE_THRESHOLD and std_color < COLOR_STD_THRESHOLD:
        return True  # kemungkinan gambar sel mikroskop
    else:
        return False  # gambar benda biasa (kaleng, meja, dsb.)

# ==========================
# MODE: DETEKSI & KLASIFIKASI
# ==========================
if menu in ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"]:
    uploaded_files = st.file_uploader(
        "📸 Unggah satu atau beberapa gambar kamu di sini:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=f"✨ Gambar: {uploaded_file.name}", use_container_width=True)

            if menu == "Deteksi Objek (YOLO)":
                if st.button(f"🌷 Jalankan Deteksi ({uploaded_file.name})"):
                    with st.spinner("🪄 Sedang memproses deteksi..."):
                        time.sleep(1.2)

                        # Jalankan YOLO dengan threshold lebih tinggi biar nggak asal deteksi
                        results = yolo_model(img, conf=0.7)
                        result_img = results[0].plot()

                        # Ambil nama kelas dan confidence
                        det = results[0].boxes
                        if det is not None and len(det) > 0:
                            st.image(result_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)
                            st.success("✨ Deteksi selesai! Objek berhasil ditemukan.")
                        else:
                            st.warning("🤔 Tidak ada objek dengan keyakinan tinggi yang terdeteksi. "
                                       "Coba gambar dengan pencahayaan lebih baik 🌞")

            elif menu == "Klasifikasi Gambar":
                if st.button(f"🌸 Jalankan Klasifikasi ({uploaded_file.name})"):
                    with st.spinner("🩺 Sedang menganalisis gambar..."):
                        time.sleep(1.2)

                        # Filter gambar dulu — jangan klasifikasi kalau bukan gambar sel
                        if not is_cell_image(img):
                            st.warning("🧫 Gambar terlalu kompleks — kemungkinan gambar benda (kaleng, botol, dsb.), bukan gambar sel mikroskop.")
                            st.stop()

                        # Lanjut klasifikasi
                        img_resized = img.resize((128, 128))
                        img_array = image.img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0) / 255.0

                        prediction = classifier.predict(img_array)
                        class_index = np.argmax(prediction)
                        confidence = np.max(prediction)
                        labels = ["Uninfected", "Parasitized"]
                        predicted_label = labels[class_index]

                        st.success("🎉 Prediksi Berhasil!")
                        st.write(f"📋 **Hasil Prediksi:** {predicted_label}")
                        st.progress(float(confidence))
                        st.info(f"🔢 Probabilitas: {confidence:.2f}")

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
    ax.set_title("💖 Perbandingan Akurasi Model Klasifikasi 💖")

    st.pyplot(fig)

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("<center>Made with 💕 by Mulya Syira 🌸</center>", unsafe_allow_html=True)
