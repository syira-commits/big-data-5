import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time
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
# TEMA PINK (CSS)
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
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # YOLO model
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # CNN model
    return yolo_model, classifier

with st.spinner("💫 Memuat model kamu... tunggu sebentar 💕"):
    yolo_model, classifier = load_models()
st.success("Model berhasil dimuat! 🌸")

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.header("🎀 Pilihan Mode")
menu = st.sidebar.selectbox(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar", "Grafik Akurasi Model"]
)
strict_mode = st.sidebar.toggle("🔒 Mode Filter Ketat", value=True)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu klik tombol untuk memproses 🌸")

# ==========================
# UNGGAH GAMBAR
# ==========================
uploaded_file = st.file_uploader("📸 Unggah gambar di sini:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar yang diunggah ✨", use_container_width=True)
    img_cv = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Hitung edge density (untuk mendeteksi gambar sel)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

    # ==========================
    # MODE DETEKSI (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        if st.button("🎯 Jalankan Deteksi"):
            with st.spinner("🐱 Sedang mendeteksi objek..."):
                results = yolo_model(img)
                boxes = results[0].boxes
                data = boxes.data.cpu().numpy() if boxes is not None else np.array([])
                names = results[0].names

                allowed_labels = ["cocopham", "sprite"]
                filtered = []

                # 🔒 Filter otomatis
                if strict_mode and edge_density < 0.03:
                    st.warning("🧫 Gambar terlalu homogen — kemungkinan gambar sel, bukan cocopham/sprite.")
                else:
                    if len(data) > 0:
                        for box in data:
                            x1, y1, x2, y2, conf, cls_id = box
                            label = names.get(int(cls_id), "Unknown")
                            area = (x2 - x1) * (y2 - y1)
                            img_area = gray.shape[0] * gray.shape[1]

                            if (
                                label in allowed_labels
                                and conf > 0.6
                                and 0.02 < area / img_area < 0.8
                            ):
                                filtered.append((label, float(conf)))

                    if len(filtered) == 0:
                        st.warning("😿 Tidak ada objek relevan terdeteksi (mungkin gambar sel).")
                    else:
                        annotated_img = results[0].plot()
                        st.image(annotated_img, caption="🎉 Hasil Deteksi!", use_container_width=True)
                        st.dataframe({
                            "Class": [f[0] for f in filtered],
                            "Confidence": [round(f[1], 2) for f in filtered],
                        })
                st.success("✨ Deteksi selesai!")

    # ==========================
    # MODE KLASIFIKASI
    # ==========================
    elif menu == "Klasifikasi Gambar":
        if st.button("🧩 Jalankan Klasifikasi"):
            with st.spinner("🩷 Sedang memprediksi jenis gambar..."):
                # 🔒 Jika terlalu banyak tepi → kemungkinan bukan gambar sel
                if strict_mode and edge_density > 0.12:
                    st.warning("📦 Gambar terlihat seperti objek non-sel (mungkin cocopham/sprite). Klasifikasi dilewati.")
                else:
                    img_resized = img.resize((128, 128))
                    img_array = image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0

                    prediction = classifier.predict(img_array)
                    if prediction.shape[-1] == 1:
                        prob = prediction[0][0]
                        label = "Uninfected" if prob > 0.5 else "Parasitized"
                        confidence = prob if prob > 0.5 else 1 - prob
                    else:
                        class_names = ["Parasitized", "Uninfected"]
                        class_index = np.argmax(prediction)
                        label = class_names[class_index]
                        confidence = np.max(prediction)

                    st.success("🎊 Prediksi Berhasil!")
                    st.write(f"📋 Hasil Prediksi: **{label}**")
                    st.progress(float(confidence))
                    st.write(f"🔢 Probabilitas: {confidence:.2f}")

# ==========================
# MODE GRAFIK AKURASI
# ==========================
elif menu == "Grafik Akurasi Model":
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
