import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
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
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # model deteksi objek
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # model klasifikasi utama
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
    "perbandingan dua model, dan menampilkan grafik akurasi dengan gaya imut tapi cerdas 🧠✨",
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
        "Perbandingan Dua Model",
        "Grafik Akurasi Model"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu klik tombol Mulai Prediksi! 🌸")

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

            if st.button(f"🌷 Mulai Prediksi: {uploaded_file.name}"):
                with st.spinner("🪄 Sedang memproses..."):
                    time.sleep(1.2)

                    # ==========================
                    # MODE DETEKSI OBJEK
                    # ==========================
                    if menu == "Deteksi Objek (YOLO)":
                        results = yolo_model.predict(img, conf=0.7, iou=0.4)
                        names = results[0].names
                        allowed_labels = ["cocopham", "sprite"]
                        filtered = []

                        # Filter hasil deteksi
                        for box in results[0].boxes:
                            cls_id = int(box.cls)
                            conf = float(box.conf)
                            label = names.get(cls_id, "Unknown")

                            if label not in allowed_labels or conf < 0.6:
                                # Hapus deteksi yang tidak termasuk
                                box.conf[:] = 0
                            else:
                                x1, y1, x2, y2 = box.xyxy[0]
                                filtered.append((label, conf, (x1, y1, x2, y2)))

                        # Tampilkan hasil hanya jika ada deteksi valid
                        if len(filtered) == 0:
                            st.warning("😿 Tidak ada objek terdeteksi (hanya mendeteksi Cocopham & Sprite)")
                        else:
                            annotated_img = results[0].plot()
                            st.image(annotated_img, caption="🎀 Hasil Deteksi Objek 🎀", use_container_width=True)

                            st.subheader("📋 Detail Deteksi")
                            st.dataframe({
                                "Class": [f[0] for f in filtered],
                                "Confidence": [round(f[1], 2) for f in filtered],
                            })

                        st.success("✨ Deteksi selesai dengan sukses! ✨")
                        st.markdown("💡 *Saran:* Jika hasil kurang akurat, coba gambar dengan pencahayaan lebih terang 🌞")

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

                        labels = ["Uninfected", "Parasitized"]
                        predicted_label = labels[class_index]

                        st.write(f"🎯 *Hasil Prediksi:* {predicted_label}")
                        st.progress(float(confidence))

                        if confidence > 0.85:
                            st.success("🌈 Model sangat yakin dengan hasil prediksi ini!")
                        elif confidence > 0.6:
                            st.warning("🌤 Model agak ragu, tapi masih cukup yakin.")
                        else:
                            st.error("😅 Model kurang yakin. Coba gambar lain yang lebih jelas ya!")

                        st.markdown("💡 *Saran:* Gunakan gambar jelas, tidak blur, agar hasil klasifikasi lebih akurat 📷")


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
