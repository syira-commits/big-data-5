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
# Setup Page
# ==========================
st.set_page_config(page_title="CuteVision App", page_icon="🐾", layout="wide")

# ==========================
# Custom CSS untuk tampilan pastel modern
# ==========================
st.markdown("""
<style>
body {
    background-color: #FFF8FB;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #FAF9F6;
    border-right: 1px solid #EEE;
}
h1, h2, h3 {
    font-weight: 600;
}
.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1E1E1E;
    margin-bottom: 0.5rem;
}
.subtext {
    font-size: 1rem;
    color: #555;
    margin-bottom: 2rem;
}
.upload-box {
    background-color: white;
    border: 2px dashed #E5E7EB;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
.result-card {
    background-color: white;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.07);
    margin-top: 1.5rem;
}
.result-card:hover {
    transform: scale(1.01);
    transition: 0.3s;
}
.footer {
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("🐾 Mode CuteVision")
menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar dan lihat keajaiban AI bekerja!")

# ==========================
# Layout Utama
# ==========================
col1, col2 = st.columns([2.3, 1.2])

with col1:
    st.markdown("<div class='main-title'>CuteVision App — Deteksi & Klasifikasi Gambar Lucu</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Aplikasi ini menggunakan <b>YOLO</b> untuk mendeteksi objek dan <b>CNN (TensorFlow)</b> untuk mengklasifikasi gambar. Cocok untuk kamu yang suka hal-hal lucu tapi tetap cerdas!</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah Gambar Lucumu di Sini", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

        # ==========================
        # MODE DETEKSI YOLO
        # ==========================
        if menu == "🎯 Deteksi Objek (YOLO)":
            st.subheader("⚙️ Pengaturan Deteksi")
            detect_option = st.checkbox("Aktifkan deteksi objek YOLO", value=True)

            if detect_option:
                with st.spinner("🐱 Sedang mendeteksi objek... tunggu sebentar ya!"):
                    results = yolo_model(img, conf=0.25, iou=0.3)
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        result_img = results[0].plot(line_width=2, font_size=12)
                        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                        st.image(result_img, caption="🎉 Hasil Deteksi!", use_container_width=True)
                        data = boxes.data.cpu().numpy()
                        st.dataframe({
                            "Class": [results[0].names[int(cls)] for cls in data[:, 5]],
                            "Confidence": [round(conf, 2) for conf in data[:, 4]],
                            "X_min": data[:, 0],
                            "Y_min": data[:, 1],
                            "X_max": data[:, 2],
                            "Y_max": data[:, 3]
                        })
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.warning("Tidak ada objek yang terdeteksi 😿")
            else:
                st.info("Deteksi YOLO dimatikan. Tidak ada bounding box yang ditampilkan.")

        # ==========================
        # MODE KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "🧩 Klasifikasi Gambar":
            with st.spinner("🐶 Sedang memprediksi jenis gambar..."):
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

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

                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.success("🎊 Prediksi Berhasil!")
                st.write("**Hasil Prediksi:**", label)
                st.write("**Probabilitas:**", f"{confidence:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='upload-box'>📤 Silakan unggah gambar terlebih dahulu 💡</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Informasi Model")
    st.markdown("<div class='result-card'><b>YOLOv8:</b> Deteksi objek cepat & akurat.<br><b>CNN:</b> Klasifikasi citra lucu.<br><br><b>Confidence Threshold:</b> 0.25<br><b>IoU:</b> 0.3</div>", unsafe_allow_html=True)

    st.markdown("### 💡 Tips")
    st.markdown("<div class='result-card'>🖼️ Gunakan gambar dengan resolusi jelas.<br>🐾 Coba bandingkan hasil YOLO dan CNN.<br>📊 Eksperimen dengan berbagai objek lucu!</div>", unsafe_allow_html=True)

# ==========================
# Footer lucu
# ==========================
st.markdown("<div class='footer'>🐾 Dibuat oleh <b>Mulya Syira</b> — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow.</div>", unsafe_allow_html=True)
