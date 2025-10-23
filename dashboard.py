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
# Konfigurasi Halaman
# ==========================
st.set_page_config(page_title="Deteksi Objek dan Klasifikasi Gambar", page_icon="📸", layout="wide")

# ==========================
# Gaya dan Warna Halaman
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fff5e4 0%, #ffe3e3 50%, #e3f2ff 100%);
    background-attachment: fixed;
    color: #2e2e2e;
    font-family: 'Quicksand', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.main-title {
    text-align: center;
    font-size: 40px;
    color: #3e2723;
    font-weight: bold;
    margin-top: 25px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.2);
}

.sub-text {
    text-align: center;
    color: #4b3832;
    font-size: 18px;
    margin-bottom: 40px;
}

.option-card {
    background: rgba(255,255,255,0.9);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
    transition: 0.3s ease;
}
.option-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 25px rgba(0,0,0,0.3);
}

.stButton>button {
    background-color: #a47c48;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    transition: 0.3s ease;
}
.stButton>button:hover {
    background-color: #8b6333;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Judul dan Deskripsi
# ==========================
st.markdown('<p class="main-title">📸 Deteksi Objek dan Klasifikasi Gambar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Gunakan teknologi AI untuk mendeteksi objek menggunakan YOLO dan mengklasifikasikan gambar menggunakan CNN (TensorFlow).</p>', unsafe_allow_html=True)

# ==========================
# Pilihan Mode
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='option-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Deteksi Objek (YOLO)")
    st.write("Gunakan model YOLO untuk mendeteksi objek seperti **Cocopham** dan **Sprite** dalam gambar.")
    deteksi_btn = st.button("Gunakan Mode Ini", key="deteksi")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='option-card'>", unsafe_allow_html=True)
    st.subheader("🧩 Klasifikasi Gambar (CNN)")
    st.write("Gunakan model CNN untuk mengenali gambar **Parasitized** atau **Uninfected**.")
    klasifikasi_btn = st.button("Gunakan Mode Ini", key="klasifikasi")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================
# Tampilkan Input Berdasarkan Pilihan
# ==========================
if deteksi_btn:
    st.markdown("### 🔍 Mode Deteksi Objek")
    uploaded_file = st.file_uploader("Unggah Gambar untuk Deteksi", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        with st.spinner("Mendeteksi objek..."):
            img_cv = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            if edge_density > 0.12:
                st.warning("📄 Gambar terdeteksi sebagai teks/grafik — tidak ada objek relevan.")
            else:
                results = yolo_model(img)
                boxes = results[0].boxes
                data = boxes.data.cpu().numpy() if boxes is not None else np.array([])
                names = results[0].names

                allowed_labels = ["cocopham", "sprite"]
                filtered = []

                if len(data) > 0:
                    for box in data:
                        x1, y1, x2, y2, conf, cls_id = box
                        label = names.get(int(cls_id), "Unknown")
                        area = (x2 - x1) * (y2 - y1)
                        img_area = gray.shape[0] * gray.shape[1]
                        if label in allowed_labels and conf > 0.6 and 0.02 < area / img_area < 0.8:
                            filtered.append((label, float(conf)))

                if len(filtered) == 0:
                    st.warning("😿 Tidak ada objek terdeteksi (hanya mendeteksi Cocopham & Sprite).")
                else:
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="🎉 Hasil Deteksi!", use_container_width=True)
                    st.subheader("📋 Detail Deteksi")
                    st.dataframe({
                        "Class": [f[0] for f in filtered],
                        "Confidence": [round(f[1], 2) for f in filtered],
                    })

if klasifikasi_btn:
    st.markdown("### 🧠 Mode Klasifikasi Gambar")
    uploaded_file = st.file_uploader("Unggah Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang Diupload", use_container_width=True)
        with st.spinner("Sedang memprediksi jenis gambar..."):
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

            st.success("🎊 Prediksi Berhasil!")
            st.write("Hasil Prediksi:", label)
            st.write("Probabilitas:", f"{confidence:.2f}")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#3e2723;'>📸 Dibuat oleh <b>Mulya Syira</b> — Aplikasi Deteksi & Klasifikasi Gambar menggunakan Streamlit, YOLO, dan TensorFlow.</p>", unsafe_allow_html=True)
