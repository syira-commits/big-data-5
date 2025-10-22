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
    # Model deteksi custom dan umum
    yolo_custom = YOLO("model/Mulya Syira_Laporan 4.pt")  # model custom kamu
    yolo_general = YOLO("yolov8n.pt")  # model umum dari COCO dataset
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # model CNN
    return yolo_custom, yolo_general, classifier

yolo_custom, yolo_general, classifier = load_models()

# ==========================
# Setup Page
# ==========================
st.set_page_config(page_title="Deteksi dan Klasifikasi Gambar", page_icon="📷", layout="wide")

# ==========================
# Custom CSS (Pastel Theme)
# ==========================
st.markdown("""
<style>
body {
    background-color: #FFFDFB;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #FFF7F3;
    border-right: 1px solid #F2E7DC;
}
.main-title {font-size: 2.3rem; font-weight: 700; color: #2D2D2D;}
.subtext {font-size: 1rem; color: #555; margin-bottom: 2rem;}
.upload-box {
    background-color: #FFFFFF;
    border: 2px dashed #E5E7EB;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}
.result-card {
    background-color: #FFFFFF;
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
.theme-peach {border-left: 8px solid #FFD8C2;}
.theme-lilac {border-left: 8px solid #E4D7FF;}
.theme-mint {border-left: 8px solid #C9F5D7;}
.theme-butter {border-left: 8px solid #FFF5B8;}
[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #FFD8C2, #E4D7FF);
    color: #2D2D2D;
    border: none;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-weight: 600;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #FFC6AA, #DCC8FF);
    transform: scale(1.02);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# Sidebar
# ==========================
st.sidebar.header("📊 Pengaturan Analisis")

menu = st.sidebar.radio("Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])

st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar dan biarkan AI menganalisis dengan cerdas dan lembut.")

# Pengaturan deteksi YOLO
st.sidebar.markdown("### ⚙ Pengaturan YOLO")
model_choice = st.sidebar.selectbox("Pilih Model YOLO:", ["Custom (Mulya Syira_Laporan 4.pt)", "Umum (yolov8n.pt)"])
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.55, 0.05)
iou_thresh = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.4, 0.05)

# ==========================
# Layout Utama
# ==========================
col1, col2 = st.columns([2.3, 1.2])

with col1:
    st.markdown("<div class='main-title'>Deteksi dan Klasifikasi Gambar</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Gunakan <b>YOLOv8</b> untuk deteksi objek dan <b>CNN TensorFlow</b> untuk klasifikasi. Cukup unggah gambar dan lihat hasil analisisnya.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

        # ==========================
        # MODE DETEKSI OBJEK (YOLO)
        # ==========================
        if menu == "🎯 Deteksi Objek (YOLO)":
            detect_option = st.checkbox("Aktifkan Deteksi YOLO", value=True)

            if detect_option:
                with st.spinner("🔍 Sedang mendeteksi objek..."):
                    img_array = np.array(img)

                    # Pilih model sesuai sidebar
                    model_yolo = yolo_custom if "Custom" in model_choice else yolo_general

                    # Jalankan deteksi
                    results = model_yolo(img_array, conf=conf_thresh, iou=iou_thresh)
                    boxes = results[0].boxes

                    if boxes is None or len(boxes) == 0:
                        st.warning("🚫 Tidak ada objek yang terdeteksi.")
                        st.image(img, caption="Hasil: Tidak ada deteksi", use_container_width=True)
                    else:
                        data = boxes.data.cpu().numpy()
                        data = data[data[:, 4] > conf_thresh]

                        if len(data) == 0:
                            st.warning(f"🚫 Tidak ada objek dengan confidence ≥ {conf_thresh:.2f}.")
                            st.image(img, caption="Hasil: Tidak ada deteksi valid", use_container_width=True)
                        else:
                            st.success("✅ Objek terdeteksi!")
                            st.image(results[0].plot(), caption="Hasil Deteksi YOLO", use_container_width=True)

                            st.dataframe({
                                "Class": [results[0].names[int(cls)] for cls in data[:, 5]],
                                "Confidence": [round(conf, 2) for conf in data[:, 4]],
                                "X_min": data[:, 0],
                                "Y_min": data[:, 1],
                                "X_max": data[:, 2],
                                "Y_max": data[:, 3]
                            })
            else:
                st.info("🕹 Deteksi YOLO dimatikan.")

        # ==========================
        # MODE KLASIFIKASI GAMBAR
        # ==========================
        elif menu == "🧩 Klasifikasi Gambar":
            with st.spinner("🧠 Sedang memprediksi jenis gambar..."):
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

                st.markdown("<div class='result-card theme-lilac'>", unsafe_allow_html=True)
                st.success("✅ Prediksi Selesai")
                st.write("Hasil Prediksi:", label)
                st.write("Probabilitas:", f"{confidence:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<div class='upload-box'>📤 Silakan unggah gambar terlebih dahulu 💡</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Informasi Model")
    st.markdown(f"""
    <div class='result-card theme-butter'>
    <b>Model:</b> {model_choice}<br>
    <b>Confidence Threshold:</b> {conf_thresh}<br>
    <b>IoU:</b> {iou_thresh}<br><br>
    <b>YOLOv8:</b> Deteksi objek cepat & akurat<br>
    <b>CNN:</b> Klasifikasi citra
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Tips Penggunaan")
    st.markdown("<div class='result-card theme-peach'>🖼 Gunakan gambar dengan resolusi jelas.<br>🤍 Coba bandingkan hasil YOLO Custom dan Umum.<br>📊 Eksperimen dengan berbagai objek menarik!</div>", unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<div class='footer'>📷 Dibuat oleh <b>Mulya Syira</b> — Dashboard AI bertema pastel lembut yang elegan dan interaktif.</div>", unsafe_allow_html=True)
