import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import io

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Setup Page
# ==========================
st.set_page_config(page_title="Deteksi dan Klasifikasi Gambar", page_icon="📷", layout="wide")

# ==========================
# Custom CSS Pastel Cerah
# ==========================
st.markdown("""
<style>
body {
    background-color: #FFFDFB;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #FFF6F0;
    border-right: 1px solid #F3E9E2;
}
h1, h2, h3 { font-weight: 600; }
.main-title {
    font-size: 2.3rem;
    font-weight: 700;
    color: #2D2D2D;
    margin-bottom: 0.5rem;
}
.subtext {
    font-size: 1rem;
    color: #555;
    margin-bottom: 2rem;
}
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
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 1.5rem;
}
.result-card:hover {
    transform: scale(1.01);
    transition: 0.3s;
}
.footer {
    text-align: center;
    color: #666;
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
st.sidebar.header("📊 Pilih Mode Analisis")
menu = st.sidebar.radio("Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar dan biarkan AI menganalisis dengan cerdas dan lembut.")

# ==========================
# Layout Utama
# ==========================
col1, col2 = st.columns([2.3, 1.2])

with col1:
    st.markdown("<div class='main-title'>Deteksi dan Klasifikasi Gambar</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtext'>Gunakan teknologi <b>YOLOv8</b> untuk deteksi objek dan <b>CNN (TensorFlow)</b> untuk klasifikasi gambar. Unggah gambar, lalu lihat hasil analisis secara instan.</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

        # ==========================
        # YOLO DETECTION
        # ==========================
        if menu == "🎯 Deteksi Objek (YOLO)":
            st.subheader("⚙️ Pengaturan Deteksi")
            detect_option = st.checkbox("Aktifkan deteksi objek YOLO", value=True)

            if detect_option:
                with st.spinner("🔍 Sedang mendeteksi objek..."):
                    results = yolo_model(img, conf=0.25, iou=0.3)
                    boxes = results[0].boxes

                    # Perbaikan: tampilkan pesan jika tidak ada objek terdeteksi
                    if boxes is None or len(boxes) == 0:
                        st.warning("⚠️ Tidak ada objek yang terdeteksi.")
                    else:
                        result_img = results[0].plot(line_width=2, font_size=12)
                        st.markdown("<div class='result-card theme-mint'>", unsafe_allow_html=True)
                        st.image(result_img, caption="🎉 Hasil Deteksi", use_container_width=True)

                        # Data deteksi pakai pandas
                        data = boxes.data.cpu().numpy()
                        df = pd.DataFrame({
                            "Class": [results[0].names[int(cls)] for cls in data[:, 5]],
                            "Confidence": [round(conf, 2) for conf in data[:, 4]],
                            "X_min": data[:, 0],
                            "Y_min": data[:, 1],
                            "X_max": data[:, 2],
                            "Y_max": data[:, 3]
                        })
                        st.dataframe(df)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # Tombol Unduh Hasil Deteksi
                        buf = io.BytesIO()
                        Image.fromarray(result_img).save(buf, format="PNG")
                        st.download_button(
                            label="📥 Unduh Hasil Deteksi",
                            data=buf.getvalue(),
                            file_name="hasil_deteksi.png",
                            mime="image/png"
                        )

        # ==========================
        # CNN CLASSIFICATION
        # ==========================
        elif menu == "🧩 Klasifikasi Gambar":
            with st.spinner("🧠 Sedang memprediksi jenis gambar..."):
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

                st.markdown("<div class='result-card theme-lilac'>", unsafe_allow_html=True)
                st.success("✅ Prediksi Selesai")
                st.write("**Hasil Prediksi:**", label)
                st.write("**Probabilitas:**", f"{confidence:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)

                hasil_text = f"Hasil Prediksi: {label}\nProbabilitas: {confidence:.2f}"
                st.download_button(
                    label="📥 Unduh Hasil Analisis",
                    data=hasil_text,
                    file_name="hasil_klasifikasi.txt",
                    mime="text/plain"
                )
    else:
        st.markdown("<div class='upload-box'>📤 Silakan unggah gambar terlebih dahulu 💡</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 📈 Informasi Model")
    st.markdown("<div class='result-card theme-butter'><b>YOLOv8:</b> Deteksi objek cepat & akurat.<br><b>CNN:</b> Klasifikasi citra.<br><br><b>Confidence Threshold:</b> 0.25<br><b>IoU:</b> 0.3</div>", unsafe_allow_html=True)

    st.markdown("### 💡 Tips Penggunaan")
    st.markdown("<div class='result-card theme-peach'>🖼️ Gunakan gambar dengan resolusi jelas.<br>🤍 Coba bandingkan hasil YOLO dan CNN.<br>📊 Eksperimen dengan berbagai objek menarik!</div>", unsafe_allow_html=True)

# ==========================
# Footer
# ==========================
st.markdown("<div class='footer'>📷 Dibuat oleh <b>Mulya Syira</b> — Dashboard AI bertema pastel lembut yang elegan dan interaktif.</div>", unsafe_allow_html=True)
