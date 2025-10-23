import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# ==========================
# CONFIG PAGE
# ==========================
st.set_page_config(page_title="Dashboard Deteksi & Klasifikasi Gambar AI", page_icon="🤖", layout="centered")

# ==========================
# STYLE: BACKGROUND & WARNA
# ==========================
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #A1FFCE, #FAFFD1);
        background-attachment: fixed;
    }
    .stButton button {
        background-color: #FF4081;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #00BFA6;
        color: white;
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================
st.markdown("""
<div style="background-color:#FF9A9E; padding:12px; border-radius:10px; text-align:center; color:white; font-size:22px; font-weight:bold;">
    🌟 Dashboard Deteksi & Klasifikasi Gambar AI 🌟
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# SIDEBAR
# ==========================
st.sidebar.markdown("## 🧭 Navigasi")
menu = st.sidebar.radio("Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("📂 Unggah gambar, lalu jalankan deteksi atau klasifikasi.")
st.sidebar.markdown("🧑‍💻 Dibuat oleh **Mulya Syira**")

# ==========================
# DESKRIPSI MODE
# ==========================
if menu == "🎯 Deteksi Objek (YOLO)":
    st.info("✨ Mode ini digunakan untuk **mendeteksi objek 'Cocopham' dan 'Sprite'** dalam gambar.")
else:
    st.info("🧬 Mode ini digunakan untuk **mengklasifikasi gambar sel sebagai 'Uninfected' atau 'Parasitized'**.")

# ==========================
# UNGGAH GAMBAR
# ==========================
uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

    # MODE DETEKSI
    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("🔍 Sedang mendeteksi objek..."):
            img_cv = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            if edge_density > 0.12:
                st.warning("📄 Gambar ini terdeteksi sebagai teks/grafik — tidak ada objek relevan 💤")
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
                        if (
                            label in allowed_labels
                            and conf > 0.6
                            and 0.02 < area / img_area < 0.8
                        ):
                            filtered.append((label, float(conf)))

                if len(filtered) == 0:
                    st.warning("😿 Tidak ada objek terdeteksi (YOLO hanya mendeteksi **Cocopham & Sprite**).")
                else:
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="🎉 Hasil Deteksi", use_container_width=True)
                    st.subheader("📋 Detail Deteksi")
                    st.dataframe({
                        "Class": [f[0] for f in filtered],
                        "Confidence": [round(f[1], 2) for f in filtered],
                    })
                    st.balloons()

    # MODE KLASIFIKASI
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

            st.success("✅ Prediksi Berhasil!")
            st.write(f"**Hasil Prediksi:** {label}")
            st.write(f"**Probabilitas:** {confidence:.2f}")
            st.markdown("### 🔮 Tingkat Keyakinan Model:")
            st.progress(float(confidence))
            st.snow()

else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div style="background-color:#A1C4FD; padding:10px; border-radius:10px; text-align:center; color:white;">
    💻 Dibuat oleh <b>Mulya Syira</b> | Streamlit + YOLO + TensorFlow
</div>
""", unsafe_allow_html=True)
