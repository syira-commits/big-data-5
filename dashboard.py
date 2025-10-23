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
st.set_page_config(
    page_title="🐾 CuteVision App",
    page_icon="🐱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM STYLE (WARNA NGEJRENG)
# ==========================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
            color: #333333;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #FF4081;
            text-shadow: 1px 1px 2px #00000030;
        }
        .subtitle {
            text-align: center;
            color: #00BFA6;
            font-size: 18px;
            margin-bottom: 25px;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #FFB6C1, #FFF5EE);
            color: black;
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
    </style>
""", unsafe_allow_html=True)

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
# TAMPILAN DASHBOARD
# ==========================
st.markdown('<h1 class="title">🐾 CuteVision App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi dan Klasifikasi Gambar Lucu dengan Kecerdasan Buatan 💖</p>', unsafe_allow_html=True)

menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")

# Deskripsi tambahan
if menu == "🎯 Deteksi Objek (YOLO)":
    st.info("✨ Mode ini digunakan untuk **mendeteksi objek 'Cocopham' dan 'Sprite'** dalam gambar.")
else:
    st.info("🧬 Mode ini digunakan untuk **mengklasifikasi gambar sel sebagai 'Uninfected' atau 'Parasitized'**.")

uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES GAMBAR
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK
    # ==========================
    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("🐱 Sedang mendeteksi objek... tunggu sebentar ya!"):
            # Konversi ke OpenCV
            img_cv = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            if edge_density > 0.12:
                st.warning("📄 Gambar terdeteksi sebagai teks/grafik — tidak ada objek yang relevan 💤")
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
                            filtered.append((label, float(conf), (x1, y1, x2, y2)))

                if len(filtered) == 0:
                    st.warning("😿 Tidak ada objek terdeteksi.\nYOLO hanya mendeteksi **Cocopham & Sprite**.")
                else:
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="🎉 Hasil Deteksi!", use_container_width=True)

                    st.subheader("📋 Detail Deteksi")
                    st.dataframe({
                        "Class": [f[0] for f in filtered],
                        "Confidence": [round(f[1], 2) for f in filtered],
                    })

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

            st.success("🎊 Prediksi Berhasil!")
            st.markdown(f"**🧬 Hasil Prediksi:** `{label}`")
            st.markdown(f"**🔮 Probabilitas:** `{confidence:.2f}`")

else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

# ==========================
# FOOTER LUCU
# ==========================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#FF4081;'>🐾 Dibuat oleh <b>Mulya Syira</b> — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow 💖</div>",
    unsafe_allow_html=True
)
