import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import time

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="💖 PinkVision: Smart & Cute AI 💖",
    page_icon="🌸",
    layout="centered",
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
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")  # YOLO
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")  # CNN klasifikasi
    return yolo_model, classifier

with st.spinner("💫 Sedang memuat model... tunggu sebentar ya 💕"):
    yolo_model, classifier = load_models()
st.success("Model berhasil dimuat! 🌸")

# ==========================
# HEADER
# ==========================
st.title("🌷 PinkVision: Cute Image & Object Detector 🌷")
st.markdown(
    "Selamat datang di *PinkVision*! 💖<br>"
    "Aplikasi ini mendeteksi objek (YOLO) & klasifikasi gambar (CNN). "
    "Dirancang agar tetap imut tapi cerdas 🧠✨",
    unsafe_allow_html=True
)

# ==========================
# SIDEBAR MENU
# ==========================
st.sidebar.header("🎀 Pilihan Mode")
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"]
)
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar di bawah, lalu lihat keajaiban AI bekerja 💫")

# ==========================
# PROSES GAMBAR
# ==========================
uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Hitung edge density dan variasi warna
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    std_color = np.std(img_cv)

    st.write(f"🔍 **Edge density:** {edge_density:.5f}")
    st.write(f"🎨 **Variasi warna (std):** {std_color:.2f}")

    EDGE_THRESHOLD = 0.0001  # sangat sensitif biar ga blokir kaleng
    COLOR_STD_THRESHOLD = 15  # batas variasi warna (kaleng > 15, sel < 15)

    # ==========================
# MODE DETEKSI (YOLO)
# ==========================
if menu == "🎯 Deteksi Objek (YOLO)":
    with st.spinner("🐱 Sedang mendeteksi objek..."):

        # 1️⃣ Cegah gambar sel homogen
        if edge_density < EDGE_THRESHOLD and std_color < COLOR_STD_THRESHOLD:
            st.warning("🧫 Gambar terlalu homogen — kemungkinan gambar sel mikroskop.")
            st.stop()

        # 2️⃣ Jalankan YOLO
        results = yolo_model(img)
        boxes = results[0].boxes
        names = results[0].names

        allowed_labels = ["cocopham", "sprite"]
        filtered_boxes = []

        # 3️⃣ Filter hanya label yang diizinkan
        if boxes is not None and len(boxes.data) > 0:
            for box in boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = box
                label = names.get(int(cls_id), "Unknown")

                if label in allowed_labels and conf > 0.5:
                    filtered_boxes.append((label, conf, (x1, y1, x2, y2)))

        # 4️⃣ Jika tidak ada deteksi valid
        if len(filtered_boxes) == 0:
            st.warning("😿 Tidak ada objek Cocopham/Sprite terdeteksi.")
        else:
            # 5️⃣ Buat ulang anotasi hanya untuk label valid
            annotated_img = np.array(img_cv).copy()
            for (label, conf, (x1, y1, x2, y2)) in filtered_boxes:
                cv2.rectangle(annotated_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 105, 180), 3)
                cv2.putText(annotated_img, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 20, 147), 2)

            st.image(annotated_img, caption="🎀 Hasil Deteksi Valid 🎀", use_container_width=True)
            st.success("✨ Objek Cocopham/Sprite berhasil dideteksi!")
            st.dataframe({
                "Class": [f[0] for f in filtered_boxes],
                "Confidence": [round(float(f[1]), 2) for f in filtered_boxes],
            })

    # ==========================
    # MODE KLASIFIKASI
    # ==========================
    elif menu == "🧩 Klasifikasi Gambar":
        with st.spinner("🐶 Sedang memprediksi jenis gambar..."):
            img_resized = img.resize((128, 128))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            prediction = classifier.predict(img_array)
            if prediction.shape[-1] == 1:
                prob = prediction[0][0]
                label = "Uninfected" if prob > 0.5 else "Parasitized"
                confidence = prob if prob > 0.5 else 1 - prob
            else:
                labels = ["Parasitized", "Uninfected"]
                label = labels[np.argmax(prediction)]
                confidence = np.max(prediction)

            st.success("🎊 Prediksi Berhasil!")
            st.write(f"**Hasil Prediksi:** {label}")
            st.write(f"**Probabilitas:** {confidence:.2f}")

else:
    st.info("💡 Silakan unggah gambar terlebih dahulu.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow 💖")
