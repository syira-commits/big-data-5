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
st.set_page_config(page_title="CuteVision Smart Filter", page_icon="🌸", layout="centered")
st.title("🐾 CuteVision Smart Filter — Deteksi & Klasifikasi")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# MODE PILIHAN
# ==========================
menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

# ==========================
# PROSES
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="✨ Gambar Diupload ✨", use_container_width=True)

    # KONVERSI KE NUMPY
    img_cv = np.array(img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)

    # Hitung edge density untuk bantu mendeteksi gambar sel
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.sum(edges > 0) / edges.size

    # MODE DETEKSI
    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("🐱 Mendeteksi objek..."):
            results = yolo_model(img, conf=0.7)
            boxes = results[0].boxes
            data = boxes.data.cpu().numpy() if boxes is not None else np.array([])
            names = results[0].names

            allowed_labels = ["cocopham", "sprite"]
            filtered = []

            if len(data) > 0:
                for box in data:
                    x1, y1, x2, y2, conf, cls_id = box
                    label = names.get(int(cls_id), "Unknown")
                    if label in allowed_labels and conf > 0.6:
                        filtered.append((label, float(conf)))

            if len(filtered) == 0:
                st.warning("🚫 Tidak ada objek relevan (hanya mengenali Cocopham & Sprite).")
            else:
                annotated_img = results[0].plot()
                st.image(annotated_img, caption="🎉 Hasil Deteksi!", use_container_width=True)
                st.success("✅ Deteksi berhasil!")
                st.dataframe({
                    "Class": [f[0] for f in filtered],
                    "Confidence": [round(f[1], 2) for f in filtered],
                })

    # MODE KLASIFIKASI
    elif menu == "🧩 Klasifikasi Gambar":
        with st.spinner("🧠 Menganalisis gambar..."):
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

            # ====== FILTER NON-GAMBAR SEL ======
            # Jika edge_density terlalu rendah → kemungkinan bukan gambar sel
            # atau confidence terlalu rendah
            if edge_density > 0.07:
                st.warning("🧃 Gambar bukan gambar sel — tidak termasuk kategori Parasitized/Uninfected.")
            elif confidence < 0.5:
                st.warning("🤔 Model kurang yakin, kemungkinan ini bukan gambar sel.")
            else:
                st.success("🎊 Prediksi Berhasil!")
                st.write("Hasil Prediksi:", label)
                st.write("Probabilitas:", f"{confidence:.2f}")

else:
    st.info("💡 Silakan unggah gambar terlebih dahulu!")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — versi perbaikan dengan filter otomatis domain gambar 💖")
