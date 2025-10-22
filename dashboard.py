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
# Tampilan Dashboard
# ==========================
st.set_page_config(page_title="CuteVision App", page_icon="🐱", layout="centered")

st.title("🐾 CuteVision App — Deteksi & Klasifikasi Gambar Lucu")
st.markdown("Aplikasi ini menggunakan *YOLO* untuk mendeteksi objek dan *CNN (TensorFlow)* untuk mengklasifikasi gambar. Cocok untuk yang suka hal-hal lucu tapi tetap cerdas!")

menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")
st.sidebar.info("Unggah gambar dan lihat keajaiban AI bekerja!")

uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

    # ==========================
    # MODE DETEKSI OBJEK
    # ==========================
    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("🐱 Sedang mendeteksi objek... tunggu sebentar ya!"):
            results = yolo_model(img)
            data = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else np.array([])

            # Ambil daftar label dari model
            names = results[0].names
            allowed_labels = ["cocopham", "sprite"]

            # Filter hanya objek yang diizinkan (cocopham/sprite)
            filtered = []
            if len(data) > 0:
                for box in data:
                    cls_id = int(box[5])
                    conf = float(box[4])
                    label = names.get(cls_id, "Unknown")
                    if label in allowed_labels and conf > 0.5:
                        filtered.append((label, conf, box))

            # === Tampilkan hasil ===
            if len(filtered) == 0:
                st.warning("Tidak ada objek terdeteksi 😿 (hanya mendeteksi Cocopham & Sprite)")
            else:
                # Gambar hasil deteksi (hanya untuk label yang lolos filter)
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
            st.write("*Hasil Prediksi:*", label)
            st.write("*Probabilitas:*", f"{confidence:.2f}")

else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

# ==========================
# Footer lucu
# ==========================
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow.")
