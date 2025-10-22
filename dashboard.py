import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

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
# Tampilan Dashboard
# ==========================
st.set_page_config(page_title="Deteksi & Klasifikasi Gambar", page_icon="🐾", layout="centered")

# Warna pastel cerah via CSS
st.markdown("""
    <style>
        body {
            background-color: #FFF8FB;
        }
        .stApp {
            background: linear-gradient(180deg, #FFEFF6 0%, #F8F9FF 100%);
            color: #4A4A4A;
        }
        .stButton>button {
            background-color: #FFD8E0;
            color: #333;
            border-radius: 10px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #FFB6C1;
            color: white;
        }
        .css-1v0mbdj, .css-1d391kg {
            background-color: transparent !important;
        }
        h1, h2, h3 {
            color: #FF6B8B;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🐾 Deteksi & Klasifikasi Gambar")
st.markdown("Gunakan model **YOLO** untuk deteksi objek dan **CNN** untuk klasifikasi. Tampilan ini dibuat lembut dengan warna pastel yang cerah!")

menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Fungsi Buat PDF
# ==========================
def create_pdf(result_text, filename="hasil_analisis.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "Laporan Analisis Gambar")
    c.setFont("Helvetica", 12)
    c.drawString(100, 770, f"Tanggal: {datetime.now().strftime('%d %B %Y %H:%M')}")
    c.drawString(100, 740, "Hasil Analisis:")
    text_object = c.beginText(100, 720)
    for line in result_text.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar Diupload ✨", use_container_width=True)

    if menu == "🎯 Deteksi Objek (YOLO)":
        with st.spinner("Sedang mendeteksi objek..."):
            results = yolo_model(img, conf=0.1, iou=0.3)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                result_img = results[0].plot(line_width=2, font_size=12)
                st.image(result_img, caption="🎉 Hasil Deteksi!", use_container_width=True)

                data = boxes.data.cpu().numpy()
                deteksi_df = {
                    "Class": [results[0].names[int(cls)] for cls in data[:, 5]],
                    "Confidence": [round(conf, 2) for conf in data[:, 4]]
                }
                st.dataframe(deteksi_df)

                result_text = "=== HASIL DETEKSI YOLO ===\n"
                for i, (cls, conf) in enumerate(zip(deteksi_df["Class"], deteksi_df["Confidence"])):
                    result_text += f"{i+1}. {cls} - Confidence: {conf}\n"

                pdf_buffer = create_pdf(result_text)
                st.download_button("💾 Unduh Hasil Analisis (PDF)", data=pdf_buffer,
                                   file_name="hasil_deteksi.pdf", mime="application/pdf")
            else:
                st.warning("Tidak ada objek yang terdeteksi.")

    elif menu == "🧩 Klasifikasi Gambar":
        with st.spinner("Sedang mengklasifikasi gambar..."):
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

            st.success("🎊 Prediksi Selesai!")
            st.write("**Hasil Prediksi:**", label)
            st.write("**Probabilitas:**", f"{confidence:.2f}")

            result_text = f"""=== HASIL KLASIFIKASI CNN ===
Label: {label}
Probabilitas: {confidence:.2f}"""

            pdf_buffer = create_pdf(result_text)
            st.download_button("💾 Unduh Hasil Analisis (PDF)", data=pdf_buffer,
                               file_name="hasil_klasifikasi.pdf", mime="application/pdf")
else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — versi pastel cerah dengan fitur unduh hasil analisis")
