import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Mulya_Syira_Laporan4.pt")  # pastikan tanpa spasi di nama file
    classifier = tf.keras.models.load_model("model/Mulya_Syira_Laporan2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Styling
# ==========================
st.set_page_config(page_title="AI Vision App", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #f9fbff, #eaf1ff);
    }
    h1 {
        text-align: center;
        color: #1c4587;
        font-weight: 700;
    }
    .uploadedImage {
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# UI
# ==========================
st.title("🧠 AI Vision App")
st.subheader("Image Classification & Object Detection")

menu = st.sidebar.radio("📂 Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("📸 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True, output_format="PNG")

    st.markdown("---")
    progress = st.progress(0)
    progress.progress(30)

    if menu == "Deteksi Objek (YOLO)":
        st.info("🔍 Sedang melakukan deteksi objek...")
        results = yolo_model(img)
        progress.progress(80)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True, output_format="PNG")

        # tampilkan hasil dalam tabel sederhana
        if hasattr(results[0], "boxes"):
            st.write("### 📊 Detail Deteksi:")
            data = []
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                data.append({"Kelas": yolo_model.names[cls], "Confidence": round(conf, 3)})
            st.table(data)

        progress.progress(100)
        st.success("✅ Deteksi selesai!")

    elif menu == "Klasifikasi Gambar":
        st.info("🧠 Sedang mengklasifikasi gambar...")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        progress.progress(70)
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)

        # ubah sesuai label model kamu
        class_labels = ["Kucing", "Anjing", "Burung"]
        predicted_label = class_labels[class_index]

        st.write("### 🏷️ Hasil Prediksi:", predicted_label)
        st.write("Probabilitas:", f"{np.max(prediction)*100:.2f}%")

        progress.progress(100)
        st.success("✅ Klasifikasi selesai!")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("Developed by **Mulya Syira** | Powered by YOLOv8 & TensorFlow")
