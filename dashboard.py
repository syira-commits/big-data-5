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
st.set_page_config(page_title="Deteksi Objek dan Klasifikasi Gambar", page_icon="🐱", layout="centered")

# Warna background pastel cerah
st.markdown(
    """
    <style>
        /* Background seluruh halaman */
        .css-18e3th9, .css-1d391kg {
            background-color: #E6E0F8;  /* Ungu pastel cerah */
        }

        /* Tombol Streamlit */
        .stButton>button {
            background-color: #DCCFFF; /* Button ungu pastel */
            color: black;
        }

        /* Sidebar */
        .css-1d391kg [data-testid="stSidebar"] {
            background-color: #F3E8FF; /* Sidebar ungu pastel lebih terang */
        }

        /* Tambah jarak atas agar rapi */
        .css-1d391kg {
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🐾 Deteksi Objek dan Klasifikasi Gambar")
st.markdown("Aplikasi ini menggunakan YOLO untuk deteksi objek dan CNN (TensorFlow) untuk klasifikasi gambar.")

# ==========================
# Sidebar Interaktif
# ==========================
menu = st.sidebar.radio("🌈 Pilih Mode:", ["🎯 Deteksi Objek (YOLO)", "🧩 Klasifikasi Gambar"])
st.sidebar.markdown("---")

# Slider ukuran gambar untuk klasifikasi
if menu == "🧩 Klasifikasi Gambar":
    img_size = st.sidebar.slider("🖼️ Ukuran Gambar untuk Klasifikasi", 64, 256, 128, 16)

# Tips penggunaan
st.sidebar.info("💡 Tips:\n- Deteksi: gunakan gambar Cocopham & Sprite\n- Klasifikasi: gunakan gambar sel Uninfected & Parasitized")

# Riwayat upload
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.subheader("📂 Riwayat Upload")
if st.session_state.history:
    for img_name in st.session_state.history[-5:]:  # 5 terakhir
        st.sidebar.write(img_name)

uploaded_file = st.file_uploader("📤 Unggah Gambar di Sini", type=["jpg", "jpeg", "png"])

# ==========================
# Proses Gambar
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="✨ Gambar yang Diupload ✨", use_container_width=True)

    # Simpan ke history
    st.session_state.history.append(uploaded_file.name)

    # --------------------------
    # MODE DETEKSI OBJEK
    # --------------------------
    if menu == "🎯 Deteksi Objek (YOLO)":
        st.info("ℹ️ Gunakan gambar Cocopham & Sprite agar hasil lebih akurat!")
        with st.spinner("🐱 Sedang mendeteksi objek... tunggu sebentar ya!"):
            img_cv = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

            if edge_density > 0.12:
                st.warning("📄 Gambar terdeteksi sebagai teks/grafik — tidak ada objek relevan 💤")
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
                    st.warning("😿 Tidak ada objek terdeteksi (hanya Cocopham & Sprite)")
                else:
                    annotated_img = results[0].plot()
                    st.image(annotated_img, caption="🎉 Hasil Deteksi!", use_container_width=True)
                    st.subheader("📋 Detail Deteksi")
                    st.dataframe({
                        "Class": [f[0] for f in filtered],
                        "Confidence": [round(f[1],2) for f in filtered],
                    })
                    # Efek balon
                    st.balloons()

    # --------------------------
    # MODE KLASIFIKASI GAMBAR
    # --------------------------
    elif menu == "🧩 Klasifikasi Gambar":
        st.info("ℹ️ Gunakan gambar sel: Uninfected & Parasitized agar hasil lebih akurat!")
        with st.spinner("🐶 Sedang memprediksi jenis gambar..."):
            img_resized = img.resize((img_size, img_size))
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

            # Efek salju
            st.snow()

else:
    st.info("Silakan unggah gambar terlebih dahulu 💡")

# ==========================
# Footer
# ==========================
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — Dashboard lucu tapi cerdas menggunakan Streamlit, YOLO & TensorFlow.")
