import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tensorflow as tf

# ========================== CONFIG PAGE ==========================
st.set_page_config(
    page_title="AI Dashboard - Mulya Syira",
    page_icon="🌸",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FFDEE9, #B5FFFC);
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background: #FFF5E4;
}
h1, h2, h3, h4 {
    color: #5E548E;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ========================== HEADER ==========================
st.markdown("""
<div style="background-color:#FFC8DD; padding:15px; border-radius:15px; text-align:center; color:white; font-size:24px; font-weight:bold;">
🌸 AI Dashboard - Deteksi & Klasifikasi Gambar 🌸
</div>
""", unsafe_allow_html=True)

st.write("")

# ========================== SIDEBAR ==========================
st.sidebar.header("🎨 Mode Tampilan")
theme = st.sidebar.radio("Pilih warna tema:", ["🌸 Pink Pastel", "🍑 Peach Pastel", "🩵 Biru Pastel"])

if theme == "🌸 Pink Pastel":
    bg = "linear-gradient(135deg, #FFDEE9, #B5FFFC)"
elif theme == "🍑 Peach Pastel":
    bg = "linear-gradient(135deg, #FAD0C4, #FFD1FF)"
else:
    bg = "linear-gradient(135deg, #A1C4FD, #C2E9FB)"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: {bg};
    background-attachment: fixed;
}}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("📂 Menu Utama")
menu = st.sidebar.radio("Pilih halaman:", ["🧩 Klasifikasi Gambar", "🎯 Deteksi Objek"])

st.sidebar.markdown("---")
st.sidebar.header("🕒 Riwayat")
if "history" not in st.session_state:
    st.session_state["history"] = []

if len(st.session_state["history"]) > 0:
    for i, item in enumerate(st.session_state["history"]):
        st.sidebar.write(f"{i+1}. {item['File']} → {item['Hasil']}")
else:
    st.sidebar.write("Belum ada riwayat.")

st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Tentang")
st.sidebar.info("""
Aplikasi ini dibuat oleh **Mulya Syira** 💕  
Menggunakan model **YOLO** dan **TensorFlow**  
untuk mendeteksi serta mengklasifikasi gambar.
""")

# ========================== MAIN PAGE ==========================
if menu == "🧩 Klasifikasi Gambar":
    st.header("🧩 Klasifikasi Gambar dengan Model CNN")

    uploaded_file = st.file_uploader("Unggah gambar di sini:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        model = tf.keras.models.load_model("model_klasifikasi.h5")  # ganti sesuai nama modelmu

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        label = np.argmax(prediction, axis=1)[0]

        st.success(f"✅ Hasil Klasifikasi: **{label}**")
        st.progress(float(np.max(prediction)))

        st.session_state["history"].append({"File": uploaded_file.name, "Hasil": str(label)})

        st.snow()  # efek lembut

elif menu == "🎯 Deteksi Objek":
    st.header("🎯 Deteksi Objek dengan YOLO")

    uploaded_file = st.file_uploader("Unggah gambar untuk deteksi:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_container_width=True)

        model = YOLO("yolov8n.pt")  # ganti model YOLO sesuai milikmu
        results = model(img)
        res_plotted = results[0].plot()  # hasil deteksi dengan bounding box

        st.image(res_plotted, caption="Hasil Deteksi", use_container_width=True)
        st.balloons()

        st.session_state["history"].append({"File": uploaded_file.name, "Hasil": "Objek Terdeteksi"})

# ========================== FOOTER ==========================
st.markdown("""
<br><br>
<div style="background-color:#A2D2FF; padding:12px; border-radius:12px; text-align:center; color:#fff;">
💻 Dibuat oleh Mulya Syira | 2025
</div>
""", unsafe_allow_html=True)
