import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from PIL import Image
import numpy as np
import cv2
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Deteksi Objek dan Klasifikasi Gambar",
                   page_icon="📸",
                   layout="wide")

# -------------------- STYLE (COKLAT PASTEL) --------------------
st.markdown(
    """
    <style>
    :root{
        --bg: #f7efe6;
        --card: #efe1d2;
        --accent: #a67c52;
        --muted: #6b4f3a;
        --soft: #f3e8df;
    }
    body { background-color: var(--bg); }
    .main-title { text-align:center; color:var(--muted); font-size:34px; font-weight:700; margin-bottom:0px; }
    .sub-text { text-align:center; color:var(--accent); font-size:15px; margin-top:4px; margin-bottom:18px; }
    .card {
        border-radius:16px;
        padding:18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.6), rgba(239,225,210,0.9));
        box-shadow: 0 6px 18px rgba(107,79,58,0.08);
        text-align:center;
    }
    .big-btn {
        background-color: transparent;
        border: 2px solid rgba(166,124,82,0.95);
        color: var(--muted);
        padding: 10px 18px;
        border-radius: 12px;
        font-weight:600;
        cursor:pointer;
    }
    .small-muted { color:#7b5e48; font-size:13px; }
    .upload-box{ border:2px dashed rgba(166,124,82,0.25); padding:12px; border-radius:12px; background:var(--soft); }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Deteksi Objek dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Pilih mode di bawah — Deteksi (Cocopham & Sprite) atau Klasifikasi (Parasitized & Uninfected)</div>', unsafe_allow_html=True)

# -------------------- SESSION STATE / NAV --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"  # home / detect / classify

def go_home():
    st.session_state.page = "home"

def go_detect():
    st.session_state.page = "detect"

def go_classify():
    st.session_state.page = "classify"

# -------------------- LOAD MODELS (cached) --------------------
@st.cache_resource
def load_models():
    # Will raise if files missing -> caught below
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

# Try to load; if fail, show error but allow UI to show (user can still view home)
models_loaded = True
try:
    yolo_model, classifier = load_models()
except Exception as e:
    models_loaded = False
    load_err = e

# -------------------- UTIL --------------------
def load_builtin_image(name):
    path = os.path.join("images", name)
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None

# -------------------- HOME PAGE --------------------
if st.session_state.page == "home":
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Pilih Mode", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            # DETEKSI CARD
            img_path = "images/cocopham.jpg" if os.path.exists("images/cocopham.jpg") else ""
            if img_path:
                st.image(img_path, use_column_width=True)
            st.markdown("**Deteksi Objek**", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Gunakan model YOLO untuk mendeteksi Cocopham & Sprite.</div>", unsafe_allow_html=True)
            st.button("Buka Deteksi →", key="btn_detect", on_click=go_detect)
        with col_b:
            # KLASIFIKASI CARD
            img_path2 = "images/parasitized.jpg" if os.path.exists("images/parasitized.jpg") else ""
            if img_path2:
                st.image(img_path2, use_column_width=True)
            st.markdown("**Klasifikasi Gambar**", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Klasifikasikan gambar sel: Parasitized / Uninfected.</div>", unsafe_allow_html=True)
            st.button("Buka Klasifikasi →", key="btn_classify", on_click=go_classify)
        st.markdown("</div>", unsafe_allow_html=True)

    # if models not loaded, show a gentle warning
    if not models_loaded:
        st.warning(f"Model tidak dapat dimuat: {load_err}. Pastikan folder `model/` berisi file `.pt` dan `.h5` yang sesuai.")

# -------------------- DETECTION PAGE --------------------
elif st.session_state.page == "detect":
    st.button("⬅ Kembali ke Menu Utama", on_click=go_home)
    st.markdown("<h3 style='color:#6b4f3a;'>🎯 Deteksi Objek (Cocopham & Sprite)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Pilih gambar bawaan atau unggah gambar sendiri untuk dideteksi.</div>", unsafe_allow_html=True)
    st.write("")

    # pilihan gambar bawaan atau upload
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Gambar Tersedia**", unsafe_allow_html=True)
        built_choice = st.radio("Pilih contoh:", ("-- pilih --", "Cocopham", "Sprite"))
        st.markdown("<br>atau</br>", unsafe_allow_html=True)
        uploaded_det = st.file_uploader("📤 Upload gambar untuk deteksi", type=["jpg", "jpeg", "png"], key="upload_detect")
        st.markdown("</div>", unsafe_allow_html=True)

    # show & process in right column
    with col2:
        selected_image = None
        if built_choice != "-- pilih --":
            name_map = {"Cocopham": "cocopham.jpg", "Sprite": "sprite.jpg"}
            selected_image = load_builtin_image(name_map.get(built_choice))
        elif uploaded_det is not None:
            selected_image = Image.open(uploaded_det).convert("RGB")

        if selected_image is None:
            st.info("Pilih gambar contoh di kiri atau upload gambar untuk memulai deteksi.")
        else:
            st.image(selected_image, caption="Gambar untuk dideteksi", use_container_width=True)

            if not models_loaded:
                st.error("Model tidak tersedia. Pastikan file `.pt` ada di folder `model/`.")
            else:
                with st.spinner("🐱 Mendeteksi objek..."):
                    # call YOLO model exactly as sebelumnya (preserve logic)
                    results = yolo_model(selected_image, conf=0.7)
                    boxes = results[0].boxes
                    data = boxes.data.cpu().numpy() if boxes is not None else np.array([])
                    names = results[0].names

                    allowed_labels = ["cocopham", "sprite"]
                    filtered = []

                    if len(data) > 0:
                        for box in data:
                            # x1, y1, x2, y2, conf, cls_id
                            x1, y1, x2, y2, conf, cls_id = box
                            label = names.get(int(cls_id), "Unknown")
                            if label in allowed_labels and conf > 0.6:
                                filtered.append((label, float(conf)))

                    if len(filtered) == 0:
                        st.warning("🚫 Tidak ada objek relevan (hanya mengenali Cocopham & Sprite).")
                    else:
                        # annotated image
                        try:
                            annotated = results[0].plot()  # ultralytics usually returns numpy array for plot
                            if isinstance(annotated, np.ndarray):
                                annotated = Image.fromarray(annotated)
                            st.image(annotated, caption="🎉 Hasil Deteksi!", use_column_width=True)
                        except Exception:
                            # fallback: just show original
                            st.image(selected_image, caption="Hasil Deteksi (fallback)", use_column_width=True)

                        st.success("✅ Deteksi berhasil!")
                        st.dataframe({
                            "Class": [f[0] for f in filtered],
                            "Confidence": [round(f[1], 2) for f in filtered],
                        })

# -------------------- CLASSIFICATION PAGE --------------------
elif st.session_state.page == "classify":
    st.button("⬅ Kembali ke Menu Utama", on_click=go_home)
    st.markdown("<h3 style='color:#6b4f3a;'>🧩 Klasifikasi Gambar (Parasitized & Uninfected)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Pilih gambar bawaan atau unggah gambar sendiri untuk diklasifikasikan.</div>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Gambar Tersedia**", unsafe_allow_html=True)
        built_choice = st.radio("Pilih contoh:", ("-- pilih --", "Parasitized", "Uninfected"))
        st.markdown("<br>atau</br>", unsafe_allow_html=True)
        uploaded_clf = st.file_uploader("📤 Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="upload_clf")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        selected_image = None
        if built_choice != "-- pilih --":
            name_map = {"Parasitized": "parasitized.jpg", "Uninfected": "uninfected.jpg"}
            selected_image = load_builtin_image(name_map.get(built_choice))
        elif uploaded_clf is not None:
            selected_image = Image.open(uploaded_clf).convert("RGB")

        if selected_image is None:
            st.info("Pilih gambar contoh di kiri atau upload gambar untuk memulai klasifikasi.")
        else:
            st.image(selected_image, caption="Gambar untuk diklasifikasikan", use_container_width=True)

            if not models_loaded:
                st.error("Model klasifikasi tidak tersedia. Pastikan file `.h5` ada di folder `model/`.")
            else:
                with st.spinner("🧠 Menganalisis gambar..."):
                    img_resized = selected_image.resize((128, 128))
                    img_array = kimage.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    prediction = classifier.predict(img_array)

                    if prediction.shape[-1] == 1:
                        prob = float(prediction[0][0])
                        label = "Uninfected" if prob > 0.5 else "Parasitized"
                        confidence = prob if prob > 0.5 else 1 - prob
                    else:
                        class_names = ["Parasitized", "Uninfected"]
                        class_index = int(np.argmax(prediction))
                        label = class_names[class_index]
                        confidence = float(np.max(prediction))

                    # FILTER NON-GAMBAR SEL (sama logika)
                    img_cv = np.array(selected_image)
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 80, 160)
                    edge_density = np.sum(edges > 0) / edges.size

                    if edge_density > 0.07:
                        st.warning("🧃 Gambar bukan gambar sel — tidak termasuk kategori Parasitized/Uninfected.")
                    elif confidence < 0.5:
                        st.warning("🤔 Model kurang yakin, kemungkinan ini bukan gambar sel.")
                    else:
                        st.success("🎊 Prediksi Berhasil!")
                        st.write("Hasil Prediksi:", label)
                        st.write("Probabilitas:", f"{confidence:.2f}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — CuteVision Pastel Edition (tema coklat pastel)")
