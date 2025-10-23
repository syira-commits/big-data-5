import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from PIL import Image
import numpy as np
import cv2
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Deteksi Objek dan Klasifikasi Gambar",
    page_icon="📸",
    layout="wide"
)

# -------------------- STYLE (COKLAT PASTEL GRADIENT) --------------------
st.markdown(
    """
    <style>
    :root{
        --bg-start: #f7efe6;   /* krem */
        --bg-end: #efe0d6;     /* coklat very light */
        --card: #f3e6dd;
        --accent: #8a5a3f;
        --muted: #6b4f3a;
        --soft: rgba(138,90,63,0.08);
    }
    html, body, [class*="css"]  {
        background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%) !important;
    }
    .main-title { text-align:center; color:var(--muted); font-size:36px; font-weight:700; margin-bottom:4px; }
    .sub-text { text-align:center; color:var(--accent); font-size:15px; margin-top:0px; margin-bottom:18px; }
    .card {
        border-radius:16px;
        padding:18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(243,230,221,0.95));
        box-shadow: 0 8px 22px rgba(107,79,58,0.06);
        text-align:center;
    }
    .small-muted { color:#7b5e48; font-size:13px; }
    .upload-box{ border:2px dashed rgba(138,90,63,0.18); padding:12px; border-radius:12px; background:rgba(255,255,255,0.6); }
    .back-btn {background:none; border:0; color:var(--accent); font-weight:600; cursor:pointer;}
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Deteksi Objek dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Pilih mode lalu pilih gambar bawaan atau unggah gambar sendiri untuk diproses</div>', unsafe_allow_html=True)

# -------------------- NAVIGATION --------------------
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
def load_models_safe():
    """
    Try to load both YOLO and classifier.
    If model files missing or cannot load, this function will raise and the caller will handle it.
    """
    yolo_model = YOLO("model/Mulya Syira_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Mulya Syira_Laporan2.h5")
    return yolo_model, classifier

models_loaded = True
yolo_model = None
classifier = None
load_err = None
try:
    yolo_model, classifier = load_models_safe()
except Exception as e:
    models_loaded = False
    load_err = e

# -------------------- UTIL --------------------
IMAGES_DIR = "images"
def builtin_path(name):
    return os.path.join(IMAGES_DIR, name) if name else None

def load_builtin_image(name):
    p = builtin_path(name)
    if p and os.path.exists(p):
        return Image.open(p).convert("RGB")
    return None

# -------------------- HOME PAGE --------------------
if st.session_state.page == "home":
    # center card with two choices
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Pilih Mode", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            # Deteksi card preview
            tmp_img = load_builtin_image("cocopham.jpg")
            if tmp_img:
                st.image(tmp_img, use_column_width=True)
            st.markdown("**Deteksi Objek**", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Gunakan model YOLO untuk mendeteksi Cocopham & Sprite.</div>", unsafe_allow_html=True)
            st.button("Buka Deteksi →", key="go_detect_btn", on_click=go_detect)
        with col_b:
            # Klasifikasi card preview
            tmp_img2 = load_builtin_image("parasitized.jpg")
            if tmp_img2:
                st.image(tmp_img2, use_column_width=True)
            st.markdown("**Klasifikasi Gambar**", unsafe_allow_html=True)
            st.markdown("<div class='small-muted'>Klasifikasikan gambar sel: Parasitized / Uninfected.</div>", unsafe_allow_html=True)
            st.button("Buka Klasifikasi →", key="go_class_btn", on_click=go_classify)
        st.markdown("</div>", unsafe_allow_html=True)

    if not models_loaded:
        st.warning(f"Model tidak dapat dimuat: {load_err}. Pastikan file `.pt` dan `.h5` ada di folder `model/`. Halaman tetap dapat menampilkan contoh gambar.")

# -------------------- DETECTION PAGE --------------------
elif st.session_state.page == "detect":
    st.markdown("<div style='margin-bottom:8px;'><button class='back-btn' onClick='window.location.reload();'>⬅ Kembali ke Menu Utama</button></div>", unsafe_allow_html=True)
    # Note: we use st.button to change session state instead of JS reload for reliability
    if st.button("⬅ Kembali ke Menu Utama (klik)"):
        go_home()
        st.experimental_rerun()

    st.markdown("<h3 style='color:#6b4f3a;'>🎯 Deteksi Objek (Cocopham & Sprite)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Pilih gambar bawaan atau unggah gambar sendiri untuk dideteksi.</div>", unsafe_allow_html=True)
    st.write("")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Gambar Tersedia**", unsafe_allow_html=True)
        # build choices from images folder
        available = []
        for fname in ("cocopham.jpg", "sprite.jpg"):
            if os.path.exists(builtin_path(fname)):
                available.append(fname)
        options = ["-- pilih --"] + [os.path.splitext(os.path.basename(f))[0].capitalize() for f in available]
        # Make a stable mapping: displayed label -> filename
        display_map = {os.path.splitext(os.path.basename(f))[0].capitalize(): f for f in available}

        built_choice = st.radio("Pilih contoh:", options)
        st.markdown("<br>atau</br>", unsafe_allow_html=True)
        uploaded_det = st.file_uploader("📤 Upload gambar untuk deteksi", type=["jpg", "jpeg", "png"], key="upload_detect")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        selected_image = None
        # determine selected image
        if built_choice and built_choice != "-- pilih --" and built_choice in display_map:
            selected_image = load_builtin_image(display_map[built_choice])
        elif uploaded_det is not None:
            selected_image = Image.open(uploaded_det).convert("RGB")

        if selected_image is None:
            st.info("Pilih gambar contoh di kiri atau upload gambar untuk memulai deteksi.")
        else:
            st.image(selected_image, caption="Gambar untuk dideteksi", use_column_width=True)

            if not models_loaded:
                st.error("Model deteksi tidak tersedia. Pastikan file `.pt` ada di folder `model/`.")
            else:
                with st.spinner("🐱 Mendeteksi objek..."):
                    # Keep same YOLO logic as original
                    try:
                        results = yolo_model(selected_image, conf=0.7)
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
                            # annotated image (ultralytics returns numpy array)
                            try:
                                annotated = results[0].plot()
                                if isinstance(annotated, np.ndarray):
                                    annotated = Image.fromarray(annotated)
                                st.image(annotated, caption="🎉 Hasil Deteksi!", use_column_width=True)
                            except Exception:
                                st.image(selected_image, caption="Hasil Deteksi (fallback)", use_column_width=True)

                            st.success("✅ Deteksi berhasil!")
                            st.dataframe({
                                "Class": [f[0] for f in filtered],
                                "Confidence": [round(f[1], 2) for f in filtered],
                            })
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat deteksi: {e}")

# -------------------- CLASSIFICATION PAGE --------------------
elif st.session_state.page == "classify":
    if st.button("⬅ Kembali ke Menu Utama"):
        go_home()
        st.experimental_rerun()

    st.markdown("<h3 style='color:#6b4f3a;'>🧩 Klasifikasi Gambar (Parasitized & Uninfected)</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Pilih gambar bawaan atau unggah gambar sendiri untuk diklasifikasikan.</div>", unsafe_allow_html=True)
    st.write("")

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Gambar Tersedia**", unsafe_allow_html=True)
        available = []
        for fname in ("parasitized.jpg", "uninfected.jpg"):
            if os.path.exists(builtin_path(fname)):
                available.append(fname)
        options = ["-- pilih --"] + [os.path.splitext(os.path.basename(f))[0].capitalize() for f in available]
        display_map = {os.path.splitext(os.path.basename(f))[0].capitalize(): f for f in available}

        built_choice = st.radio("Pilih contoh:", options, key="clf_choice")
        st.markdown("<br>atau</br>", unsafe_allow_html=True)
        uploaded_clf = st.file_uploader("📤 Upload gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="upload_clf")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        selected_image = None
        if built_choice and built_choice != "-- pilih --" and built_choice in display_map:
            selected_image = load_builtin_image(display_map[built_choice])
        elif uploaded_clf is not None:
            selected_image = Image.open(uploaded_clf).convert("RGB")

        if selected_image is None:
            st.info("Pilih gambar contoh di kiri atau upload gambar untuk memulai klasifikasi.")
        else:
            st.image(selected_image, caption="Gambar untuk diklasifikasikan", use_column_width=True)

            if not models_loaded:
                st.error("Model klasifikasi tidak tersedia. Pastikan file `.h5` ada di folder `model/`.")
            else:
                with st.spinner("🧠 Menganalisis gambar..."):
                    try:
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
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat klasifikasi: {e}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — CuteVision Pastel Edition (tema coklat pastel)")
