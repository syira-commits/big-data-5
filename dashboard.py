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

# -------------------- STYLE (COKLAT PASTEL BERWARNA) --------------------
st.markdown(
    """
    <style>
    :root{
        --bg-start: #f4e1c1;   /* cream */
        --bg-mid:   #f0d8bb;
        --bg-end:   #c49a6c;   /* kopi muda */
        --card: #fff6f0;
        --accent: #7b4b2a;
        --muted: #6b4f3a;
        --soft-border: rgba(123,75,42,0.12);
    }
    html, body, [class*="css"]  {
        background: linear-gradient(135deg, var(--bg-start) 0%, var(--bg-mid) 50%, var(--bg-end) 100%) !important;
    }
    .main-title { text-align:center; color:var(--muted); font-size:36px; font-weight:800; margin-bottom:4px; }
    .sub-text { text-align:center; color:var(--accent); font-size:15px; margin-top:0px; margin-bottom:18px; }
    .card {
        border-radius:14px;
        padding:16px;
        background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(255,250,245,0.9));
        box-shadow: 0 8px 28px rgba(107,79,58,0.08);
        border: 1px solid var(--soft-border);
    }
    .muted { color:#7b5e48; font-size:14px; }
    .hint { color:#7b5e48; font-size:13px; margin-top:8px; }
    .upload-box{ border:2px dashed rgba(123,75,42,0.16); padding:12px; border-radius:12px; background:rgba(255,255,255,0.6); }
    .btn { background-color:#c49a6c; color:white; padding:8px 14px; border-radius:10px; border: none; font-weight:700; }
    .small { font-size:13px; color:#7b5e48; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Deteksi Objek dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Pilih mode lalu unggah gambar (atau baca keterangan contoh gambar yang sesuai)</div>', unsafe_allow_html=True)

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

# -------------------- UTILS --------------------
def reset_uploads():
    # clear uploaded files/selections
    for k in ["uploaded_detect","uploaded_clf","built_choice_detect","built_choice_clf"]:
        if k in st.session_state:
            st.session_state.pop(k)

# -------------------- HOME PAGE --------------------
if st.session_state.page == "home":
    # Centered card with choices
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("## Pilih Mode", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.image("", width=280)  # visual spacing (no admin images)
            st.markdown("### 🎯 Deteksi Objek", unsafe_allow_html=True)
            st.markdown("<div class='muted'>Gunakan model YOLO untuk mendeteksi objek: <strong>Cocopham</strong> atau <strong>Sprite</strong>.</div>", unsafe_allow_html=True)
            st.write("")
            st.button("Buka Deteksi →", key="open_detect", on_click=go_detect)
        with c2:
            st.image("", width=280)
            st.markdown("### 🧩 Klasifikasi Gambar", unsafe_allow_html=True)
            st.markdown("<div class='muted'>Gunakan model klasifikasi untuk gambar sel: <strong>Parasitized</strong> atau <strong>Uninfected</strong>.</div>", unsafe_allow_html=True)
            st.write("")
            st.button("Buka Klasifikasi →", key="open_classify", on_click=go_classify)
        st.markdown("</div>", unsafe_allow_html=True)

    # show model loading warning if any
    if not models_loaded:
        st.warning(f"Model tidak dimuat: {load_err}. Pastikan folder `model/` berisi file `.pt` & `.h5`. UI tetap dapat digunakan tetapi proses tidak akan berjalan tanpa model.")

# -------------------- DETECTION PAGE --------------------
elif st.session_state.page == "detect":
    top_cols = st.columns([1,6,1])
    with top_cols[1]:
        if st.button("⬅ Kembali ke Menu Utama"):
            go_home()
            reset_uploads()
            st.experimental_rerun()
        st.markdown("<h3 style='color:#6b4f3a;'>🎯 Deteksi Objek (Cocopham & Sprite)</h3>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Pilih gambar dari keterangan contoh di bawah atau unggah gambar sendiri untuk dideteksi.</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Keterangan Gambar (contoh)**", unsafe_allow_html=True)
        st.markdown("<div class='small'>• Cocopham — botol minuman contoh (gunakan gambar botol/kemasan)</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>• Sprite — botol minuman contoh (gunakan gambar botol/kemasan)</div>", unsafe_allow_html=True)
        st.markdown("<div class='hint'>Catatan: pilihan di sini hanya keterangan. Untuk menjalankan deteksi, silakan unggah gambar yang ingin diuji.</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        # uploader and reset
        uploaded_det = st.file_uploader("📤 Unggah gambar untuk deteksi", type=["jpg","jpeg","png"], key="uploaded_detect")
        if st.button("🔁 Reset Gambar (Deteksi)"):
            if "uploaded_detect" in st.session_state:
                st.session_state.pop("uploaded_detect")
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        if "uploaded_detect" in st.session_state and st.session_state.uploaded_detect is not None:
            img = Image.open(st.session_state.uploaded_detect).convert("RGB")
            st.image(img, caption="Gambar yang akan dideteksi", use_column_width=True)
            if not models_loaded:
                st.error("Model deteksi tidak tersedia. Masukkan file model di folder `model/`.")
            else:
                with st.spinner("☕ Sedang mendeteksi..."):
                    try:
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
                            try:
                                annotated = results[0].plot()
                                if isinstance(annotated, np.ndarray):
                                    annotated = Image.fromarray(annotated)
                                st.image(annotated, caption="🎉 Hasil Deteksi!", use_column_width=True)
                            except Exception:
                                st.image(img, caption="Hasil Deteksi (fallback)", use_column_width=True)

                            st.success("✅ Deteksi berhasil!")
                            st.dataframe({
                                "Class": [f[0] for f in filtered],
                                "Confidence": [round(f[1], 2) for f in filtered],
                            })
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat deteksi: {e}")
        else:
            st.info("Belum ada gambar untuk dideteksi. Unggah file di panel kiri (format: jpg/png).")

# -------------------- CLASSIFICATION PAGE --------------------
elif st.session_state.page == "classify":
    top_cols = st.columns([1,6,1])
    with top_cols[1]:
        if st.button("⬅ Kembali ke Menu Utama"):
            go_home()
            reset_uploads()
            st.experimental_rerun()
        st.markdown("<h3 style='color:#6b4f3a;'>🧩 Klasifikasi Gambar (Parasitized & Uninfected)</h3>", unsafe_allow_html=True)
        st.markdown("<div class='muted'>Pilih gambar dari keterangan contoh di bawah atau unggah gambar sendiri untuk diklasifikasikan.</div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Keterangan Gambar (contoh)**", unsafe_allow_html=True)
        st.markdown("<div class='small'>• Parasitized — gambar sel yang terinfeksi (gunakan gambar sel mikroskop)</div>", unsafe_allow_html=True)
        st.markdown("<div class='small'>• Uninfected — gambar sel normal (gunakan gambar sel mikroskop)</div>", unsafe_allow_html=True)
        st.markdown("<div class='hint'>Catatan: pilihan di sini hanya keterangan. Untuk menjalankan klasifikasi, silakan unggah gambar sel yang ingin diuji.</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        uploaded_clf = st.file_uploader("📤 Unggah gambar untuk klasifikasi", type=["jpg","jpeg","png"], key="uploaded_clf")
        if st.button("🔁 Reset Gambar (Klasifikasi)"):
            if "uploaded_clf" in st.session_state:
                st.session_state.pop("uploaded_clf")
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        if "uploaded_clf" in st.session_state and st.session_state.uploaded_clf is not None:
            img = Image.open(st.session_state.uploaded_clf).convert("RGB")
            st.image(img, caption="Gambar yang akan diklasifikasikan", use_column_width=True)
            if not models_loaded:
                st.error("Model klasifikasi tidak tersedia. Masukkan file model di folder `model/`.")
            else:
                with st.spinner("☕ Sedang menganalisis..."):
                    try:
                        img_resized = img.resize((128, 128))
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
                        img_cv = np.array(img)
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
        else:
            st.info("Belum ada gambar untuk diklasifikasikan. Unggah file di panel kiri (format: jpg/png).")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("🐾 Dibuat oleh Mulya Syira — CuteVision Pastel Edition (tema coklat pastel hangat)")

