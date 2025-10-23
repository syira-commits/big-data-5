import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np

# ========================== CONFIG ==========================
st.set_page_config(
    page_title="Deteksi Objek dan Klasifikasi Gambar",
    page_icon="🩺",
    layout="wide"
)

# ========================== CUSTOM STYLE ==========================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Quicksand', sans-serif;
            background: linear-gradient(135deg, #f9ede3 0%, #f6e9ff 50%, #e6f7f5 100%);
            color: #2e2a26;
        }

        section[data-testid="stSidebar"] {
            background-color: #fdf8f4;
            border-right: 2px solid #f0e1d6;
        }

        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #4a3426;
            margin-bottom: 15px;
        }

        .sidebar-item {
            font-size: 16px;
            padding: 8px 0;
            color: #4a3426;
        }

        .sidebar-item:hover {
            color: #8b6333;
        }

        .main-header {
            font-size: 36px;
            font-weight: bold;
            color: #2f2a27;
            margin-bottom: -5px;
        }

        .sub-header {
            color: #6d635a;
            font-size: 16px;
            margin-bottom: 40px;
        }

        .card {
            background-color: #fff8f3;
            border-radius: 14px;
            padding: 25px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            margin-bottom: 25px;
            text-align: center;
        }

        .card h3 {
            color: #4b3f3a;
            margin-bottom: 10px;
        }

        .stButton>button {
            background-color: #d3b4f0;
            color: #2f2a27;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            font-weight: bold;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #c59ef0;
            transform: scale(1.05);
        }

        .info-card {
            background-color: #fff7f0;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }

        hr {
            border: none;
            height: 2px;
            background: #f0e1d6;
            margin: 40px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== SIDEBAR ==========================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🩺 Dashboard</div>', unsafe_allow_html=True)
    menu = st.radio("Navigasi:", ["🏠 Dashboard", "🔍 Deteksi Objek", "🧠 Klasifikasi Gambar", "ℹ️ Tentang"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("AI Vision System © 2025")

# ========================== DASHBOARD ==========================
if menu == "🏠 Dashboard":
    st.markdown('<div class="main-header">Deteksi Objek dan Klasifikasi Gambar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pilih fitur yang ingin digunakan dan jelajahi kemampuan model AI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🔍 Deteksi Objek</h3>", unsafe_allow_html=True)
        st.write("Gunakan model YOLOv8 untuk mendeteksi berbagai objek dari gambar.")
        if st.button("Aktifkan Deteksi Objek"):
            st.session_state["menu"] = "🔍 Deteksi Objek"
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h3>🧠 Klasifikasi Gambar</h3>", unsafe_allow_html=True)
        st.write("Gunakan model Keras untuk mengklasifikasikan gambar sel sebagai Parasitized atau Uninfected.")
        if st.button("Aktifkan Klasifikasi Gambar"):
            st.session_state["menu"] = "🧠 Klasifikasi Gambar"
            st.experimental_rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="info-card"><b>📅 Tanggal:</b> 23 Oktober 2025<br><b>Status:</b> Siap digunakan</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="info-card"><b>🧾 Catatan:</b> Pilih mode untuk memulai deteksi atau klasifikasi gambar.</div>', unsafe_allow_html=True)

# ========================== MODE DETEKSI ==========================
elif menu == "🔍 Deteksi Objek":
    st.title("🔍 Mode Deteksi Objek")
    st.info("Fitur deteksi objek belum diaktifkan. Aktifkan model untuk melanjutkan pengujian.")
    st.button("Kembali ke Dashboard", on_click=lambda: st.experimental_set_query_params(menu="🏠 Dashboard"))

# ========================== MODE KLASIFIKASI ==========================
elif menu == "🧠 Klasifikasi Gambar":
    st.title("🧠 Mode Klasifikasi Gambar")
    st.info("Fitur klasifikasi belum diaktifkan. Aktifkan model untuk memulai klasifikasi.")
    st.button("Kembali ke Dashboard", on_click=lambda: st.experimental_set_query_params(menu="🏠 Dashboard"))

# ========================== TENTANG ==========================
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini dirancang untuk mendemonstrasikan dua fungsi utama:
    - **Deteksi Objek:** Menggunakan model YOLOv8 untuk mengenali objek.
    - **Klasifikasi Gambar:** Menggunakan model Keras untuk membedakan sel Parasitized dan Uninfected.

    Dibangun dengan:
    - Streamlit  
    - TensorFlow / Keras  
    - Ultralytics YOLO  
    """)
    st.markdown('<div class="info-card">Versi Aplikasi: v1.0 — Dirancang untuk eksplorasi dan pembelajaran AI</div>', unsafe_allow_html=True)
