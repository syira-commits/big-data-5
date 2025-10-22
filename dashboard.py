        # ==========================
        # YOLO DETECTION (PERBAIKAN)
        # ==========================
        if menu == "🎯 Deteksi Objek (YOLO)":
            st.subheader("⚙️ Pengaturan Deteksi")
            detect_option = st.checkbox("Aktifkan deteksi objek YOLO", value=True)

            if detect_option:
                with st.spinner("🔍 Sedang mendeteksi objek..."):
                    results = yolo_model(img, conf=0.25, iou=0.3)
                    boxes = results[0].boxes

                    # Jika tidak ada box sama sekali
                    if boxes is None or boxes.data is None or len(boxes.data) == 0:
                        st.markdown("""
                        <div class='result-card theme-peach' style='text-align:center;'>
                            <b>⚠️ Tidak ada objek yang terdeteksi.</b><br>
                            Gambar ini tampaknya tidak mengandung objek yang dikenali oleh model.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Ambil data box dan filter confidence manual (>0.25)
                        data = boxes.data.cpu().numpy()
                        filtered_data = data[data[:, 4] > 0.25]

                        if len(filtered_data) == 0:
                            # Tidak ada box yang lolos threshold
                            st.markdown("""
                            <div class='result-card theme-peach' style='text-align:center;'>
                                <b>⚠️ Tidak ada objek yang terdeteksi.</b><br>
                                Semua prediksi memiliki tingkat kepercayaan rendah.
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Tampilkan hasil deteksi jika ada box valid
                            result_img = results[0].plot(line_width=2, font_size=12)
                            st.markdown("<div class='result-card theme-mint'>", unsafe_allow_html=True)
                            st.image(result_img, caption="🎉 Hasil Deteksi", use_container_width=True)

                            df = pd.DataFrame({
                                "Class": [results[0].names[int(cls)] for cls in filtered_data[:, 5]],
                                "Confidence": [round(conf, 2) for conf in filtered_data[:, 4]],
                                "X_min": filtered_data[:, 0],
                                "Y_min": filtered_data[:, 1],
                                "X_max": filtered_data[:, 2],
                                "Y_max": filtered_data[:, 3]
                            })
                            st.dataframe(df)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Tombol unduh hasil
                            buf = io.BytesIO()
                            Image.fromarray(result_img).save(buf, format="PNG")
                            st.download_button(
                                label="📥 Unduh Hasil Deteksi",
                                data=buf.getvalue(),
                                file_name="hasil_deteksi.png",
                                mime="image/png"
                            )
