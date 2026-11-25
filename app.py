
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.express as px

from pathlib import Path

# ==========================================
# KONFIGURASI WEBSITE
# ==========================================

st.set_page_config(
    page_title="Karhutla AI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

# ==========================================
# DESAIN WEBSITE
# ==========================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(
        120deg, #052e16, #166534, #15803d
    );
    padding: 40px;
    border-radius: 20px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    color: white;
    margin-bottom: 10px;
}

.hero p {
    color: #dcfce7;
    font-size: 16px;
}

[data-testid="stMetric"] {
    border: 1px solid #64748b55;
    padding: 20px;
    border-radius: 15px;
}

.stButton > button {
    background-color: #166534;
    color: white;
    border-radius: 10px;
    border: none;
    min-height: 45px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #15803d;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# MEMUAT MODEL DAN PREPROCESSING
# ==========================================

@st.cache_resource
def load_artifacts():

    with open(ARTIFACTS / "config.json") as f:
        config = json.load(f)

    model = tf.keras.models.load_model(
        ARTIFACTS / "model_lstm_best.keras",
        compile=False
    )

    imputer = joblib.load(
        ARTIFACTS / "imputer.pkl"
    )

    scaler = joblib.load(
        ARTIFACTS / "scaler.pkl"
    )

    return model, imputer, scaler, config


model, imputer, scaler, config = load_artifacts()

FEATURES = config["features"]
LOOKBACK = config["lookback"]


@st.cache_data
def load_csv(filename):
    return pd.read_csv(ARTIFACTS / filename)


results = load_csv("model_comparison.csv")
validation = load_csv("validation_results.csv")
importance = load_csv("feature_importance.csv")
ablation = load_csv("ablation_results.csv")
top5 = load_csv("top5_errors.csv")


# ==========================================
# FUNGSI PREDIKSI
# ==========================================

def predict_hotspot(data):

    if len(data) != LOOKBACK:
        raise ValueError(
            f"Data harus berisi tepat {LOOKBACK} bulan."
        )

    missing = [
        f for f in FEATURES
        if f not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom belum lengkap: {missing}"
        )

    data = data.copy()

    dates = pd.to_datetime(
        data["Bulan"],
        format="%Y-%m",
        errors="raise"
    )

    data["Bulan"] = dates
    data = data.sort_values("Bulan")

    month_ids = (
        data["Bulan"].dt.year * 12
        + data["Bulan"].dt.month
    ).to_numpy()

    if not np.all(np.diff(month_ids) == 1):
        raise ValueError(
            "Bulan harus berurutan tanpa duplikasi."
        )

    values = data[FEATURES].apply(
        pd.to_numeric,
        errors="raise"
    ).to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            "Pastikan seluruh nilai fitur terisi dan valid."
        )

    for feature in ["fire_count_lnh", "tp", "wind_speed"]:
        if feature in FEATURES:
            if (data[feature].astype(float) < 0).any():
                raise ValueError(
                    f"Nilai {feature} tidak boleh negatif."
                )

    values = imputer.transform(values)
    values = scaler.transform(values)

    values = values.reshape(
        1, LOOKBACK, len(FEATURES)
    ).astype(np.float32)

    prediction_log = model.predict(
        values,
        verbose=0
    )[0][0]

    prediction = max(
        float(np.expm1(prediction_log)), 0
    )

    if not np.isfinite(prediction):
        raise ValueError("Prediksi tidak valid.")

    next_month = (
        data["Bulan"].iloc[-1]
        + pd.offsets.MonthBegin(1)
    )

    return prediction, next_month


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🔥 KARHUTLA AI")

    st.caption(
        "Wildfire Prediction System"
    )

    st.divider()

    menu = st.radio(
        "Navigasi",
        [
            "Dashboard",
            "Prediksi Hotspot",
            "Evaluasi Model",
            "Analisis Model",
            "Tentang Sistem"
        ]
    )

    st.divider()

    st.caption("Deep Learning — UAS Jalur B")
    st.caption("Kelompok 4")
    st.caption("Universitas Palangka Raya")
    st.caption("2026")


# ==========================================
# HALAMAN DASHBOARD
# ==========================================

if menu == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>Prediksi Karhutla</h1>
        <p>
        Sistem prediksi jumlah hotspot Kalimantan Tengah
        menggunakan Long Short-Term Memory dan Optuna.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Model", "LSTM")
    c2.metric("Time Steps", f"{LOOKBACK} Bulan")
    c3.metric("Jumlah Fitur", len(FEATURES))
    c4.metric("Horizon", "1 Bulan")

    st.subheader("Tren Historis Hotspot")

    st.image(
        str(
            ARTIFACTS /
            "tren_hotspot_kalimantan_tengah.png"
        ),
        use_container_width=True
    )

    st.subheader("Performa LSTM + Optuna")

    row = results[
        results["Model"] == "LSTM + Optuna"
    ].iloc[0]

    a, b, c, d = st.columns(4)

    a.metric("MAE", f"{row['MAE']:.4f}")
    b.metric("MSE", f"{row['MSE']:.4f}")
    c.metric("RMSE", f"{row['RMSE']:.4f}")
    d.metric("R²", f"{row['R2']:.4f}")

    st.info(
        "Model memprediksi jumlah hotspot bulan berikutnya "
        "untuk satu sel grid geografis."
    )


# ==========================================
# HALAMAN PREDIKSI
# ==========================================

elif menu == "Prediksi Hotspot":

    st.title("Prediksi Jumlah Hotspot")

    st.write(
        f"Masukkan riwayat {LOOKBACK} bulan "
        "dari satu sel grid geografis."
    )

    st.caption(
        "Gunakan satuan dan definisi fitur yang sama "
        "dengan dataset penelitian."
    )

    method = st.radio(
        "Metode Input",
        ["Input Manual", "Upload CSV"],
        horizontal=True
    )

    sample = load_csv("sample_input.csv")

    st.download_button(
        "Download Contoh Format CSV",
        sample.to_csv(index=False),
        "template_karhutla.csv",
        "text/csv"
    )

    input_data = None

    if method == "Input Manual":

        st.info(
            "Tabel diisi dengan contoh data historis asli. "
            "Ubah nilainya sesuai data yang ingin diprediksi."
        )

        input_data = st.data_editor(
            sample,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key="manual_input"
        )

    else:

        uploaded_file = st.file_uploader(
            "Unggah CSV riwayat 12 bulan",
            type=["csv"]
        )

        if uploaded_file is not None:

            input_data = pd.read_csv(uploaded_file)

            st.dataframe(
                input_data,
                use_container_width=True
            )

    if st.button(
        "Jalankan Prediksi",
        use_container_width=True
    ):

        if input_data is None:

            st.warning("Masukkan data terlebih dahulu.")

        else:

            try:

                with st.spinner("Memproses prediksi..."):

                    prediction, next_month = (
                        predict_hotspot(input_data)
                    )

                st.success("Prediksi berhasil!")

                st.metric(
                    "Estimasi Jumlah Hotspot",
                    f"{prediction:,.2f}"
                )

                st.write(
                    "Periode prediksi:",
                    next_month.strftime("%B %Y")
                )

                if "fire_count_lnh" in input_data.columns:

                    chart_data = input_data[
                        ["Bulan", "fire_count_lnh"]
                    ].copy()

                    chart_data = chart_data.sort_values("Bulan")

                    chart_data = chart_data.rename(
                        columns={
                            "fire_count_lnh": "Hotspot"
                        }
                    )

                    chart_data["Jenis"] = "Historis"

                    future = pd.DataFrame({
                        "Bulan": [
                            next_month.strftime("%Y-%m")
                        ],
                        "Hotspot": [prediction],
                        "Jenis": ["Prediksi"]
                    })

                    chart_data = pd.concat(
                        [chart_data, future],
                        ignore_index=True
                    )

                    fig = px.bar(
                        chart_data,
                        x="Bulan",
                        y="Hotspot",
                        color="Jenis",
                        title="Riwayat dan Prediksi Hotspot"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            except Exception as e:

                st.error(f"Prediksi gagal: {e}")

    st.warning(
        "Hasil merupakan estimasi penelitian, "
        "bukan peringatan resmi kebakaran. "
        "Model dapat meremehkan lonjakan hotspot ekstrem."
    )


# ==========================================
# HALAMAN EVALUASI MODEL
# ==========================================

elif menu == "Evaluasi Model":

    st.title("Evaluasi Performa Model")

    tab1, tab2, tab3 = st.tabs([
        "Testing",
        "Validation",
        "Learning Curve"
    ])

    with tab1:

        st.dataframe(
            results.round(4),
            use_container_width=True,
            hide_index=True
        )

        metric = st.selectbox(
            "Pilih Metrik",
            ["MAE", "MSE", "RMSE", "R2"]
        )

        fig = px.bar(
            results,
            x="Model",
            y=metric,
            color="Model",
            text_auto=".3f",
            title=f"Perbandingan {metric}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab2:

        st.dataframe(
            validation.round(4),
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        st.image(
            str(
                ARTIFACTS /
                "learning_curve_mlp_lstm.png"
            ),
            use_container_width=True
        )

        st.caption(
            "Loss dan MAE pelatihan dihitung "
            "pada target hasil transformasi log1p."
        )


# ==========================================
# HALAMAN ANALISIS MODEL
# ==========================================

elif menu == "Analisis Model":

    st.title("Interpretabilitas dan Analisis Model")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Importance",
        "Ablation Study",
        "Optuna",
        "Error Analysis"
    ])

    with tab1:

        st.dataframe(
            importance.round(4),
            use_container_width=True
        )

        fig = px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            error_x="Std",
            orientation="h",
            title="Permutation Feature Importance"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Feature importance tidak membuktikan "
            "hubungan sebab-akibat."
        )

    with tab2:

        st.dataframe(
            ablation.round(4),
            use_container_width=True
        )

        st.image(
            str(ARTIFACTS / "ablation_study.png"),
            use_container_width=True
        )

    with tab3:

        with open(
            ARTIFACTS / "best_params.json"
        ) as f:

            best_params = json.load(f)

        st.subheader("Best Hyperparameters")

        st.json(best_params)

        st.image(
            str(ARTIFACTS / "optuna_history.png"),
            use_container_width=True
        )

        st.image(
            str(ARTIFACTS / "optuna_importance.png"),
            use_container_width=True
        )

    with tab4:

        st.subheader("Lima Kesalahan Terbesar")

        st.dataframe(
            top5.round(4),
            use_container_width=True
        )

        st.image(
            str(ARTIFACTS / "top5_errors.png"),
            use_container_width=True
        )

        st.subheader("Analisis Kondisi Iklim")

        st.dataframe(
            load_csv(
                "top5_climate_analysis.csv"
            ).round(4),
            use_container_width=True
        )


# ==========================================
# HALAMAN TENTANG SISTEM
# ==========================================

elif menu == "Tentang Sistem":

    st.title("Tentang Karhutla AI")

    st.markdown("""
    ## Latar Belakang

    Karhutla AI merupakan proyek UAS mata kuliah
    Deep Learning untuk memprediksi jumlah hotspot
    di Kalimantan Tengah.

    ## Metodologi

    - Dataset Indonesian Wildfire Panel 2012–2025.
    - Preprocessing menggunakan median imputation
      dan StandardScaler.
    - Sliding window dengan 12 time steps.
    - Arsitektur Long Short-Term Memory (LSTM).
    - Hyperparameter optimization menggunakan Optuna.
    - Studi ablasi dan permutation feature importance.

    ## Kajian Etika

    **Bias sensor satelit:** Deteksi hotspot dapat
    dipengaruhi tutupan awan, resolusi sensor, dan
    waktu pengamatan satelit.

    **Privasi:** Sistem menggunakan data iklim
    dan grid geografis. Pengguna tidak perlu
    memasukkan identitas pribadi.

    **Risiko penyalahgunaan:** Prediksi tidak boleh
    digunakan sebagai satu-satunya dasar penetapan
    lokasi kebakaran maupun tuduhan terhadap individu.

    **Keterbatasan:** Model dapat menghasilkan
    underprediction pada lonjakan hotspot ekstrem.
    Hotspot juga tidak selalu menunjukkan kejadian
    kebakaran yang telah terkonfirmasi.

    ## Informasi Pengembang

    **Kelompok 4 — Deep Learning**

    Jurusan Teknik Informatika

    Fakultas Teknik

    Universitas Palangka Raya

    2026
    """)

st.divider()

st.caption(
    "KARHUTLA AI | Deep Learning UAS Jalur B | "
    "Universitas Palangka Raya 2026"
)
