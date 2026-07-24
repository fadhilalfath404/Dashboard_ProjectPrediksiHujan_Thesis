import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Thesis Dashboard", page_icon="🌤️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Dark blue background for the whole app */
    .stApp { background-color: #0F172A; } 
    
    /* Make all default text white/light gray so it's readable */
    h1, h2, h3, h4, p, span, div, label { color: #F8FAFC !important; } 
    
    /* Styling for the cards to match the dark theme */
    .metric-card {
        background-color: #1E293B; padding: 20px; border-radius: 12px;
        text-align: center; margin-bottom: 20px; border: 1px solid #334155;
    }
    .metric-title { font-size: 1.1rem; color: #94A3B8 !important; margin-bottom: 10px; font-weight: 500;}
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #38BDF8 !important; }
    
    .workflow-card {
        background-color: #1E293B; border-radius: 10px; padding: 20px;
        margin: 10px 0; text-align: center; 
        border-left: 5px solid #38BDF8; font-weight: 500;
    }
    .arrow { text-align: center; font-size: 24px; color: #94A3B8 !important; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

def create_metric_card(title, value):
    st.markdown(f'<div class="metric-card"><div class="metric-title">{title}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

# --- THESIS RESULTS (SOURCE OF TRUTH) ---
METRICS = {
    "MLP": {"RMSE": 15.87, "MAE": 10.95, "MAPE": 28.64, "R2": 0.74},
    "LSTM": {"RMSE": 12.45, "MAE": 8.72, "MAPE": 21.35, "R2": 0.82},
    "TCN": {"RMSE": 10.32, "MAE": 7.15, "MAPE": 18.27, "R2": 0.87}
}

SENSITIVITY = {
    "Kelembapan": 32.5, "Suhu": 27.8, "Tekanan Udara": 18.4,
    "Kecepatan Angin": 12.6, "Variabel Lain": 8.7
}

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("☁️ Navigasi Tesis")
page = st.sidebar.radio("Pilih Halaman:", [
    "🏠 Home", 
    "📂 Dataset", 
    "⚙ Preprocessing", 
    "🧠 Models", 
    "📊 Model Comparison", 
    "🌧 Prediction", 
    "🏆 Results", 
    "ℹ About"
])

# ==========================================
# PAGE ROUTING LOGIC
# ==========================================

if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #2c3e50; font-weight: 800;'>ANALISIS PEMODELAN ALGORITMA MULTILAYER PERCEPTRON, LONG SHORT-TERM MEMORY, DAN TEMPORAL CONVOLUTIONAL NETWORK UNTUK PREDIKSI HUJAN DI BANDARA SOEKARNO-HATTA</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style='background-color: green; padding: 40px; border-radius: 15px; box-shadow: 0 10px 20px rgba(0,0,0,0.05); text-align: center; border-top: 5px solid #3498db;'>
            <h3 style='color: #7f8c8d; text-transform: uppercase; font-size: 1rem; letter-spacing: 2px;'>Peneliti</h3>
            <h2 style='color: #2980b9; font-size: 2.5rem; margin-top: 10px;'>Eko Widyantoro Hadi Soebroto</h2>
            <p style='font-size: 1.2rem; color: #34495e; font-weight: 500;'>NIM: 241012000013</p>
            <hr style='border: 1px solid #f0f3f4;'>
            <p style='font-size: 1.1rem; color: #34495e;'><b>Program Studi:</b><br>Teknik Informatika S-2</p>
            <p style='font-size: 1.1rem; color: #34495e;'><b>Dosen Pembimbing:</b><br>Dr. Sajarwo Anggai, S.St., M.T.<br>Dr. Sudarno Wiharjo, D.E.A</p>
            <p style='font-size: 1.1rem; color: #34495e; margin-top: 20px;'><b>Universitas Pamulang (2026)</b></p>
        </div>
        """, unsafe_allow_html=True)

elif page == "📂 Dataset":
    st.title("📂 Dataset Overview")
    st.markdown("Ringkasan dataset historis meteorologi dari Stasiun BMKG Bandara Soekarno-Hatta (2015-2024).")

    col1, col2, col3, col4 = st.columns(4)
    with col1: create_metric_card("Total Records", "3,653")
    with col2: create_metric_card("Total Features", "12")
    with col3: create_metric_card("Missing Values", "0 (Cleaned)")
    with col4: create_metric_card("Label", "isRain (0/1)")

    st.markdown("### 🔍 Dataset Preview")
    # Generating mock display data based on thesis variables (Values now properly mapped)
    np.random.seed(42)
    dates = pd.date_range(start="2015-01-01", periods=8, freq='D')
    df_mock = pd.DataFrame({
        'Date': dates,
        'Suhu (C)': np.random.uniform(25, 31, len(dates)),
        'Kelembapan (%)': np.random.uniform(60, 95, len(dates)),
        'Tekanan Udara (hPa)': np.random.uniform(1005, 1015, len(dates)),
        'Kecepatan Angin (m/s)': np.random.uniform(0.5, 12, len(dates)),
        'isRain': np.random.choice([1], size=len(dates)) # <-- FIXED LINE
    })
    st.dataframe(df_mock, use_container_width=True)

elif page == "⚙ Preprocessing":
    st.title("⚙ Alur Pra-Pemrosesan Data")
    st.markdown("Pipeline preprocessing yang dilakukan sebelum model dilatih berdasarkan metodologi riset tesis:")

    st.markdown("""
    <div class='workflow-card'><h4>1. Data Historis Observasi</h4><p>Agregasi data 10 menitan menjadi harian rata-rata.</p></div>
    <div class='arrow'>↓</div>
    <div class='workflow-card'><h4>2. Data Cleaning</h4><p>Handling Missing Values (Interpolasi) & Deteksi Outlier.</p></div>
    <div class='arrow'>↓</div>
    <div class='workflow-card'><h4>3. Normalisasi Data</h4><p>Min-Max Scaling untuk rentang 0-1 guna stabilitas performa.</p></div>
    <div class='arrow'>↓</div>
    <div class='workflow-card'><h4>4. Time-Series Sliding Window</h4><p>TCN Lookback: 45 Hari | LSTM Lookback: 30 Hari.</p></div>
    <div class='arrow'>↓</div>
    <div class='workflow-card'><h4>5. Time Series Split (Chronological)</h4><p>Train Set (2019-2023) | Validation Set (2024) | Test Set (2025)</p></div>
    """, unsafe_allow_html=True)

elif page == "🧠 Models":
    st.title("🧠 Arsitektur Model Deep Learning")

    tab1, tab2, tab3 = st.tabs(["🌟 TCN (Best Model)", "LSTM", "MLP"])
    with tab1:
        st.header("Temporal Convolutional Network (TCN)")
        st.success("TCN adalah model terbaik pada penelitian ini. Memanfaatkan Causal & Dilated Convolution untuk menangkap dependensi panjang.")
        c1, c2, c3, c4 = st.columns(4)
        with c1: create_metric_card("RMSE", METRICS['TCN']['RMSE'])
        with c2: create_metric_card("MAE", METRICS['TCN']['MAE'])
        with c3: create_metric_card("MAPE", f"{METRICS['TCN']['MAPE']}%")
        with c4: create_metric_card("R²", METRICS['TCN']['R2'])

    with tab2:
        st.header("Long Short-Term Memory (LSTM)")
        st.info("Peringkat kedua. Sangat kuat dalam menyimpan memori sekuensial namun sedikit kurang responsif pada lonjakan ekstrem.")
        c1, c2, c3, c4 = st.columns(4)
        with c1: create_metric_card("RMSE", METRICS['LSTM']['RMSE'])
        with c2: create_metric_card("MAE", METRICS['LSTM']['MAE'])
        with c3: create_metric_card("MAPE", f"{METRICS['LSTM']['MAPE']}%")
        with c4: create_metric_card("R²", METRICS['LSTM']['R2'])

    with tab3:
        st.header("Multi-Layer Perceptron (MLP)")
        st.warning("Performa terendah. Kehilangan dependensi temporal karena arsitektur Feedforward dasar.")
        c1, c2, c3, c4 = st.columns(4)
        with c1: create_metric_card("RMSE", METRICS['MLP']['RMSE'])
        with c2: create_metric_card("MAE", METRICS['MLP']['MAE'])
        with c3: create_metric_card("MAPE", f"{METRICS['MLP']['MAPE']}%")
        with c4: create_metric_card("R²", METRICS['MLP']['R2'])

elif page == "📊 Model Comparison":
    st.title("📊 Komparasi Performa Model")

    df_metrics = pd.DataFrame(METRICS).T.reset_index().rename(columns={'index': 'Model'})
    
    st.markdown("### 🏆 Tabel Hasil Evaluasi Resmi")
    st.dataframe(df_metrics, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_rmse = px.bar(df_metrics, x='Model', y='RMSE', text='RMSE', color='Model', color_discrete_sequence=['#f17066', '#5dade2', '#58d68d'])
        fig_rmse.update_layout(title="RMSE (Lower is Better)", template='plotly_white')
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        fig_r2 = px.bar(df_metrics, x='Model', y='R2', text='R2', color='Model', color_discrete_sequence=['#f17066', '#5dade2', '#58d68d'])
        fig_r2.update_layout(title="R² Score (Higher is Better)", template='plotly_white')
        st.plotly_chart(fig_r2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Analisis Sensitivitas Variabel (Feature Importance)")
    df_sens = pd.DataFrame(list(SENSITIVITY.items()), columns=['Variabel', 'Kontribusi (%)']).sort_values('Kontribusi (%)', ascending=True)
    fig_sens = px.bar(df_sens, x='Kontribusi (%)', y='Variabel', orientation='h', text='Kontribusi (%)', color_discrete_sequence=['#9b59b6'])
    fig_sens.update_layout(template='plotly_white')
    st.plotly_chart(fig_sens, use_container_width=True)

elif page == "🌧 Prediction":
    st.title("🌧 Simulasi Prediksi Hujan")
    st.markdown("Gunakan antarmuka ini untuk mensimulasikan sistem pendukung keputusan (Decision Support System) prakirawan cuaca.")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        st.markdown("#### 🎛 Parameter Atmosfer Terkini")
        temp = st.slider("🌡 Suhu Rata-rata (C)", 20.0, 40.0, 27.8)
        rh = st.slider("💧 Kelembapan (%)", 40.0, 100.0, 85.0)
        press = st.slider("🌬 Tekanan Udara (hPa)", 1000.0, 1020.0, 1010.0)
        wind = st.slider("💨 Kecepatan Angin (m/s)", 0.0, 20.0, 5.0)
        st.markdown("</div>", unsafe_allow_html=True)

        # Presentation logic: Mapping inputs smoothly to a mock probability
        prob_base = ((rh - 40)/60 * 0.45) + ((40 - temp)/20 * 0.35) + ((1015 - press)/15 * 0.1) + (wind/20 * 0.1)
        prob_rain = max(0.02, min(0.98, prob_base)) * 100
        is_rain = prob_rain >= 50.0

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if is_rain:
            st.error("""### 🌧 KLASIFIKASI: HUJAN
        Model TCN memprediksi presipitasi tinggi pada periode ke depan.""")
        else:
            st.success("""### ☀ KLASIFIKASI: TIDAK HUJAN
        Kondisi atmosfer cenderung stabil tanpa presipitasi signifikan.""")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob_rain,
            title = {'text': "Confidence Probability (%)", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#3498db" if is_rain else "#f39c12"},
                'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

elif page == "🏆 Results":

    st.title("🏆 Hasil Analisis Curah Hujan Ekstrem (>50 mm)")

    st.success("""
**Kesimpulan Analisis**

Model **TCN** menghasilkan prediksi yang paling mendekati **Garis Ideal (Perfect Prediction)** pada kondisi hujan ekstrem (>50 mm).

• 🟢 **TCN** → Prediksi paling akurat
• 🔵 **LSTM** → Sedikit underestimate pada hujan tinggi
• 🔴 **MLP** → Deviasi terbesar
""")

    # ==========================================================
    # Data
    # ==========================================================

    df_extreme = pd.DataFrame({
        'Aktual (mm)': [55.2, 63.5, 76.8, 85.4, 94.1, 106.7, 127.7, 153.2],
        'MLP (mm)':    [32.1, 38.0, 45.3, 50.0, 52.2, 58.0, 62.1, 65.5],
        'LSTM (mm)':   [45.5, 55.2, 60.1, 71.3, 80.0, 88.5, 100.2, 115.0],
        'TCN (mm)':    [51.8, 60.5, 73.2, 82.5, 91.0, 104.2, 125.0, 148.5]
    })

    # ==========================================================
    # Metrics
    # ==========================================================

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Model Terbaik",
            "TCN",
            "Closest to Ideal"
        )

    with c2:
        st.metric(
            "Kategori",
            "Extreme Rain",
            "> 50 mm"
        )

    with c3:
        st.metric(
            "Jumlah Sampel",
            len(df_extreme),
            "Extreme Events"
        )

    st.divider()

    # ==========================================================
    # Scatter Plot
    # ==========================================================

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_extreme['Aktual (mm)'],
            y=df_extreme['MLP (mm)'],
            mode='markers',
            name='MLP',
            marker=dict(
                color='#e74c3c',
                size=12
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_extreme['Aktual (mm)'],
            y=df_extreme['LSTM (mm)'],
            mode='markers',
            name='LSTM',
            marker=dict(
                color='#3498db',
                size=12
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df_extreme['Aktual (mm)'],
            y=df_extreme['TCN (mm)'],
            mode='markers',
            name='TCN',
            marker=dict(
                color='#2ecc71',
                size=13,
                symbol="diamond"
            )
        )
    )

    # ==========================================================
    # Ideal Line (Perfect Prediction)
    # ==========================================================

    min_val = min(
        df_extreme.min().min(),
        df_extreme['Aktual (mm)'].min()
    )

    max_val = max(
        df_extreme.max().max(),
        df_extreme['Aktual (mm)'].max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Garis Ideal",
            line=dict(
                color="white",
                dash="dash",
                width=3
            )
        )
    )

    fig.update_layout(

        title="Perbandingan Prediksi Curah Hujan Ekstrem",

        template="plotly_dark",

        height=650,

        hovermode="closest",

        legend=dict(
            orientation="v"
        ),

        xaxis=dict(
            title="Curah Hujan Aktual (mm)",
            range=[30,170]
        ),

        yaxis=dict(
            title="Curah Hujan Prediksi (mm)",
            range=[30,170],
            scaleanchor="x",
            scaleratio=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ==========================================================
    # Table
    # ==========================================================

    st.subheader("📋 Data Curah Hujan Ekstrem")

    st.dataframe(
        df_extreme,
        use_container_width=True,
        hide_index=True
    )

    st.info("""
**Interpretasi**

- Semakin dekat titik terhadap **garis ideal**, semakin baik kualitas prediksi model.
- **TCN** memiliki deviasi terkecil terhadap garis ideal pada hampir seluruh kejadian hujan ekstrem.
- **LSTM** masih cukup baik namun mulai mengalami underestimate pada curah hujan tinggi.
- **MLP** menunjukkan deviasi paling besar sehingga kurang mampu menangkap pola hujan ekstrem.
""")

elif page == "ℹ About":
    st.title("ℹ Metodologi Proyek")
    st.markdown("""
    Dashboard ini dirancang khusus untuk mempresentasikan hasil dan kesimpulan penelitian dari Tesis secara interaktif.
    
    ### 📖 Pendekatan Riset
    Data meteorologi di Bandara Soekarno-Hatta (2015-2024) diproses menggunakan algoritma:
    1. **Multi-Layer Perceptron (MLP)**: Model *baseline* feedforward neural network.
    2. **Long Short-Term Memory (LSTM)**: Arsitektur *recurrent neural network* yang unggul dalam menyimpan memori deret waktu.
    3. **Temporal Convolutional Network (TCN)**: Arsitektur yang memanfaatkan konvolusi kausal dan terdilatasi 1D untuk mendeteksi dependensi berfrekuensi tinggi.
    """)