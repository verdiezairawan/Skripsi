import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from tcn import TCN
import os

# ==============================================================================
# Konfigurasi Halaman Streamlit
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Harga Bitcoin Real-time",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Fungsi untuk Memuat Model dan Scaler
# ==============================================================================
@st.cache_resource
def load_model_and_scaler():
    """Memuat model Keras dan scaler dari file."""
    model_path = 'model_tcn_bilstm_gru.h5'
    scaler_path = 'scaler_btc.save'

    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di path: {model_path}")
        return None, None
    if not os.path.exists(scaler_path):
        st.error(f"File scaler tidak ditemukan di path: {scaler_path}")
        return None, None

    try:
        custom_objects = {'TCN': TCN}
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler. Detail kesalahan:")
        st.exception(e)
        return None, None

# ==============================================================================
# Fungsi untuk Mengambil Data dari CoinGecko API
# ==============================================================================
@st.cache_data(ttl=600) # Cache data selama 10 menit
def get_coingecko_data(days=365): # -> Nilai default diubah menjadi 365
    """Mengambil data OHLCV Bitcoin dari CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date')
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
        return df, None
    except requests.exceptions.RequestException as e:
        error_message = f"Gagal mengambil data dari CoinGecko: {e}"
        st.error(error_message)
        return None, error_message

# ==============================================================================
# Judul dan Sidebar Aplikasi
# ==============================================================================
st.title("📈 Prediksi Harga Bitcoin Real-time")
st.markdown("Aplikasi ini menampilkan harga OHLCV Bitcoin dan memprediksi harga penutupan untuk hari berikutnya menggunakan model TCN-BiLSTM-GRU.")

with st.sidebar:
    st.header("Pengaturan")
    st.info(
        "Aplikasi ini secara otomatis mengambil data harga selama 365 hari terakhir "
        "untuk membuat prediksi."
    )
    st.markdown("---")
    st.markdown("Dibuat dengan [Streamlit](https://streamlit.io) dan [CoinGecko API](https://www.coingecko.com/en/api).")


# ==============================================================================
# Memuat Model dan Data
# ==============================================================================
model, scaler = load_model_and_scaler()
# -> Langsung panggil fungsi dengan 365 hari, tanpa slider
data, api_error = get_coingecko_data(days=365)

# ==============================================================================
# Logika Utama Aplikasi
# ==============================================================================
if model is None or scaler is None:
    st.warning("Aplikasi tidak dapat berjalan karena model atau scaler gagal dimuat. Silakan periksa pesan kesalahan di atas.")
elif api_error:
    st.warning(f"Aplikasi tidak dapat berjalan karena gagal mengambil data. Kesalahan: {api_error}")
elif data is None or data.empty:
     st.warning("Data tidak tersedia atau kosong. Tidak dapat melanjutkan.")
else:
    st.subheader("Tabel Harga Bitcoin (USD)")
    st.dataframe(data.tail(10).style.format("{:.2f}"))

    with st.spinner('Melakukan prediksi...'):
        last_60_days = data['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))
        predicted_price_scaled = model.predict(X_pred)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        last_close_price = data['Close'].iloc[-1]
        change_percent = ((predicted_price[0][0] - last_close_price) / last_close_price) * 100
        
    st.subheader("Hasil Prediksi Harga Penutupan Bitcoin untuk Besok")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    st.metric(
        label=f"Prediksi untuk {tomorrow}",
        value=f"${predicted_price[0][0]:,.2f}",
        delta=f"{change_percent:.2f}%"
    )

    st.subheader("Grafik Harga Penutupan Historis & Prediksi")
    pred_date = data.index[-1] + timedelta(days=1)
    prediction_df = pd.DataFrame({'Close': [predicted_price[0][0]]}, index=[pred_date])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'], mode='lines', name='Harga Historis',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=prediction_df.index, y=prediction_df['Close'], mode='markers', name='Harga Prediksi',
        marker=dict(color='orange', size=10, symbol='star')
    ))
    fig.add_trace(go.Scatter(
        x=[data.index[-1], prediction_df.index[0]],
        y=[data['Close'].iloc[-1], prediction_df['Close'].iloc[0]],
        mode='lines', name='Tren Prediksi',
        line=dict(color='orange', width=2, dash='dash')
    ))
    fig.update_layout(
        title='Pergerakan Harga Penutupan Bitcoin',
        xaxis_title='Tanggal', yaxis_title='Harga (USD)',
        xaxis_rangeslider_visible=True, template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
