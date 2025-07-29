import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import requests
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
from tcn import TCN

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
    try:
        # Beri tahu Keras tentang layer kustom 'TCN' saat memuat model
        custom_objects = {'TCN': TCN}
        model = tf.keras.models.load_model(
            'model_tcn_bilstm_gru.h5',
            custom_objects=custom_objects,
            compile=False  # Tidak perlu compile model saat hanya untuk prediksi
        )
        scaler = joblib.load('scaler_btc.save')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model atau scaler: {e}")
        return None, None

# ==============================================================================
# Fungsi untuk Mengambil Data dari CoinGecko API
# ==============================================================================
@st.cache_data(ttl=600) # Cache data selama 10 menit untuk mengurangi panggilan API
def get_coingecko_data(days=90):
    """Mengambil data OHLCV Bitcoin dari CoinGecko untuk beberapa hari terakhir."""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days={days}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Akan error jika status code bukan 200
        data = response.json()
        
        # Mengubah data menjadi DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('Date')
        df = df[['open', 'high', 'low', 'close']]
        df.columns = ['Open', 'High', 'Low', 'Close']
        return df, None # Mengembalikan data dan tidak ada error
    except requests.exceptions.RequestException as e:
        error_message = f"Gagal mengambil data dari CoinGecko: {e}"
        st.error(error_message)
        return None, error_message # Mengembalikan None dan pesan error

# ==============================================================================
# Judul dan Sidebar Aplikasi
# ==============================================================================
st.title("📈 Prediksi Harga Bitcoin Real-time")
st.markdown("Aplikasi ini menampilkan harga OHLCV Bitcoin secara *real-time* dan memprediksi harga penutupan untuk hari berikutnya menggunakan model TCN-BiLSTM-GRU.")

with st.sidebar:
    st.header("Pengaturan")
    st.info(
        "Model ini menggunakan data harga penutupan selama 60 hari terakhir "
        "untuk memprediksi harga penutupan pada hari berikutnya."
    )
    days_to_fetch = st.slider("Jumlah hari data historis:", 61, 365, 90)
    st.markdown("---")
    st.markdown("Dibuat dengan [Streamlit](https://streamlit.io) dan [CoinGecko API](https://www.coingecko.com/en/api).")


# ==============================================================================
# Memuat Model dan Data
# ==============================================================================
model, scaler = load_model_and_scaler()
data, api_error = get_coingecko_data(days=days_to_fetch)

# ==============================================================================
# Logika Utama Aplikasi
# ==============================================================================
# Cek error sebelum melanjutkan
if api_error:
    st.warning(
        "Terjadi masalah saat mengambil data harga dari CoinGecko. "
        "Ini mungkin karena terlalu banyak permintaan ke API. "
        "Silakan tunggu beberapa saat sebelum mencoba lagi."
    )
elif model is None or scaler is None:
    st.warning("Tidak dapat melanjutkan karena model atau scaler gagal dimuat.")
elif data is None or data.empty:
     st.warning("Data tidak tersedia atau kosong. Tidak dapat melanjutkan.")
else:
    # ==========================================================================
    # Tampilkan Data Real-time
    # ==========================================================================
    st.subheader("Tabel Harga Bitcoin (USD)")
    st.dataframe(data.tail(10).style.format("{:.2f}"))

    # ==========================================================================
    # Proses Prediksi
    # ==========================================================================
    with st.spinner('Melakukan prediksi...'):
        # 1. Ambil 60 data terakhir dari kolom 'Close'
        last_60_days = data['Close'].values[-60:]

        # 2. Lakukan penskalaan data
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))

        # 3. Reshape data menjadi input untuk model [1, 60, 1]
        X_pred = np.reshape(last_60_days_scaled, (1, 60, 1))

        # 4. Lakukan prediksi
        predicted_price_scaled = model.predict(X_pred)

        # 5. Kembalikan harga prediksi ke skala semula
        predicted_price = scaler.inverse_transform(predicted_price_scaled)
        
        # Ambil harga penutupan terakhir
        last_close_price = data['Close'].iloc[-1]
        
        # Hitung persentase perubahan
        change_percent = ((predicted_price[0][0] - last_close_price) / last_close_price) * 100
        
    # ==========================================================================
    # Tampilkan Hasil Prediksi
    # ==========================================================================
    st.subheader("Hasil Prediksi Harga Penutupan Bitcoin untuk Besok")
    
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    st.metric(
        label=f"Prediksi untuk {tomorrow}",
        value=f"${predicted_price[0][0]:,.2f}",
        delta=f"{change_percent:.2f}%"
    )

    # ==========================================================================
    # Visualisasi Grafik
    # ==========================================================================
    st.subheader("Grafik Harga Penutupan Historis & Prediksi")
    
    # Buat DataFrame untuk prediksi
    pred_date = data.index[-1] + timedelta(days=1)
    prediction_df = pd.DataFrame({'Close': [predicted_price[0][0]]}, index=[pred_date])

    # Buat plot
    fig = go.Figure()

    # Tambahkan data historis
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'],
        mode='lines',
        name='Harga Historis',
        line=dict(color='royalblue', width=2)
    ))

    # Tambahkan titik prediksi
    fig.add_trace(go.Scatter(
        x=prediction_df.index,
        y=prediction_df['Close'],
        mode='markers',
        name='Harga Prediksi',
        marker=dict(color='orange', size=10, symbol='star')
    ))
    
    # Tambahkan garis putus-putus yang menghubungkan harga terakhir ke prediksi
    fig.add_trace(go.Scatter(
        x=[data.index[-1], prediction_df.index[0]],
        y=[data['Close'].iloc[-1], prediction_df['Close'].iloc[0]],
        mode='lines',
        name='Tren Prediksi',
        line=dict(color='orange', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Pergerakan Harga Penutupan Bitcoin',
        xaxis_title='Tanggal',
        yaxis_title='Harga (USD)',
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)
