import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import datetime
import os

# Set page title
st.set_page_config(page_title="Stock Price Prediction", layout="centered")
st.title("📈 Stock Price Predictor - Tata Global Beverages")

# Load the data
DATA_PATH = "stock_data/stock_data"
FILE_NAME = "Tata-Global-Beverages.csv"  # adjust based on your actual filename
file_path = os.path.join(DATA_PATH, FILE_NAME)

@st.cache_data
def load_data():
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()
st.write("### Raw Stock Data")
st.write(df.tail())

# Plot the closing price
st.write("### Closing Price Chart")
st.line_chart(df['Close'])

# Preprocess for prediction (sample logic)
def prepare_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    return X, y, scaler

X, y, scaler = prepare_data(df)

# Load model (placeholder path — update if needed)
MODEL_PATH = "models/stock_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

    # Make predictions
    predicted = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted)

    # Plot predictions
    st.write("### Predicted vs Actual")
    actual = df['Close'].values[-len(predicted_prices):]
    result_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted_prices.flatten()}, index=df.index[-len(predicted_prices):])
    st.line_chart(result_df)
else:
    st.warning(f"Model file not found at `{MODEL_PATH}`. Please upload a trained model to use predictions.")
