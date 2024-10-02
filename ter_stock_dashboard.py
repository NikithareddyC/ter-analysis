# ter_stock_dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Streamlit Page Configuration
st.set_page_config(page_title="TER Stock Analysis Dashboard", layout="wide")

# Sidebar Section for User Input
st.sidebar.header("User Input Options")
selected_stock = "TER"  # Teradyne, Inc. stock ticker
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2016-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())
forecast_days = st.sidebar.slider("Number of Days to Forecast", min_value=30, max_value=180, step=30)

# Fetch Stock Data
@st.cache_data
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

# Load the Stock Data
stock_data = fetch_data(selected_stock, start_date, end_date)

# Calculate Additional Metrics
stock_data['Daily Return'] = stock_data['Close'].pct_change()
stock_data['20-Day MA'] = stock_data['Close'].rolling(window=20).mean()
stock_data['50-Day MA'] = stock_data['Close'].rolling(window=50).mean()
stock_data['Log Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
stock_data['30-Day Volatility'] = stock_data['Log Return'].rolling(window=30).std() * np.sqrt(252)
stock_data['Cumulative Return'] = (1 + stock_data['Daily Return']).cumprod()

# Display the Stock Data
st.header(f"Stock Data for {selected_stock}")
st.write(stock_data.tail())

# Create Columns for the Main Charts
st.markdown("---")  # Add a separator line for visual separation
col1, col2 = st.columns(2)  # Create two equal-sized columns

# Plot Historical Prices and Moving Averages in the First Column
with col1:
    st.subheader(f"{selected_stock} Stock Price with Moving Averages")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue')
    ax.plot(stock_data.index, stock_data['20-Day MA'], label='20-Day MA', color='green')
    ax.plot(stock_data.index, stock_data['50-Day MA'], label='50-Day MA', color='red')
    ax.set_title(f'{selected_stock} Stock Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)

# Plot Trading Volume in the Second Column
with col2:
    st.subheader(f"{selected_stock} Trading Volume Over Time")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(stock_data.index, stock_data['Volume'], color='orange')
    ax2.set_title(f'{selected_stock} Trading Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    st.pyplot(fig2)

# Volatility Analysis and Cumulative Returns in Expander Section
with st.expander("Additional Analysis"):
    col3, col4 = st.columns(2)

    with col3:
        st.subheader(f"{selected_stock} 30-Day Annualized Volatility")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(stock_data.index, stock_data['30-Day Volatility'], label='30-Day Volatility', color='purple')
        ax3.set_title(f'{selected_stock} 30-Day Annualized Volatility')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Volatility')
        st.pyplot(fig3)

    with col4:
        st.subheader(f"{selected_stock} Cumulative Returns Over Time")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        ax4.plot(stock_data.index, stock_data['Cumulative Return'], label='Cumulative Return', color='brown')
        ax4.set_title(f'{selected_stock} Cumulative Returns')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative Return')
        st.pyplot(fig4)

# Preparing Data for LSTM Model
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Split the Data into Training and Testing Sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]

# Creating the Training Dataset
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build and Train the LSTM Model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Forecasting Future Stock Prices
last_60_days = scaled_data[-60:]
future_input = last_60_days.reshape(1, last_60_days.shape[0], 1)
future_predictions = []

for _ in range(forecast_days):
    future_pred = model.predict(future_input)
    reshaped_pred = np.reshape(future_pred, (1, 1, 1))
    future_input = np.append(future_input[:, 1:, :], reshaped_pred, axis=1)
    future_predictions.append(future_pred[0, 0])

# Transform and Display Future Price Predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
last_date = stock_data.index[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days, freq='B')
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Price'])

# Display Future Projections
st.subheader("Future Price Projections")
st.line_chart(future_df)
