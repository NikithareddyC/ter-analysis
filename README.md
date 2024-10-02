# TER Stock Analysis and Prediction Dashboard 📈

This project is an interactive stock analysis and prediction dashboard for **Teradyne, Inc. (TER)**. The dashboard provides detailed insights into historical stock performance, calculates key financial metrics, and forecasts future stock prices using a Long Short-Term Memory (LSTM) neural network. It is designed to help investors and analysts make informed decisions about TER stock.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Installation and Setup](#installation-and-setup)
5. [How to Use the Dashboard](#how-to-use-the-dashboard)
6. [Screenshots](#screenshots)
7. [Model and Analysis Summary](#model-and-analysis-summary)
8. [Future Enhancements](#future-enhancements)
9. [Contact](#contact)

---

## Project Overview
The **TER Stock Analysis Dashboard** provides an intuitive and interactive way to explore historical data and gain insights into the performance of Teradyne, Inc. (TER) stock. With built-in features for visualizing trends, trading volume, volatility, and future price predictions, this dashboard serves as a powerful tool for both novice and experienced investors.

**Purpose**: The primary objective is to use historical stock data to analyze patterns and trends, predict future prices using an LSTM model, and present findings through an interactive Streamlit dashboard.

## Features
- **Historical Stock Price Analysis**: Visualize closing prices, moving averages (20-day and 50-day), and trading volume.
- **Volatility Analysis**: Calculate and display 30-day annualized volatility to understand stock risk.
- **LSTM-Based Price Predictions**: Forecast future stock prices using an LSTM model for a user-selected period.
- **Interactive Date Range Selection**: Customize the analysis by selecting specific date ranges.
- **Dynamic Forecast Periods**: Change the forecast duration between 30 and 180 days.
- **User-Friendly Dashboard**: An interactive Streamlit interface for easy exploration and analysis.

## Technologies Used
- **Programming Language**: Python
- **Data Analysis Libraries**: `pandas`, `numpy`
- **Machine Learning Framework**: `tensorflow` (LSTM Model)
- **Data Visualization**: `matplotlib`, `streamlit`
- **Data Source**: `yfinance` (Yahoo Finance API)

## Installation and Setup
### Prerequisites:
Make sure you have Python installed. You can download it [here](https://www.python.org/downloads/).

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/ter_stock_analysis.git
2. **Navigate to the Project Directory**:
   ```bash
   cd ter_stock_analysis
3. **Install the Required Libraries**: Install all the necessary dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit Dashboard**: Once the dependencies are installed, you can run the Streamlit dashboard using the command
   ```bash
   streamlit run ter_stock_dashboard.py
5. **View the Dashboard**: After running the command above, open your browser and go to
   ```arduino
   http://localhost:8501
**Model and Analysis Summary**

1. LSTM Model for Time Series Forecasting:
The LSTM model was trained using a 60-day lookback period on normalized stock prices.
The model captures sequential patterns and dependencies in the data, allowing for accurate short-term price forecasting.
2. Volatility Analysis:
High volatility indicates higher risk, while lower volatility suggests stable stock behavior.
TER stock shows moderate volatility over the analyzed period.
3. Cumulative Returns:
Cumulative returns provide a long-term view of the stock’s performance.
These returns are calculated based on daily price movements, reflecting the stock’s growth over time.
