import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import plotly.express as px

# Header formatting using Markdown and CSS
header_style = """
    <style>
        .header {
            color: #0077b6;
            font-size: 48px;
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.15);
        }
        .subheader {
            color: #0077b6;
            font-size: 30px;
        }
    </style>
"""

# Apply header formatting to the title
st.markdown(header_style, unsafe_allow_html=True)
st.markdown('<h1 class="header">Stock Price Forecast</h1>', unsafe_allow_html=True)

# Define content for the sidebar
sidebar_content = """
    <div style="background-color: #0077b6; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: #fff; font-size: 24px;">Stock Price Forecast</h1>
        <p style="color: #fff; font-size: 16px;">Powered by LSTM Modeling</p>
    </div>
    
    ## About This App
    
    This web app utilizes a pretrained LSTM model to forecast stock prices for the next 30 days.

    ### Disclaimer
    
    - The information provided is for educational and demonstration purposes only.
    - It should not be considered as financial advice.
    - Always consult with a qualified financial advisor before making investment decisions.
    - Past performance is not indicative of future results.
    
    ### Contact Information
    
    For questions and inquiries, please contact:
    
    [![Email](https://img.shields.io/badge/Email-olanreakinbo@gmail.com-informational)](mailto:olanreakinbo@gmail.com)
    
     ### Read About the App
    
    [![Medium](https://img.shields.io/badge/Read%20on%20Medium-Click%20Here-blue)](https://medium.com/your-medium-article-url)
"""

# Add a styled sidebar
st.sidebar.markdown(sidebar_content, unsafe_allow_html=True)

# User input for stock symbol
stock_symbol = st.text_input("Enter the stock symbol (e.g. TSLA for Tesla):", 'TSLA')

# Validate the stock symbol
if stock_symbol:
    ticker = yf.Ticker(stock_symbol)
    try:
        ticker_info = ticker.info
    except:
        st.warning(f"'{stock_symbol}' is not a valid stock ticker symbol. Please enter a valid ticker.")

# Fetch historical stock data using yfinance
start_date = st.date_input("Enter the start date:", datetime.date(2013, 1, 1))
end_date = datetime.datetime.now()
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Quick Stock Quote Assessment
st.markdown('<h2 class="subheader">Stock Quote</h2>', unsafe_allow_html=True)

# Fetch stock quote using yfinance
quote_data = yf.Ticker(stock_symbol)
quote_info = quote_data.info

# Display relevant information in a tabular format
quote_table = {
    "Category": ["Company Name", "Current Stock Price", "Change Perecentage", "Open Price", "High Price", "Low Price",
                 "Volume", "Market Capitalization", "52-Week Range", "Dividend Yield", "P/E", "EPS"],
    "Value": [quote_info.get('longName', 'N/A'),
              f"${quote_info.get('currentPrice', 'N/A'):.2f}" if isinstance(quote_info.get('currentPrice'), float) else 'N/A',
              f"{quote_info.get('regularMarketChangePercent', 'N/A'):.2%}" if quote_info.get('regularMarketChangePercent') is not None else 'N/A',
              f"${quote_info.get('open', 'N/A'):.2f}" if isinstance(quote_info.get('open'), float) else 'N/A',
              f"${quote_info.get('dayHigh', 'N/A'):.2f}" if isinstance(quote_info.get('dayHigh'), float) else 'N/A',
              f"${quote_info.get('dayLow', 'N/A'):.2f}" if isinstance(quote_info.get('dayLow'), float) else 'N/A',
              f"{quote_info.get('regularMarketVolume', 'N/A') / 1000000:.2f}M" if isinstance(quote_info.get('regularMarketVolume'), int) else 'N/A',
              f"${quote_info.get('marketCap', 'N/A'):,}" if isinstance(quote_info.get('marketCap'), int) else 'N/A',
              f"${quote_info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${quote_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}" if isinstance(quote_info.get('fiftyTwoWeekLow'), float) and isinstance(quote_info.get('fiftyTwoWeekHigh'), float) else 'N/A',
              f"{quote_info.get('dividendYield', 'N/A'):.2%}" if quote_info.get('dividendYield') is not None else 'N/A',
              quote_info.get('trailingPE', 'N/A'),
              quote_info.get('trailingEps', 'N/A')]
}

quote_table_df = pd.DataFrame(quote_table)
quote_table_df.index = range(1, len(quote_table_df) + 1)
st.table(quote_table_df)

# Visualize Stock Price
st.markdown('<h2 class="subheader">Stock Prices Over Time</h2>', unsafe_allow_html=True)

# Plot stock prices using Plotly
fig = px.line(stock_data, x=stock_data.index, y='Close')
st.plotly_chart(fig)

# Visualize Technical Indicators
st.markdown('<h2 class="subheader">Technical Indicators</h2>', unsafe_allow_html=True)

# Create a horizontal slider to navigate through different indicators
selected_indicator = st.selectbox("Select Indicator", ["SMA", "EMA", "RSI", "MACD"])

if selected_indicator == "SMA":
    # Plot 50-day and 200-day Simple Moving Averages
    sma_50 = stock_data['Close'].rolling(window=50).mean()
    sma_200 = stock_data['Close'].rolling(window=200).mean()

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price'))
    fig_sma.add_trace(go.Scatter(x=sma_50.index, y=sma_50, mode='lines', name='50-day SMA', line=dict(color='green')))
    fig_sma.add_trace(go.Scatter(x=sma_200.index, y=sma_200, mode='lines', name='200-day SMA', line=dict(color='orange')))
    fig_sma.update_layout(title='Simple Moving Averages (SMA)', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_sma)

elif selected_indicator == "EMA":
    # Plot 50-day and 200-day Exponential Moving Averages
    ema_50 = stock_data['Close'].ewm(span=50, adjust=False).mean()
    ema_200 = stock_data['Close'].ewm(span=200, adjust=False).mean()

    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Stock Price'))
    fig_ema.add_trace(go.Scatter(x=ema_50.index, y=ema_50, mode='lines', name='50-day EMA', line=dict(color='green')))
    fig_ema.add_trace(go.Scatter(x=ema_200.index, y=ema_200, mode='lines', name='200-day EMA', line=dict(color='orange')))
    fig_ema.update_layout(title='Exponential Moving Averages (EMA)', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_ema)

elif selected_indicator == "RSI":
    # Plot Relative Strength Index (RSI)
    rsi_period = 14
    delta = stock_data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=rsi, mode='lines', name=f'RSI ({rsi_period}-day)'))
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=[70] * len(stock_data), mode='lines', name='Overbought (70)', line=dict(color='red', dash='dash')))
    fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=[30] * len(stock_data), mode='lines', name='Oversold (30)', line=dict(color='green', dash='dash')))
    fig_rsi.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI Value')
    st.plotly_chart(fig_rsi)

elif selected_indicator == "MACD":
    # Plot Moving Average Convergence Divergence (MACD)
    short_period = 12
    long_period = 26
    ema_short = stock_data['Close'].ewm(span=short_period, adjust=False).mean()
    ema_long = stock_data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=macd_line, mode='lines', name='MACD Line'))
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange')))
    fig_macd.add_trace(go.Bar(x=stock_data.index, y=macd_histogram, name='MACD Histogram', marker_color='grey'))
    fig_macd.add_trace(go.Scatter(x=stock_data.index, y=[0] * len(stock_data), mode='lines', name='Zero Line', line=dict(color='black', dash='dash')))
    fig_macd.update_layout(title='Moving Average Convergence Divergence (MACD)', xaxis_title='Date', yaxis_title='MACD Value')
    st.plotly_chart(fig_macd)
    
# Load the pre-trained LSTM model
model = load_model('model.keras')

# Data Preprocessing
sequence_length = 100
scaler = MinMaxScaler(feature_range=(0,1))
combined_data_scaled = scaler.fit_transform(stock_data[['Close']])

# Select a number of forecast days
forecast_days = 30

# Make forecasts
forecast = []
for _ in range(forecast_days):
    next_pred = model.predict(combined_data_scaled[-sequence_length:].reshape(1, sequence_length, 1))
    forecast.append(next_pred[0, 0])
    combined_data_scaled = np.roll(combined_data_scaled, -1)
    combined_data_scaled[-1] = next_pred[0, 0]

# Scale back the forecasted prices correctly
scale = 1 / scaler.scale_[0]
forecast = np.array(forecast) * scale

# Create date range for the forecast
forecast_dates = pd.date_range(start=stock_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

# Visualize Forecasted Price
st.markdown('<h2 class="subheader">Forecasted Prices for the next 30 days</h2>', unsafe_allow_html=True)

# Plot forecasted prices using Plotly
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Prices': forecast})
fig = px.line(forecast_df, x='Date', y='Forecasted Prices')
st.plotly_chart(fig)
