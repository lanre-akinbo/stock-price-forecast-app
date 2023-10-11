# Stock Price Prediction Web App

## Overview
This project aims to predict stock prices using machine learning techniques and provide visualizations for technical indicators. This web application is built with Streamlit and powered by a pretrained LSTM model to forecast stock prices for the next 30 days.

## Technologies Used
- Python
- Streamlit
- TensorFlow
- Pandas
- NumPy
- Matplotlib

## Features
- Historical stock price data visualization.
- LSTM model-based forecasting for the next 30 days.
- User-friendly interface.

## Getting Started
### Prerequisites
Make sure you have the following prerequisites installed:
- Python 3.9
- Required Python libraries (provided in requirements.txt)

### Installation
1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/lanre-akinbo/stock-price-forecast-app.git
   cd stock-price-forecast-app
   ```

2. Install the required Python libraries.
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app.
   ```bash
   streamlit run app.py
   ```

## Usage
- Open the app in your browser.
- Enter a valid stock ticker symbol.
- Choose a start date.
- Explore historical TSLA stock prices.
- View technical indicators.
- Check the forecasted prices for the next 30 days.
- Gain insights into potential price trends.

## Demo
A live demo of this project is available [here](https://stock-price-forecast-app.streamlit.app/).

## Project Structure
- `app.py`: The main Streamlit application.
- 'LSTM Model.ipynb': The training of the machine learning model.
- `model.h5`: The trained model.
- `requirements.txt`: List of Python libraries used in the project.

## Contributing
Contributions are welcome. Please check the [contributing guidelines](CONTRIBUTING.md).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Data provided by Yahoo Finance.
- Inspired by various deep learning tutorials and resources.
