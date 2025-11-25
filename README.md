Stock Price Prediction Using LSTM

This project predicts stock prices using an LSTM neural network and displays results in a Streamlit web app.

 Features
- Fetches historical stock data using yFinance
- Uses Adjusted Close price for training
- LSTM learns stock trends
- Shows:
  - Adjusted Close price
  - 100/250 Day Moving Averages
  - Test vs Predicted graph
  - Next 7-day forecast

 Run locally:
pip install -r requirements.txt
streamlit run app2.py

## Deployment:
Push files to GitHub → Visit share.streamlit.io → Select repo → Deploy.