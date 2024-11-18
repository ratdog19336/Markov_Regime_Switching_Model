import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
# from fredapi import Fred
import sqlite3
# import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib import gridspec
# import matplotlib.dates as mdates
from pandas.tseries.offsets import Day
from decimal import Decimal
import requests
# import seaborn as sns
# color_pal = sns.color_palette()
from pandas.tseries.offsets import BDay
# from multiprocessing import Pool
# from tqdm import tqdm
from IPython.display import display
from IPython.display import HTML, display

# Function to get the next trading day
def get_next_trading_day():
    today = datetime.today()
    next_trading_day = today + BDay(1)
    return next_trading_day

# Parameters
ticker = "^GSPC"
start_date = "1950-01-01"
end_date = get_next_trading_day().strftime('%Y-%m-%d')
end_date
data = yf.download(ticker, start=start_date, end=end_date)
print(data)

data['Index_Returns'] = data['Adj Close'].pct_change()
data.dropna(inplace=True)
print(data)

# Check for NaNs in 'Index_Returns' and drop them
returns = data['Index_Returns'].dropna()
print(returns)

# Fit Markov Switching Model
model = MarkovRegression(returns, k_regimes=2, trend='c', switching_variance=True)
result = model.fit()
print(result.summary())

# Add regime to the data
data.loc[returns.index, 'Vol_Regime'] = result.smoothed_marginal_probabilities.idxmax(axis=1)

# Extract smoothed probabilities and last known state probabilities
smoothed_probs = result.smoothed_marginal_probabilities
smoothed_probs
last_probs = smoothed_probs.iloc[-1].values
print(last_probs)

# Extract transition probabilities from the model parameters
params = result.params
p_00 = params['p[0->0]']
p_10 = params['p[1->0]']
p_01 = 1 - p_00
p_11 = 1 - p_10

# Construct the transition matrix
transition_matrix = np.array([
    [p_00, p_01],
    [p_10, p_11]
])
print(transition_matrix)

# Update state probabilities to predict the next day's regime
state_probs = np.dot(last_probs, transition_matrix)
print(state_probs)

# Determine the most likely regime at t+1
regime_labels = smoothed_probs.columns.tolist()  # Should be [0, 1]
predicted_most_likely_regime = regime_labels[np.argmax(state_probs)]

print(predicted_most_likely_regime)

# Assuming end_date is already defined as a string in the format 'YYYY-MM-DD'
end_date = datetime.strptime(end_date, '%Y-%m-%d')  # Convert to datetime object
predicted_date = (end_date + timedelta(days=0)).strftime('%Y-%m-%d')  # Increment and convert back to string
print(predicted_date)
print(data)

def triangular_moving_average(series, n):
    smoothed_series = series.rolling(window=n//2, min_periods=1).mean()
    smoothed_series = smoothed_series.rolling(window=n//2, min_periods=1).mean()
    return smoothed_series

# Calculate 250-day triangular moving average
data['250_TMA'] = triangular_moving_average(data['Adj Close'], 250)
data

# Define the four market regimes for 250 TMA
conditions = [
    (data['Vol_Regime'] == 1) & (data['Adj Close'] < data['250_TMA']),
    (data['Vol_Regime'] == 1) & (data['Adj Close'] >= data['250_TMA']),
    (data['Vol_Regime'] == 0) & (data['Adj Close'] < data['250_TMA']),
    (data['Vol_Regime'] == 0) & (data['Adj Close'] >= data['250_TMA']),
]
choices = [
    'Bearish High Variance',
    'Bullish High Variance',
    'Bearish Low Variance',
    'Bullish Low Variance'
]

# Specify a default value that matches the data type of choices
data['Market_Regime'] = np.select(conditions, choices, default='Unknown')
data

# Calculate 30-day and 60-day Triangular Moving Averages and shift by 1 day
data['30_TMA'] = triangular_moving_average(data['Adj Close'], 30).shift(1)
data['60_TMA'] = triangular_moving_average(data['Adj Close'], 60).shift(1)

# Define 30-Day and 60-Day Indicators
data['30_Day_Indicator'] = np.where(data['Adj Close'] > data['30_TMA'], 'Bullish', 'Bearish')
data['60_Day_Indicator'] = np.where(data['Adj Close'] > data['60_TMA'], 'Bullish', 'Bearish')
data

# Define initial exposure based on Adjusted_Market_Regime
exposure_mapping = {
    'Bullish Low Variance': 2.0,
    'Bearish Low Variance': 1.0,
    'Bullish High Variance': 1.0,
    'Bearish High Variance': 0.0
}
data['Portfolio_Exposure'] = data['Market_Regime'].map(exposure_mapping).fillna(1.0)  # Default exposure is 1.0 if regime is NaN

# Adjust exposure based on 30-Day and 60-Day Indicators
for index, row in data.iterrows():
    if row['Portfolio_Exposure'] == 2.0:
        if row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bearish':
            data.at[index, 'Portfolio_Exposure'] = 1.0
        elif row['30_Day_Indicator'] == 'Bullish' and row['60_Day_Indicator'] == 'Bearish':
            data.at[index, 'Portfolio_Exposure'] = 1.5
        elif row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bullish':
            data.at[index, 'Portfolio_Exposure'] = 1.5
            
# Adjust exposure based on 30-Day and 60-Day Indicators for exposure = 1.0 and Bearish Low Variance regime
for index, row in data.iterrows():
    if row['Portfolio_Exposure'] == 1.0 and row['Market_Regime'] == 'Bearish Low Variance':
        if row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bearish':
            data.at[index, 'Portfolio_Exposure'] = 0.0
        elif row['30_Day_Indicator'] == 'Bullish' and row['60_Day_Indicator'] == 'Bearish':
            data.at[index, 'Portfolio_Exposure'] = 1.0
        elif row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bullish':
            data.at[index, 'Portfolio_Exposure'] = 1.0
data

last_day_250_TMA = data['250_TMA'][-1]
last_day_adjusted_close = data['Adj Close'][-1]

# Define the four market regimes for 250 TMA
next_day_conditions = [
    
    predicted_most_likely_regime == 1 and last_day_adjusted_close < last_day_250_TMA,
    predicted_most_likely_regime == 1 and last_day_adjusted_close >= last_day_250_TMA,
    predicted_most_likely_regime == 0 and last_day_adjusted_close < last_day_250_TMA,
    predicted_most_likely_regime == 0 and last_day_adjusted_close >= last_day_250_TMA,
    
]
next_day_choices = [
    'Bearish High Variance',
    'Bullish High Variance',
    'Bearish Low Variance',
    'Bullish Low Variance'
]

next_day_market_regime = np.select(next_day_conditions, next_day_choices, default='Unknown')
print(next_day_market_regime)

next_day_30_TMA = data['30_TMA'][-1]
next_day_60_TMA = data['60_TMA'][-1]
next_30_Day_Indicator = np.where(last_day_adjusted_close > next_day_30_TMA, 'Bullish', 'Bearish')
next_60_Day_Indicator = np.where(last_day_adjusted_close > next_day_60_TMA, 'Bullish', 'Bearish')

print(next_day_30_TMA)
print(next_day_60_TMA)
print(next_30_Day_Indicator)
print(next_60_Day_Indicator)

# Example value of next_day_market_regime as a numpy array
next_day_market_regime = np.array(['Bullish Low Variance'])  # Single value stored as an array

# Define the exposure mapping
exposure_mapping = {
    'Bullish Low Variance': 2.0,
    'Bearish Low Variance': 1.0,
    'Bullish High Variance': 1.0,
    'Bearish High Variance': 0.0
}

# Extract the single value from the numpy array
next_day_market_regime = next_day_market_regime.item()  # Or use next_day_market_regime[0]

# Map the regime to exposure
next_day_exposure = exposure_mapping[next_day_market_regime]

print(f"Next Day Market Regime: {next_day_market_regime}")
print(f"Exposure: {next_day_exposure}")

# Adjust exposure based on 30-Day and 60-Day Indicators
for index, row in data.iterrows():
    if next_day_exposure == 2.0:
        if next_30_Day_Indicator == 'Bearish' and next_60_Day_Indicator == 'Bearish':
            next_day_exposure = 1.0  # Fix assignment here
        elif next_30_Day_Indicator == 'Bullish' and next_60_Day_Indicator == 'Bearish':
            next_day_exposure = 1.5
        elif next_30_Day_Indicator == 'Bearish' and next_60_Day_Indicator == 'Bullish':
            next_day_exposure = 1.5

# Adjust exposure based on 30-Day and 60-Day Indicators for exposure = 1.0 and Bearish Low Variance regime
for index, row in data.iterrows():
    if next_day_exposure == 1.0 and next_day_market_regime == 'Bearish Low Variance':
        if next_30_Day_Indicator == 'Bearish' and next_60_Day_Indicator == 'Bearish':
            next_day_exposure = 0.0
        elif next_30_Day_Indicator == 'Bullish' and next_60_Day_Indicator == 'Bearish':
            next_day_exposure = 1.0
        elif next_30_Day_Indicator == 'Bearish' and next_60_Day_Indicator == 'Bullish':
            next_day_exposure = 1.0
print(next_day_exposure)

#Telegram Messenger
# Telegram Bot API token and Channel ID
bot_token = '7328648943:AAH3gHyGf2xgjxBfzPd05F_7IagASgs-Dj0'
channel_id = '-1002309744206'

# Initialize the message variable each time the code runs with bold header
message = "<b>Your Daily Portfolio Exposure Update</b>\n\n"  # Reset message here
# labels = ["Tomorrow's Predicted Market Regime"]

# Parse the predicted_date and add 1 day
formatted_date = (datetime.strptime(predicted_date, '%Y-%m-%d') + timedelta(days=0)).strftime('%m/%d/%Y')

# Remove leading zeros from month and day
formatted_date = formatted_date.lstrip("0").replace("/0", "/")

print(f"Formatted Date: {formatted_date}")

# Add the labeled message for each row with line breaks for better formatting
message += f"<u>Tomorrow's Predicted Market Regime</u>\n"
message += f"<i>Date</i>: {formatted_date}\n"
message += f"<i>Adjusted Market Regime</i>: {next_day_market_regime}\n"
message += f"<i>Portfolio Exposure</i>: {next_day_exposure * 100:.0f}%\n\n" # Format Portfolio_Exposure as a percentage with 2 decimal places

# Telegram API URL
api_url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
# Payload to send with HTML formatting enabled
payload = {
    'chat_id': channel_id,
    'text': message,  # Combine with the rest of your message
    'parse_mode': 'HTML'  # Enables HTML for bold formatting
}
# Send the request
response = requests.post(api_url, json=payload)
# Check the response
if response.status_code == 200:
    print('Message sent successfully!')
else:
    print(f'Failed to send message. Error: {response.text}')

