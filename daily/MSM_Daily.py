import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from fredapi import Fred
import sqlite3
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import gridspec
import matplotlib.dates as mdates
from pandas.tseries.offsets import Day
from decimal import Decimal
import requests

def main():
    # Set up your FRED API key
    fred_api_key = 'bf9400b3f6a177d421bda60a77384789'  # Replace with your FRED API key
    fred = Fred(api_key=fred_api_key)
    
    # Define ticker and date range
    ticker = "^GSPC"
    edited_ticker = ticker.replace("^", "")
    start_date = "1950-01-01"
    end_date = (get_previous_trading_day() + timedelta(days=1)).strftime('%Y-%m-%d')
    # end_date = "2024-11-08"
    
    # Fetch market data
    market_data = fetch_market_data(ticker, start_date, end_date)
    print(market_data.tail())
    
    # Fetch FRED data
    fred_data = fetch_fred_data(start_date, end_date, fred)
    
    # Process data
    data = process_data(market_data, fred_data)
    
    # Calculate Markov regimes
    data, result = calculate_markov_regimes(data)
    
    # Define market regimes
    data = define_market_regimes(data)
    
    # Add Triangular Moving Averages and Indicators
    data = add_triangular_moving_averages_and_indicators(data)  # This must be called before calculate_exposures
    
    # Calculate exposures
    data = calculate_exposures(data)
    
    # Calculate returns and portfolio values
    data = calculate_returns_and_portfolio_values(data)
    
    
    # Output to SQLite database
    output_to_database(data)
    
    #Telegram Send
    telegram_messenger()

def get_previous_trading_day():
    today = datetime.now().date()
    previous_day = today - timedelta(days=1)
    
    while previous_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        previous_day -= timedelta(days=1)
    
    return previous_day

def fetch_market_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Create 'Syn_Open' column
    data['Syn_Open'] = data['Open']
    mask = (data['Open'] == 0) | (data['Open'].isna())
    data.loc[mask, 'Syn_Open'] = data.loc[mask, ['High', 'Low', 'Close']].mean(axis=1)
    
    # Use 'Syn_Open' where 'Open' is NaN or 0
    data['Adjusted_Open'] = data['Syn_Open']
    
    # Calculate daily returns and set the first day's return to 0
    data['Index_Returns'] = data['Adj Close'].pct_change().fillna(0)
    
    return data

def fetch_fred_data(start_date, end_date, fred):
    ffr_daily = fred.get_series('FEDFUNDS', start_date, end_date)
    ffr_10yr = fred.get_series('DGS10', start_date, end_date)
    tb3ms = fred.get_series('TB3MS', start_date, end_date)
    effr_daily = fred.get_series('EFFR', start_date, end_date)
    
    # Convert to DataFrame
    ffr_daily = pd.DataFrame(ffr_daily, columns=['FEDFUNDS'])
    ffr_10yr = pd.DataFrame(ffr_10yr, columns=['DGS10'])
    tb3ms = pd.DataFrame(tb3ms, columns=['TB3MS'])
    effr_daily = pd.DataFrame(effr_daily, columns=['EFFR'])
    
    # Resample to daily frequency and fill missing values
    ffr_daily = ffr_daily.resample('D').ffill()
    ffr_10yr = ffr_10yr.resample('D').ffill()
    tb3ms = tb3ms.resample('D').ffill()
    effr_daily = effr_daily.resample('D').ffill()
    
    fred_data = {
        'FEDFUNDS': ffr_daily,
        'DGS10': ffr_10yr,
        'TB3MS': tb3ms,
        'EFFR': effr_daily
    }
    
    return fred_data

def process_data(data, fred_data):
    # Merge S&P 500 data with Fed Funds Rate data
    data = data.join(fred_data['FEDFUNDS'])
    data = data.join(fred_data['DGS10'])
    data = data.join(fred_data['TB3MS'])
    data = data.join(fred_data['EFFR'])
    
    # Use TB3MS for dates before 1954-07-01, daily rate if available, otherwise use 10-year rate
    data['Effective_Fed_Rate'] = np.where(
        data.index < '1954-07-01',
        data['TB3MS'],
        data['FEDFUNDS']
    )
    data['Effective_Fed_Rate'] = np.where(
        data.index >= '2000-07-03',
        data['EFFR'],
        data['Effective_Fed_Rate']
    )
    data['Effective_Fed_Rate'] = data['Effective_Fed_Rate'].combine_first(data['DGS10'])
    
    # Handle NaN values by using the previous day's value
    data['Effective_Fed_Rate'] = data['Effective_Fed_Rate'].ffill()
    
    # Convert Effective Fed Rate to percentage format
    data['Effective_Fed_Rate'] = data['Effective_Fed_Rate'] / 100
    
    # Define IBKR Fee
    ibkr_fee = 0.0075  # 0.75% as a decimal\
    
    # Add IBKR Fee as a new column
    data['IBKR_Rate'] = ibkr_fee
    
    # Calculate Daily Leverage Rate
    data['Daily_Leverage_Rate'] = (data['Effective_Fed_Rate'] + ibkr_fee) / 360
    
    return data

def calculate_markov_regimes(data):
    # Check for NaNs in 'Index_Returns' and drop them
    returns = data['Index_Returns'].dropna()
    
    # Fit Markov Switching Model
    model = MarkovRegression(returns, k_regimes=2, trend='c', switching_variance=True)
    result = model.fit()
    print(result.summary())
    
    # Add regime to the data
    data.loc[returns.index, 'Vol_Regime'] = result.smoothed_marginal_probabilities.idxmax(axis=1)
    
    return data, result

def triangular_moving_average(series, n):
    smoothed_series = series.rolling(window=n//2, min_periods=1).mean()
    smoothed_series = smoothed_series.rolling(window=n//2, min_periods=1).mean()
    return smoothed_series

def define_market_regimes(data):
    # Calculate 250-day triangular moving average
    data['250_TMA'] = triangular_moving_average(data['Adj Close'], 250)
    
    # #Calculate Exponential Moving Average
    # data['EMA_100'] = data['Close'].ewm(span=250, adjust=False).mean()
    
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
    
    # Define adjusted market regimes with offset (shifted by 1 day)
    data['Adjusted_Market_Regime'] = data['Market_Regime'].shift(1)
    
    return data

def calculate_exposures(data):
    # Define initial exposure based on Adjusted_Market_Regime
    exposure_mapping = {
        'Bullish Low Variance': 2.0,
        'Bearish Low Variance': 1.0,
        'Bullish High Variance': 1.0,
        'Bearish High Variance': 0.0
    }
    data['Portfolio_Exposure'] = data['Adjusted_Market_Regime'].map(exposure_mapping).fillna(1.0)  # Default exposure is 1.0 if regime is NaN
    
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
        if row['Portfolio_Exposure'] == 1.0 and row['Adjusted_Market_Regime'] == 'Bearish Low Variance':
            if row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bearish':
                data.at[index, 'Portfolio_Exposure'] = 0.0
            elif row['30_Day_Indicator'] == 'Bullish' and row['60_Day_Indicator'] == 'Bearish':
                data.at[index, 'Portfolio_Exposure'] = 1.0
            elif row['30_Day_Indicator'] == 'Bearish' and row['60_Day_Indicator'] == 'Bullish':
                data.at[index, 'Portfolio_Exposure'] = 1.0
                
    return data

def calculate_returns_and_portfolio_values(data):
    initial_value = 100000
    
    # Initialize 'Beginning_Portfolio_Value' if it does not exist
    if 'Beginning_Portfolio_Value' not in data.columns:
        data['Beginning_Portfolio_Value'] = initial_value
    
    # Calculate strategy returns and adjust for leverage cost and transaction costs
    data['Leveraged_Portion'] = data['Portfolio_Exposure'] - 1
    data['Leveraged_Portion'] = data['Leveraged_Portion'].apply(lambda x: max(x, 0))  # Only positive leverage
    
    # Adjust leverage cost calculation based on the current portfolio value
    data['Leverage_Cost_Amount'] = data['Beginning_Portfolio_Value'] * data['Leveraged_Portion'] * data['Daily_Leverage_Rate']
    
    # Transaction and Slippage costs calculation
    transaction_cost_per_trade = 0.002  # Example: 0.1% per trade + 0.1% per trade on slippage
    data['Transaction_Slippage_Costs'] = transaction_cost_per_trade * np.abs(data['Portfolio_Exposure'].diff().fillna(0))
    
    # Calculate transaction cost in dollar amounts
    data['Transaction_Cost_Dollars'] = data['Transaction_Slippage_Costs'] * data['Beginning_Portfolio_Value']
    
    shorting_cost = 0.003  # Example: 0.3% for shorting
    data['Shorting_Costs'] = shorting_cost * (data['Portfolio_Exposure'] < 0).astype(int)
    
    # Update the strategy return calculation to use leverage cost directly
    data['Strategy_Return'] = (
        data['Index_Returns'] * data['Portfolio_Exposure']
        - data['Leverage_Cost_Amount'] / data['Beginning_Portfolio_Value']  # Use current portfolio value instead of initial
        - data['Transaction_Slippage_Costs']
        - data['Shorting_Costs']
    )
    
    # Set the strategy return for the first date to 0
    data.at[data.index[0], 'Strategy_Return'] = 0
    
    # Calculate cumulative returns starting with $100,000
    data['Portfolio_Value'] = initial_value * (1 + data['Strategy_Return']).cumprod()
    data['Market_Value'] = initial_value * (1 + data['Index_Returns']).cumprod()
    
    # Calculate the beginning portfolio value for each day
    data['Beginning_Portfolio_Value'] = data['Portfolio_Value'].shift(1).fillna(initial_value)
    
    # Recalculate transaction cost in dollar amounts after 'Beginning_Portfolio_Value' is updated
    data['Transaction_Cost_Dollars'] = data['Transaction_Slippage_Costs'] * data['Beginning_Portfolio_Value']
    
    # Calculate the daily return based on the beginning portfolio value
    data['Daily_Return'] = data['Beginning_Portfolio_Value'] * data['Strategy_Return']
    
    # Identify buy/sell signals based on changes in exposure
    data['Trade_Signal'] = ''
    data['Trade_Signal'] = np.where(data['Portfolio_Exposure'].diff() > 0, 'Buy', data['Trade_Signal'])
    data['Trade_Signal'] = np.where(data['Portfolio_Exposure'].diff() < 0, 'Sell', data['Trade_Signal'])
    data.at[data.index[0], 'Trade_Signal'] = 'Buy'
    
    # Adjust trade signals for next day's open price
    data['Next_Open'] = data['Open'].shift(0)
    data['Trade_Signal_Next_Open'] = data['Trade_Signal'].shift(0)
    
    # Add new columns to find the Beginning Portfolio Value and Date of the last "Buy" signal for each "Sell"
    data['Last_Buy_Value'] = None
    data['Last_Buy_Date'] = None
    last_buy_value = None
    last_buy_date = None

    for i in range(len(data)):
        if data['Trade_Signal_Next_Open'].iloc[i] == 'Buy':
            last_buy_value = data['Next_Open'].iloc[i]
            last_buy_date = data.index[i].date()  # Keep only the date element
        elif data['Trade_Signal_Next_Open'].iloc[i] == 'Sell' and last_buy_value is not None:
            data.at[data.index[i], 'Last_Buy_Value'] = last_buy_value
            data.at[data.index[i], 'Last_Buy_Date'] = last_buy_date  # Keep only the date element

    # Convert 'Last_Buy_Date' to string format to ensure only date is stored, not time.
    data['Last_Buy_Date'] = pd.to_datetime(data['Last_Buy_Date']).dt.date
    
    # Add a new column for Profit/Loss
    def calculate_profit_loss(row):
        if (
            row['Trade_Signal_Next_Open'] == 'Sell'
            and pd.notnull(row['Last_Buy_Value'])
            and pd.notnull(row['Next_Open'])
        ):
            return row['Next_Open'] - row['Last_Buy_Value']
        return 0
    
    data['Profit/Loss'] = data.apply(calculate_profit_loss, axis=1)
    
    def calculate_tax(row):
        if row['Trade_Signal_Next_Open'] == 'Sell' and pd.notnull(row['Last_Buy_Date']):
            # Convert Last_Buy_Date to Timestamp to match row.name type
            last_buy_date = pd.Timestamp(row['Last_Buy_Date'])
            days_held = (row.name - last_buy_date).days
            profit_loss = row['Profit/Loss']
            if days_held > 365:
                return 0.00 * profit_loss
            else:
                return 0.00 * profit_loss
        return 0
    
    data['Tax'] = data.apply(calculate_tax, axis=1)
    
    # Calculate Tax Amount in dollars based on the Profit/Loss when selling
    data['Tax_Amount'] = data['Tax']
    
    # Adjust the ending portfolio value for tax when selling
    data['Ending_Portfolio_Value'] = np.where(
        data['Trade_Signal_Next_Open'] == 'Sell',
        data['Beginning_Portfolio_Value'] + data['Daily_Return'] - data['Tax_Amount'],
        data['Beginning_Portfolio_Value'] + data['Daily_Return']
    )
    
    # Handle NaN in 'Ending_Portfolio_Value' by filling it with the previous value or the initial value
    data['Ending_Portfolio_Value'] = data['Ending_Portfolio_Value'].ffill().fillna(initial_value)
    
    # Calculate the beginning portfolio value for the next day including tax
    data['Beginning_Portfolio_Value'] = data['Ending_Portfolio_Value'].shift(1).fillna(initial_value)
    
    # Calculate drawdowns for the market (index) and the strategy
    data['Index_Drawdown'] = data['Market_Value'] / data['Market_Value'].cummax() - 1
    data['Strategy_Drawdown'] = data['Ending_Portfolio_Value'] / data['Ending_Portfolio_Value'].cummax() - 1
    
    return data

# Function to add Short Term Triangular Moving Averages and Indicators
def add_triangular_moving_averages_and_indicators(data):
    # Calculate 30-day and 60-day Triangular Moving Averages and shift by 1 day
    data['30_TMA'] = triangular_moving_average(data['Adj Close'], 30).shift(1)
    data['60_TMA'] = triangular_moving_average(data['Adj Close'], 60).shift(1)
    
    # Define 30-Day and 60-Day Indicators
    data['30_Day_Indicator'] = np.where(data['Adj Close'] > data['30_TMA'], 'Bullish', 'Bearish')
    data['60_Day_Indicator'] = np.where(data['Adj Close'] > data['60_TMA'], 'Bullish', 'Bearish')
    
    return data

# Function to calculate a triangular moving average
def triangular_moving_average(series, n):
    # Calculate the triangular moving average with a two-step rolling mean
    smoothed_series = series.rolling(window=(n // 2), min_periods=1).mean()
    smoothed_series = smoothed_series.rolling(window=(n // 2), min_periods=1).mean()
    return smoothed_series
    
def output_to_database(data):
    # Define the desired column order
    column_order = [
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Syn_Open', 'Adjusted_Open',
        'FEDFUNDS', 'DGS10', 'TB3MS', 'EFFR', 'Effective_Fed_Rate', 'IBKR_Rate', 'Daily_Leverage_Rate',
        'Vol_Regime', '250_TMA', 'Market_Regime', 'Adjusted_Market_Regime', '30_TMA', '60_TMA',
        '30_Day_Indicator', '60_Day_Indicator', 'Leveraged_Portion', 'Beginning_Portfolio_Value',
        'Index_Returns', 'Portfolio_Exposure', 'Leverage_Cost_Amount', 'Transaction_Slippage_Costs', 'Transaction_Cost_Dollars',
        'Shorting_Costs', 'Strategy_Return', 'Daily_Return', 'Trade_Signal', 'Next_Open', 'Trade_Signal_Next_Open',
        'Last_Buy_Value', 'Last_Buy_Date', 'Profit/Loss', 'Tax', 'Tax_Amount', 'Ending_Portfolio_Value',
        'Index_Drawdown', 'Strategy_Drawdown', 'Portfolio_Value', 'Market_Value'
    ]
    
    # Reorder the columns in the data
    data = data[column_order]
    
    # Reset index to ensure it's a column and not an index
    data_reset = data.reset_index()

    # Convert all columns of datetime64 dtype to string format
    for col in data_reset.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
        data_reset[col] = data_reset[col].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Convert all 'object' dtype columns containing datetime-like objects to strings
    for col in data_reset.select_dtypes(include=['object']).columns:
        if isinstance(data_reset[col].iloc[0], (pd.Timestamp, datetime)):
            data_reset[col] = data_reset[col].astype(str)

    # Iterate through all columns and convert datetime-like objects to strings
    for col in data_reset.columns:
        data_reset[col] = data_reset[col].apply(lambda x: str(x) if isinstance(x, (pd.Timestamp, datetime)) else x)

    # Output to SQLite database
    with sqlite3.connect('output/financial_model.db') as conn:
        try:
            data_reset.to_sql('financial_data', conn, if_exists='replace', index=False)
        except Exception as e:
            print(f"An error occurred while writing to the database: {e}")
            print("Data types of DataFrame columns:")
            print(data_reset.dtypes)
            print("First few rows of DataFrame:")
            print(data_reset.head())

def telegram_messenger():
    # Telegram Bot API token and Channel ID
    bot_token = '7328648943:AAH3gHyGf2xgjxBfzPd05F_7IagASgs-Dj0'
    channel_id = '-1002309744206'

    # Define the path to your SQLite database
    database_path = r"C:\Users\NicholasRatti\OneDrive - Fernandina Capital, LLC\Fernandina Capital\Projects\Active\Python\Markov_Regime_Switching_Model\daily\output\financial_model.db"

    # Connect to the database
    conn = sqlite3.connect(database_path)

    # Query the last two rows from 'Adjusted_Market_Regime' and 'Date' columns
    query = """
    SELECT Date, Adjusted_Market_Regime, Portfolio_Exposure 
    FROM financial_data 
    ORDER BY rowid DESC 
    LIMIT 2
    """  # Make sure 'financial_data' is the correct table name

    # Execute query and load into a DataFrame
    data = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Initialize the message variable each time the code runs with bold header
    message = "<b>Your Daily Portfolio Exposure Update</b>\n\n"  # Reset message here
    labels = ["Tomorrow's Market Regime", "Today's Market Regime"]

    # Loop through the DataFrame and format the message
    for index, row in data.iloc[::-1].iterrows():  # Reverse order for Previous Day first
        # Format Date
        from datetime import datetime, timedelta

        # Parse the date and add 1 day
        formatted_date = (datetime.strptime(row['Date'], '%Y-%m-%d %H:%M:%S') + timedelta(days=1)).strftime('%m/%d/%Y')

        formatted_date = formatted_date.lstrip("0").replace("/0", "/")  # Remove leading zeros from month and day

        # Add the labeled message for each row with line breaks for better formatting
        message += f"<u>{labels[index]}</u>\n"
        message += f"<i>Date</i>: {formatted_date}\n"
        message += f"<i>Adjusted Market Regime</i>: {row['Adjusted_Market_Regime']}\n"
        message += f"<i>Portfolio Exposure</i>: {row['Portfolio_Exposure'] * 100:.0f}%\n\n" # Format Portfolio_Exposure as a percentage with 2 decimal places


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
            
if __name__ == "__main__":
    main()
