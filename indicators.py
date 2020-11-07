import pandas as pd
import numpy as np
import util
import matplotlib.pyplot as plt

# Receives Adj Close df. Returns SMA df
def calculate_SMA(df, period=20):
    return df.rolling(period).mean()

# Receive Adj CLose df. Return EMA df
def calculate_EMA(df, period=20):
    return df.ewm(span=period, min_periods=period).mean()

# Receive Adj Close df. Return Bollinger value, NOT the band
def calculate_Bollinger(df, period=20, dev=2):
    return df.rolling(period).std() * dev

# Receive Adj Close df. Return ROC value df
def calculate_ROC(df, period=20):
    N = df.diff(periods=period-1)
    D = df.shift(periods=period-1)
    return N/D

# Receive Adj Close. Calculate K and return D Stochastic df. NOTE: typical lookback = 14 and smooth = 3
def calculate_Stochastic(df, lookback=14, smooth=3):
    K = (df - df.rolling(lookback).min()) / (df.rolling(lookback).max() - df.rolling(lookback).min())
    D = K.rolling(smooth).mean()
    return D

# Receive Adj Close. Calculate the MACD which is 12 days EMA - 26 days EMA
def calculate_MACD(df):
    EMA_12 = calculate_EMA(df, period=12)
    EMA_26 = calculate_EMA(df, period=26)
    MACD = EMA_12 - EMA_26
    return MACD

# Calculate 5 (6) indicators and append them to the dataframe: 
# SMA , EMA as an extra, Bollinger band, ROC, Stochastics Oscillator, and On Balance Volume, and MACD
def calculate_indicators(data):
    #1. simple moving average. Period = 20 days
    data['SMA'] = calculate_SMA(data['Adj Close'], period=20)
    # Extra: Do the EMA as well
    data['EMA'] = calculate_EMA(data['Adj Close'], period=20)
    #2. Bollinger value (2 std away from the SMA. std calculated rolling based on same number of days)
    data['Bollinger'] = calculate_Bollinger(data['Adj Close'])
    #3. Momentum (Rate of change)
    data['ROC'] = calculate_ROC(data['Adj Close'], period=23)
    #4. Stochastic Oscillator. Lookback 20 days
    data['Stochastic'] = calculate_Stochastic(data['Adj Close'], lookback=14)
    # 5. MACD
    data['MACD'] = calculate_MACD(data['Adj Close'])
    return data

def report(symbol='JPM', start_date='2008-01-01', end_date='2009-12-31'):
    # Prepare symbols and dates to retrieve data
    symbols = [symbol]
    dates = pd.date_range(start_date, end_date)
    # Repeated calls to util.get_data to get multiple columns of data for symbol. Then join to have a complete dataframe.
    data_close = util.get_data(symbols, dates, addSPY=False, colname='Adj Close')
    data_close.rename(columns={data_close.columns[0]:'Adj Close'}, inplace=True)
    data_volume = util.get_data(symbols, dates, addSPY=False, colname='Volume')
    data_volume.rename(columns={data_volume.columns[0]:'Volume'}, inplace=True)
    data = data_close.join(data_volume)
    data.dropna(axis=0, how='all', inplace=True)
    # Pass to calculate_indicators
    data_final = calculate_indicators(data)
    # It's ok to fill backwards on the 2 prices indicators now for the initial NaN values, then normalize those 2 prices
    data_final['SMA'].fillna(method='bfill', inplace=True)
    data_final['EMA'].fillna(method='bfill', inplace=True)
    data_normed = data_final.copy()
    data_normed['SMA'] = data_normed['SMA'] / data_normed['SMA'].ix[0,:]
    data_normed['EMA'] = data_normed['EMA'] / data_normed['EMA'].ix[0,:]
    data_normed['Adj Close'] = data_normed['Adj Close'] / data_normed['Adj Close'].ix[0,:]
    #print(data_normed.head(60))
    #1. Price / EMA Indicator
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_title('Normalized Price and EMA, with Price/EMA ratio chart')
    ax1.plot(data_normed['Adj Close'], label='Price')
    ax1.plot(data_normed['EMA'], label='EMA')
    ax2.plot(data_final['Adj Close']/data_final['EMA'], label='Price/EMA')
    ax2.axhline(y=1, color='r', linestyle='--')
    ax1.legend()
    ax2.legend()
    plt.savefig('Price_EMA.png')

    # MACD and MACD Signal Line (which is 9-day EMA of MACD itself)
    fig, (ax11, ax12) = plt.subplots(2)
    ax11.set_title('MACD, with MACD signal line crossover')
    ax11.set_ylabel('Price')
    EMA_12 = calculate_EMA(data_final['Adj Close'], period=12)
    EMA_26 = calculate_EMA(data_final['Adj Close'], period=26)
    ax11.plot(EMA_12, label='12 days EMA')
    ax11.plot(EMA_26, label='26 days EMA')
    ax12.plot(data_final['MACD'], label='MACD')
    MACD_signal = calculate_EMA(data_final['MACD'], period=9)
    ax12.plot(MACD_signal, label='Signal line')
    ax12.axhline(y=0, linestyle='--')
    ax11.legend()
    ax12.legend()
    plt.savefig('MACD')

    #2. Price and Bollinger Band: %B = (Price - Lower Band)/(Upper Band - Lower Band)
    fig, (ax3, ax4) = plt.subplots(2)
    ax3.set_title('Price and Bollinger bands, with B% value chart')
    ax3.plot(data_final['Adj Close'], label='Price')
    ax3.plot(data_final['SMA'] + data_final['Bollinger'], label='Upper Bollinger band')
    ax3.plot(data_final['SMA'] - data_final['Bollinger'], label='Lower Bollinger band')
    B_percent = (data_final['Adj Close']-(data_final['SMA']-data_final['Bollinger']))\
        /(2*data_final['Bollinger'])
    ax4.plot(B_percent,label='B% value')
    ax4.axhline(y=0, linestyle='--')
    ax4.axhline(y=1, linestyle='--')
    ax3.legend()
    ax4.legend()
    plt.savefig('Bollinger.png')
    #3. Price and Momentum (ROC)
    fig, (ax5, ax6) = plt.subplots(2)
    ax5.set_title('Normalized price and momentum (ROC)')
    ax5.plot(data_normed['Adj Close'], label='Price')
    ax6.plot(data_normed['ROC'], label='Momentum')
    ax6.axhline(y=0, linestyle='--')
    ax5.axhline(y=1, linestyle='--')
    ax5.legend()
    ax6.legend()
    plt.savefig('ROC.png')
    #4. Price and Stochastic Oscillator
    fig, ax7 = plt.subplots()
    ax7.set_title('Normalized price and Stochastic Oscillator')
    ax7.set_ylabel('Price')
    ax7.plot(data_normed['Adj Close'], label='Price', color='b')
    ax8 =ax7.twinx()
    ax8.set_ylabel('Stochastic Oscillator')
    ax8.plot(data_normed['Stochastic'], label='Stochastic oscillator', color='g')
    #ax8.axhline(y=0, linestyle='--')
    #ax8.axhline(y=1, linestyle='--')
    ax7.legend()
    ax8.legend()
    plt.savefig('Stochastic.png')

def author():
    return 'tbui61'

if __name__ == "__main__":
    report('JPM', '2008-01-01', '2009-12-31')