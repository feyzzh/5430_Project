import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

class StockView:
    def __init__(self, symbol, pd='10y', int='1d', out_dir='more_data/'):
        self.symbol = symbol
        self.period = pd
        self.interval = int

        self.data_fp = os.path.join(out_dir, f'{symbol.lower()}_{pd}_{int}.csv')
        self.raw_data = self.generate_data()
        self.data = self.process_data()

    def generate_data(self):
        ticker = yf.Ticker(self.symbol)
        data = ticker.history(period=self.period, interval=self.interval)
        data.to_csv(self.data_fp)
        return data

    def process_data(self):
        data_df = self.raw_data.drop(['Dividends', 'Stock Splits'], axis=1)
        data_df['Log Returns'] = np.log(data_df['Close'] / data_df['Close'].shift(1))
        data_df['Intraday Volatility'] = 0.5 * (np.log(data_df['High'] / data_df['Low']) ** 2) - 0.386 * (np.log(data_df['Close'] / data_df['Open']) ** 2)
        data_df['Range'] = data_df['High'] - data_df['Low']

        # convert date column from string type to date type
        if 'Datetime' in data_df.columns:
            data_df.rename(columns={'Datetime': 'Date'}, inplace=True)
        elif 'Date' in data_df.columns:
            data_df = data_df['Date'].str.split()[0]
        data_df['Date'] = pd.to_datetime(data_df['Date'])

        data_df = data_df[['Date', 'Log Returns', 'Intraday Volatility', 'Range']]
        data_df.set_index('Date', inplace=True)
        return data_df


    def get_moving_ave(span=10):
        pass
