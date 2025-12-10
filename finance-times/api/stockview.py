import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
from fit import FitHMM
import fit
from datetime import datetime
import json

class StockView:
    def __init__(self, symbol, pd='10y', int='1d', out_dir='app_data/', fp=None):
        self.symbol = symbol
        self.period = pd
        self.interval = int

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if fp:
            self.data_fp = fp
            self.raw_data = pd.read_csv(fp)
        else:
            formatted_date = datetime.now().strftime('%m%d%y')
            self.data_fp = os.path.join(out_dir, f'{symbol.lower()}_{pd}_{int}_{formatted_date}.csv')
            self.raw_data = self.generate_data()
        self.data = self.process_data()

        self.set_K(2)

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

        # Convert index into a Date column
        if isinstance(data_df.index, pd.DatetimeIndex):
            data_df = data_df.reset_index().rename(columns={'index':'Date'})
        elif 'Date' not in data_df.columns and 'Datetime' in data_df.columns:
            data_df.rename(columns={'Datetime':'Date'}, inplace=True)
        elif 'Date' not in data_df.columns:
            raise Exception("No Date column or datetime index found")

        # Now ensure Date column is datetime
        data_df['Date'] = pd.to_datetime(data_df['Date'])

        # data_df = data_df[['Date', 'Log Returns', 'Intraday Volatility', 'Range']]
        # data_df.set_index('Date', inplace=True)
        return data_df


    def get_moving_ave(self, span=10):
        self.data[f'EMA_{span}'] = self.data['Close'].ewm(span=span, adjust=False).mean()
        pass

    def fit_hmm_regimes(self):
        print(self.data.columns)
        data_df = self.data[['Date', 'Log Returns', 'Intraday Volatility', 'Range']]
        data_df.set_index('Date', inplace=True)
        hmm = FitHMM(data_df, full_df=self.data, K=self.K)
        result, df_with_regimes = hmm.pipeline()
        result['ticker'] = self.symbol
        print(df_with_regimes)

        regimes_reset = df_with_regimes.reset_index()
        merged = self.data.merge(regimes_reset, on=['Date', 'Log Returns', 'Intraday Volatility', 'Range'], how='left')
        print(merged)
        print(merged.columns)

        # 4) Build time_series for the frontend (candles + regimes)
        time_series = []
        for _, row in merged.iterrows():
            date = row['Date']
            regime_val = row.get('Regime')
            # convert NaN → None for JSON safety
            if isinstance(regime_val, float) and pd.isna(regime_val):
                regime_val = None

            time_series.append({
                "date": date.isoformat(),
                "open": float(row['Open']) if not pd.isna(row['Open']) else None,
                "high": float(row['High']) if not pd.isna(row['High']) else None,
                "low": float(row['Low']) if not pd.isna(row['Low']) else None,
                "close": float(row['Close']) if not pd.isna(row['Close']) else None,
                "volume": float(row['Volume']) if not pd.isna(row['Volume']) else None,
                "logReturn": float(row['Log Returns']) if not pd.isna(row['Log Returns']) else None,
                "regime": regime_val,
                "regimeIndex": int(row['RegimeIndex']) if not pd.isna(row.get('RegimeIndex')) else None,
            })

        result["time_series"] = time_series
        with open('output_viewer.json', 'w') as f:
            json.dump(result, f, indent=4)
        return result

    def get_data(self):
        return self.data
    
    def set_K(self, K):
        self.K = K