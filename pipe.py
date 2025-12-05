import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

def get_data(symbol, pd='10y', int='1d', out_dir='more_data/'):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=pd, interval=int)
    out_fp = os.path.join(out_dir, f'{symbol.lower()}_{pd}_{int}.csv')
    data.to_csv(out_fp)
    return out_fp

def process_data(data_fp):
    data_in = pd.read_csv(data_fp)
    data_df = data_in.drop(['Dividends', 'Stock Splits'], axis=1)
