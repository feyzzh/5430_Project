import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go


# Source: https://www.marketcalls.in/python/introduction-to-hidden-markov-models-hmm-for-traders-python-tutorial.html
def fetch_intraday_data(symbol, interval='1D', days=3650):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    return data

def calculate_roc(data, window=12):
    return data['Close'].pct_change(periods=window)

def fit_hmm_predict(data, n_components=3):
    model = hmm.GaussianHMM(n_components=n_components, n_iter=100, random_state=0)
    model.fit(np.array(data).reshape(-1, 1))
    hidden_states = model.predict(np.array(data).reshape(-1, 1))
    return pd.Series(hidden_states, index=data.index)

def generate_signals(hidden_states):
    signals = pd.Series(index=hidden_states.index, data=0)
    # 1 is buy signal for bullish state
    # -1 is sell signal for bearish state
    signals[hidden_states == 2] = 1
    signals[hidden_states == 0] = -1
    return signals

# Main function
def main():
    # Fetch data
    # symbol = "^NSEI"  # Nifty 50 index
    # data = fetch_intraday_data(symbol)
    symbol = 'MDB'
    data = pd.read_csv('dataset/mdb_prices.csv')
    
    # Calculate ROC
    roc = calculate_roc(data)
    
    # Fit HMM and predict regimes
    hidden_states = fit_hmm_predict(roc.dropna())
    
    # Generate signals
    signals = generate_signals(hidden_states)
    
    # Create DataFrame with all necessary data
    result = pd.DataFrame({
        'Close': data['Close'],
        'ROC': roc,
        'Regime': hidden_states,
        'Signal': signals
    }).dropna()
    
    # Visualize using Plotly
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(x=result.index, y=result['Close'], mode='lines', name=f'{symbol} Close'))
    
    # Add buy signals
    buy_points = result[result['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points['Close'], mode='markers', 
                             marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy Signal'))
    
    # Add sell signals
    sell_points = result[result['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points['Close'], mode='markers', 
                             marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell Signal'))
    
    # Update layout with the new specifications
    fig.update_layout(
        title=f'{symbol} HMM Strategy',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        xaxis=dict(type='category')
    )
    
    # Show the plot
    fig.show()

    # Print some statistics
    print(f"Total data points: {len(result)}")
    print(f"Buy signals: {len(buy_points)}")
    print(f"Sell signals: {len(sell_points)}")

if __name__ == "__main__":
    main()