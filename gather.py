import yfinance as yf

# https://ericpien.github.io/yfinance/reference/api/methods/yfinance.Tickers.history.html
# Tickers.history(period='1mo', interval='1d', start=None, end=None, prepost=False, actions=True, auto_adjust=True, repair=False, proxy=None, threads=True, group_by='column', progress=True, timeout=10, **kwargs)
# Period must be one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
# Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 4h, 1d, 5d, 1wk, 1mo, 3mo]

def gather_data(symbol, pd='5y'):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=pd)
    data.to_csv(f'{symbol.lower()}_prices.csv')


if __name__ == "__main__":
    gather_data('MDB')
    gather_data('AAPL')
    gather_data('TSLA')
    gather_data('META')
    gather_data('NVDA')
    gather_data('CRWD')
    gather_data('SPY')