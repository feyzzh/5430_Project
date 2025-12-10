from flask import Flask
from flask import request, jsonify
import time
from stockview import StockView

app = Flask(__name__)

@app.route('/api/time')
def get_time():
    return {'time': time.time()}

@app.route('/api/hmm', methods=['GET'])
def hmm_endpoint():
    symbol = request.args.get("symbol", "CRWD")
    period = request.args.get("period", "10y")
    interval = request.args.get("interval", "1d")
    K = int(request.args.get("K", 3))

    # validate arguments inputted by user
    PERIOD = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h', '1d', '5d', '1wk', '1mo', '3mo']
    errors = []
    if period not in PERIOD:
        errors.append(f'Period [{period}] must be one of the options: {PERIOD}')
    if interval not in INTERVALS:
        errors.append(f'Interval [{interval}] must be one of the options: {INTERVALS}')
    if K != 2 and K != 3:
        errors.append('Please try 2 or 3 states')
    if errors:
        return jsonify({'ok': False, 'errors': errors}), 400

    print(f'Received request! {request}')

    sv = StockView(symbol, pd=period, int=interval)
    sv.set_K(K)
    result = sv.fit_hmm_regimes()

    return jsonify(result)