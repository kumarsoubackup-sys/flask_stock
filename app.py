from flask import Flask, jsonify, request
import yfinance as yf
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "yFinance Stock API",
        "endpoints": {
            "GET /quote/<ticker>": "Get current stock quote",
            "GET /history/<ticker>": "Get historical price data (params: period, interval)",
            "GET /info/<ticker>": "Get company info",
            "GET /financials/<ticker>": "Get financial statements"
        }
    })

@app.route("/quote/<ticker>")
def get_quote(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        return jsonify({
            "ticker": ticker.upper(),
            "name": info.get("longName", "N/A"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "open": info.get("open"),
            "high": info.get("dayHigh"),
            "low": info.get("dayLow"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange"),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/history/<ticker>")
def get_history(ticker):
    """
    Query params:
      period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max (default: 1mo)
      interval: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo (default: 1d)
    """
    try:
        period = request.args.get("period", "1mo")
        interval = request.args.get("interval", "1d")
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            return jsonify({"error": "No data found for ticker"}), 404

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": str(date),
                "open": round(row["Open"], 4),
                "high": round(row["High"], 4),
                "low": round(row["Low"], 4),
                "close": round(row["Close"], 4),
                "volume": int(row["Volume"]),
            })

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "count": len(records),
            "data": records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/info/<ticker>")
def get_info(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        # Return a clean subset of company info
        fields = [
            "longName", "sector", "industry", "country", "website",
            "longBusinessSummary", "fullTimeEmployees", "dividendYield",
            "beta", "forwardPE", "priceToBook", "returnOnEquity",
            "debtToEquity", "revenueGrowth", "grossMargins", "operatingMargins"
        ]
        return jsonify({k: info.get(k) for k in fields})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/financials/<ticker>")
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker.upper())
        income_stmt = stock.income_stmt
        if income_stmt is not None and not income_stmt.empty:
            data = income_stmt.fillna(0).to_dict()
            # Convert Timestamp keys to strings
            data = {str(k): {str(row): val for row, val in v.items()} for k, v in data.items()}
        else:
            data = {}
        return jsonify({"ticker": ticker.upper(), "income_statement": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
