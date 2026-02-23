from flask import Flask, jsonify, request
import requests
from datetime import datetime

app = Flask(__name__)

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com",
    "Origin": "https://finance.yahoo.com",
})

YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


@app.route("/history/<ticker>")
def get_history(ticker):
    """
    OHLCV historical data.
    Query params:
      range    : 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max  (default: 1mo)
      interval : 1m 5m 15m 30m 1h 1d 1wk 1mo              (default: 1d)
    """
    try:
        range_   = request.args.get("range",    "1mo")
        interval = request.args.get("interval", "1d")

        url = f"{YAHOO_BASE}/{ticker.upper()}"
        params = {"interval": interval, "range": range_, "includePrePost": False}
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()

        data   = r.json()
        result = data.get("chart", {}).get("result")
        if not result:
            error = data.get("chart", {}).get("error", {})
            return jsonify({"error": error.get("description", "No data found")}), 404

        result     = result[0]
        timestamps = result.get("timestamp", [])
        quote      = result["indicators"]["quote"][0]

        opens   = quote.get("open",   [])
        highs   = quote.get("high",   [])
        lows    = quote.get("low",    [])
        closes  = quote.get("close",  [])
        volumes = quote.get("volume", [])

        records = []
        for i, ts in enumerate(timestamps):
            if closes[i] is None:
                continue
            records.append({
                "date":   datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d"),
                "open":   round(opens[i],  2) if opens[i]  else None,
                "high":   round(highs[i],  2) if highs[i]  else None,
                "low":    round(lows[i],   2) if lows[i]   else None,
                "close":  round(closes[i], 2) if closes[i] else None,
                "volume": volumes[i],
            })

        return jsonify({
            "ticker":   ticker.upper(),
            "range":    range_,
            "interval": interval,
            "count":    len(records),
            "data":     records,
        })

    except requests.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
