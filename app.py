from flask import Flask, jsonify, request
import requests
from datetime import datetime

app = Flask(__name__)

# ── shared session with browser-like headers ──────────────────────────────────
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

YAHOO_BASE  = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_QUOTE = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"


def fetch_chart(ticker: str, interval: str = "1d", range_: str = "1mo") -> dict:
    """Fetch OHLCV chart data from Yahoo Finance."""
    url = f"{YAHOO_BASE}/{ticker}"
    params = {"interval": interval, "range": range_, "includePrePost": False}
    r = session.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        error = data.get("chart", {}).get("error", {})
        raise ValueError(error.get("description", "No data returned from Yahoo Finance"))
    return result[0]


def fetch_summary(ticker: str, modules: str) -> dict:
    """Fetch company summary modules from Yahoo Finance."""
    url = f"{YAHOO_QUOTE}/{ticker}"
    params = {"modules": modules, "formatted": False}
    r = session.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteSummary", {}).get("result")
    if not result:
        error = data.get("quoteSummary", {}).get("error", {})
        raise ValueError(error.get("description", "No summary data returned"))
    return result[0]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "message": "📈 Yahoo Finance Stock API",
        "note": "For Indian stocks append .NS (NSE) or .BO (BSE)",
        "examples": {
            "ITC quote":      "/quote/ITC.NS",
            "ITC history":    "/history/ITC.NS?range=3mo&interval=1d",
            "ITC info":       "/info/ITC.NS",
            "ITC financials": "/financials/ITC.NS",
            "Compare":        "/compare?tickers=ITC.NS,RELIANCE.NS,HDFCBANK.NS",
        },
        "intervals": ["1m","5m","15m","30m","1h","1d","1wk","1mo"],
        "ranges":    ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"],
    })


@app.route("/quote/<ticker>")
def get_quote(ticker):
    """Current price snapshot."""
    try:
        result = fetch_chart(ticker.upper(), interval="1d", range_="1d")
        meta   = result.get("meta", {})
        curr   = meta.get("regularMarketPrice", 0)
        prev   = meta.get("chartPreviousClose", 0)

        return jsonify({
            "ticker":           meta.get("symbol"),
            "name":             meta.get("longName") or meta.get("shortName"),
            "currency":         meta.get("currency"),
            "exchange":         meta.get("exchangeName"),
            "current_price":    curr,
            "previous_close":   prev,
            "price_change":     round(curr - prev, 2),
            "price_change_pct": round((curr - prev) / prev * 100, 2) if prev else None,
            "52w_high":         meta.get("fiftyTwoWeekHigh"),
            "52w_low":          meta.get("fiftyTwoWeekLow"),
            "market_timezone":  meta.get("timezone"),
            "timestamp":        datetime.utcnow().isoformat() + "Z",
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except requests.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        result     = fetch_chart(ticker.upper(), interval=interval, range_=range_)
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
                "date":   datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M"),
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

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except requests.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/info/<ticker>")
def get_info(ticker):
    """Company profile and key fundamentals."""
    try:
        data    = fetch_summary(ticker.upper(), "assetProfile,defaultKeyStatistics,summaryDetail")
        profile = data.get("assetProfile", {})
        stats   = data.get("defaultKeyStatistics", {})
        summary = data.get("summaryDetail", {})

        return jsonify({
            "ticker":         ticker.upper(),
            "sector":         profile.get("sector"),
            "industry":       profile.get("industry"),
            "country":        profile.get("country"),
            "website":        profile.get("website"),
            "employees":      profile.get("fullTimeEmployees"),
            "description":    profile.get("longBusinessSummary"),
            "market_cap":     summary.get("marketCap"),
            "trailing_pe":    summary.get("trailingPE"),
            "forward_pe":     summary.get("forwardPE"),
            "dividend_yield": summary.get("dividendYield"),
            "beta":           summary.get("beta"),
            "price_to_book":  stats.get("priceToBook"),
            "52w_high":       summary.get("fiftyTwoWeekHigh"),
            "52w_low":        summary.get("fiftyTwoWeekLow"),
            "avg_volume":     summary.get("averageVolume"),
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except requests.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/financials/<ticker>")
def get_financials(ticker):
    """Annual income statement key figures."""
    try:
        data  = fetch_summary(ticker.upper(), "incomeStatementHistory")
        stmts = data.get("incomeStatementHistory", {}).get("incomeStatementHistory", [])

        records = []
        for s in stmts:
            records.append({
                "end_date":           s.get("endDate", {}).get("fmt"),
                "total_revenue":      s.get("totalRevenue", {}).get("raw"),
                "gross_profit":       s.get("grossProfit", {}).get("raw"),
                "ebit":               s.get("ebit", {}).get("raw"),
                "net_income":         s.get("netIncome", {}).get("raw"),
                "operating_income":   s.get("operatingIncome", {}).get("raw"),
                "income_tax_expense": s.get("incomeTaxExpense", {}).get("raw"),
            })

        return jsonify({"ticker": ticker.upper(), "income_statement": records})

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except requests.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare")
def compare():
    """
    Compare multiple tickers side by side.
    Usage: /compare?tickers=ITC.NS,RELIANCE.NS,HDFCBANK.NS
    """
    tickers_param = request.args.get("tickers", "")
    if not tickers_param:
        return jsonify({"error": "Pass ?tickers=ITC.NS,RELIANCE.NS"}), 400

    tickers = [t.strip().upper() for t in tickers_param.split(",")]
    results = {}

    for ticker in tickers:
        try:
            result = fetch_chart(ticker, interval="1d", range_="1d")
            meta   = result.get("meta", {})
            prev   = meta.get("chartPreviousClose", 0)
            curr   = meta.get("regularMarketPrice", 0)
            results[ticker] = {
                "price":      curr,
                "change":     round(curr - prev, 2),
                "change_pct": round((curr - prev) / prev * 100, 2) if prev else None,
                "currency":   meta.get("currency"),
                "exchange":   meta.get("exchangeName"),
            }
        except Exception as e:
            results[ticker] = {"error": str(e)}

    return jsonify({"comparison": results, "timestamp": datetime.utcnow().isoformat() + "Z"})


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
