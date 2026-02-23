# yFinance Stock Data API

A Flask REST API that pulls real-time and historical stock data using yfinance, deployable on Railway.com.

## API Endpoints

| Endpoint | Description | Example |
|---|---|---|
| `GET /` | API info | `/` |
| `GET /quote/<ticker>` | Current stock quote | `/quote/AAPL` |
| `GET /history/<ticker>` | Historical OHLCV data | `/history/AAPL?period=3mo&interval=1d` |
| `GET /info/<ticker>` | Company info & fundamentals | `/info/MSFT` |
| `GET /financials/<ticker>` | Income statement | `/financials/TSLA` |

### History Parameters
- `period`: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
- `interval`: `1m`, `5m`, `15m`, `30m`, `1h`, `1d`, `1wk`, `1mo`

## Local Development

```bash
pip install -r requirements.txt
python app.py
# Visit http://localhost:5000
```

## Deploy to Railway.com

### Option 1: GitHub (Recommended)
1. Push this folder to a GitHub repository
2. Go to [railway.com](https://railway.com) → **New Project** → **Deploy from GitHub repo**
3. Select your repository
4. Railway auto-detects Python and deploys — no configuration needed
5. Click **Generate Domain** to get a public URL

### Option 2: Railway CLI
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize and deploy
railway init
railway up
```

### Option 3: One-click via railway.toml
The included `railway.toml` handles all build and deploy settings automatically.

## Environment Variables
No environment variables are required. Railway automatically injects `$PORT`.

## Example Responses

### `/quote/AAPL`
```json
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "price": 189.30,
  "market_cap": 2900000000000,
  "pe_ratio": 29.5,
  "currency": "USD"
}
```

### `/history/AAPL?period=5d&interval=1d`
```json
{
  "ticker": "AAPL",
  "period": "5d",
  "interval": "1d",
  "count": 5,
  "data": [
    {"date": "2024-01-15", "open": 185.0, "high": 190.0, "low": 184.5, "close": 189.3, "volume": 52000000}
  ]
}
```
