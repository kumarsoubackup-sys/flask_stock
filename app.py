import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from datetime import datetime, time as dtime, timezone, timedelta
from flask import Flask, jsonify, request, send_file, render_template
import requests as req_lib

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ═══════════════════════════════════════════════════════════
# YAHOO FINANCE SESSION (no yfinance — direct API calls)
# ═══════════════════════════════════════════════════════════
yf_session = req_lib.Session()
yf_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://finance.yahoo.com",
    "Origin":          "https://finance.yahoo.com",
})

YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"


def fetch_ohlcv(ticker: str, interval: str = "1m", range_: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance and return as DataFrame (UTC-indexed)."""
    url    = f"{YAHOO_BASE}/{ticker.upper()}"
    params = {"interval": interval, "range": range_, "includePrePost": False}
    r      = yf_session.get(url, params=params, timeout=15)
    r.raise_for_status()
    data   = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error", {})
        raise ValueError(err.get("description", "No data from Yahoo Finance"))

    result     = result[0]
    timestamps = result.get("timestamp", [])
    quote      = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "open":   quote.get("open",   []),
        "high":   quote.get("high",   []),
        "low":    quote.get("low",    []),
        "close":  quote.get("close",  []),
        "volume": quote.get("volume", []),
    }, index=pd.to_datetime(timestamps, unit="s", utc=True))

    df.dropna(subset=["close"], inplace=True)
    return df


# ═══════════════════════════════════════════════════════════
# TIMEZONE
# ═══════════════════════════════════════════════════════════
IST_OFFSET = timedelta(hours=5, minutes=30)
IST        = timezone(IST_OFFSET)

# ═══════════════════════════════════════════════════════════
# INDICATOR INPUTS
# ═══════════════════════════════════════════════════════════
SESSION_START    = dtime(9, 15)
SESSION_END      = dtime(15, 30)
ORB_MINUTES      = 5
DONCHIAN_LEN     = 20
SMOOTH_LEN       = 1
GRAVITY_LEN      = 60
SPEED_LEN        = 5
CRASH_MULT       = 2.0
SPEED_MULT       = 1.5
WINDOW_FALL      = 100
THRESHOLD_FALL   = 3.0
WINDOW_AD        = 300
THRESHOLD_AD     = 3.0
USE_VOLUME       = True
VOL_MULTIPLIER   = 1.5
USE_RSI          = True
RSI_THRESHOLD    = 50
DETECT_BREAKOUTS = True
EXCESS_SMOOTH_BARS = 100

# ═══════════════════════════════════════════════════════════
# CHART CONSTANTS
# ═══════════════════════════════════════════════════════════
DARK_BG    = "#0d1117"
GRID_COL   = "#21262d"
TEXT_COL   = "#c9d1d9"
UP_COL     = "#26a641"
DN_COL     = "#f85149"
VWAP_COL   = "#58a6ff"
GRAV_COL   = "#d2a8ff"
FIB0_COL   = "#3fb950"
FIB50_COL  = "#388bfd"
FIB100_COL = "#f85149"
COMB_COL   = "#ffa657"
SMOOTH_COL = "#79c0ff"

# ═══════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════

def _to_ist(index):
    if index.tz is None:
        index = index.tz_localize("UTC")
    return index.tz_convert(IST)

def _in_session_mask(index):
    ist = _to_ist(index)
    return np.array(
        [(t.time() >= SESSION_START and t.time() <= SESSION_END) for t in ist],
        dtype=bool,
    )

def _rsi(close, period=14):
    n = len(close); rsi = np.full(n, np.nan); delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0); loss = np.where(delta < 0, -delta, 0.0)
    if n <= period: return rsi
    avg_gain = np.mean(gain[:period]); avg_loss = np.mean(loss[:period])
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
        rsi[i + 1] = 100 - (100 / (1 + rs))
    return rsi

def _sma(arr, period):
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        w = arr[i - period + 1: i + 1]; v = w[~np.isnan(w)]
        if len(v) == period: out[i] = v.mean()
    return out

def _rolling_std(arr, period):
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        w = arr[i - period + 1: i + 1]
        if not np.any(np.isnan(w)): out[i] = np.std(w, ddof=1)
    return out

def get_vol_str(vol):
    sign, av = ("-" if vol < 0 else ""), abs(vol)
    if av >= 1e9: return f"{sign}{av/1e9:.1f}B"
    if av >= 1e6: return f"{sign}{av/1e6:.1f}M"
    if av >= 1e3: return f"{sign}{av/1e3:.1f}K"
    return f"{sign}{av:.2f}"

# ═══════════════════════════════════════════════════════════
# EXCESS COMBINED INTRADAY VOLUME
# ═══════════════════════════════════════════════════════════

def calculate_excess_combined_intraday_volume(df: pd.DataFrame, smooth_bars: int = 100) -> pd.DataFrame:
    df = df.copy().sort_index()
    df['date']       = df.index.date
    df['is_new_day'] = df['date'] != df['date'].shift(1)
    df['is_up_bar']  = df['close'] > df['open']
    df['is_down_bar']= df['close'] < df['open']
    df['day_group']  = df['is_new_day'].cumsum()
    df['bars_in_day']= df.groupby('day_group').cumcount()
    df['buy_vol_increment'] = np.where(df['is_up_bar'],   df['volume'], 0)
    df['sel_vol_increment'] = np.where(df['is_down_bar'], df['volume'], 0)
    df['buy_volume'] = df.groupby('day_group')['buy_vol_increment'].cumsum()
    df['sel_volume'] = df.groupby('day_group')['sel_vol_increment'].cumsum()
    first_bar = df['is_new_day']
    df.loc[first_bar, 'buy_volume'] = df.loc[first_bar, 'volume']
    df.loc[first_bar, 'sel_volume'] = df.loc[first_bar, 'volume']
    df['combined_volume'] = df['buy_volume'] - df['sel_volume']

    def rolling_avg_variable_window(series, bars_in_day, max_window):
        result = pd.Series(np.nan, index=series.index)
        arr = series.values; bid = bars_in_day.values
        for i in range(len(arr)):
            window_size = int(min(bid[i] + 1, max_window))
            if window_size > 0:
                result.iloc[i] = np.mean(arr[max(0, i - window_size + 1):i+1])
        return result

    smoothed = []
    for _, gdf in df.groupby('day_group'):
        smoothed.append(rolling_avg_variable_window(gdf['combined_volume'], gdf['bars_in_day'], smooth_bars))
    df['smoothed_volume'] = pd.concat(smoothed)

    df.drop(columns=['date','is_new_day','is_up_bar','is_down_bar',
                      'day_group','bars_in_day','buy_vol_increment','sel_vol_increment'], inplace=True)
    return df

# ═══════════════════════════════════════════════════════════
# INDICATOR CLASS
# ═══════════════════════════════════════════════════════════

class ScreenerIndicator:

    def __init__(self, df):
        self.df = df.copy()
        if all(c in df.columns for c in ['Open','High','Low','Close','Volume']):
            self.df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}, inplace=True)
        self.df.index = _to_ist(df.index)
        self.df["in_session"] = _in_session_mask(self.df.index)

    def run(self):
        df = self.df
        self._calc_orb(df); self._calc_fib_counts(df); self._calc_daily_stats(df)
        self._calc_vwap(df); self._calc_donchian(df); self._calc_gravity_speed(df)
        self._calc_majority_fall(df); self._calc_advanced_rise(df); self._calc_flip_signals(df)
        return df

    def _calc_orb(self, df):
        n = len(df); orb_high = np.full(n, np.nan); orb_low = np.full(n, np.nan)
        orb_done = np.zeros(n, dtype=bool)
        cur_high = np.nan; cur_low = np.nan; done = False; prev_date = None
        orb_cutoff = dtime(SESSION_START.hour, SESSION_START.minute + ORB_MINUTES)
        for i in range(n):
            ts = df.index[i]; d, t = ts.date(), ts.time(); row = df.iloc[i]
            if d != prev_date: cur_high = np.nan; cur_low = np.nan; done = False; prev_date = d
            if row["in_session"] and not done:
                cur_high = row["high"] if np.isnan(cur_high) else max(cur_high, row["high"])
                cur_low  = row["low"]  if np.isnan(cur_low)  else min(cur_low,  row["low"])
                if t >= orb_cutoff: done = True
            orb_high[i] = cur_high if done else np.nan
            orb_low[i]  = cur_low  if done else np.nan
            orb_done[i] = done
        df["orb_high"] = orb_high; df["orb_low"] = orb_low; df["orb_done"] = orb_done
        rng = orb_high - orb_low
        df["fib0"] = orb_high; df["fib50"] = orb_high - rng * 0.5; df["fib100"] = orb_low

    def _calc_fib_counts(self, df):
        c1 = c2 = c3 = c4 = 0; cnt1, cnt2, cnt3, cnt4 = (np.zeros(len(df)) for _ in range(4))
        for i in range(len(df)):
            row = df.iloc[i]
            if row["in_session"] and not np.isnan(row.get("fib0", np.nan)):
                c = row["close"]; f0, f50, f100 = row["fib0"], row["fib50"], row["fib100"]
                if   c >= f0:   c1 += 1
                elif c >= f50:  c2 += 1
                elif c > f100:  c3 += 1
                else:           c4 += 1
            cnt1[i], cnt2[i], cnt3[i], cnt4[i] = c1, c2, c3, c4
        df["zone_above_l1"] = cnt1; df["zone_l1_l2"] = cnt2
        df["zone_l2_l3"]    = cnt3; df["zone_below_l3"] = cnt4

    def _calc_daily_stats(self, df):
        n = len(df); hh_cnt = np.zeros(n); ll_cnt = np.zeros(n)
        day_vol = np.zeros(n); hl_diff = np.zeros(n)
        prev_date = None; cur_hh = cur_ll = 0; cum_vol = 0.0
        day_high = -np.inf; day_low = np.inf; prev_high = prev_low = np.nan
        for i in range(n):
            ts = df.index[i]; row = df.iloc[i]; d = ts.date()
            if d != prev_date:
                cur_hh = cur_ll = 0; cum_vol = 0.0; day_high = -np.inf; day_low = np.inf
                prev_high = row["high"]; prev_low = row["low"]; prev_date = d
            if row["in_session"]:
                if row["high"] > prev_high: cur_hh += 1; prev_high = row["high"]
                if row["low"]  < prev_low:  cur_ll += 1; prev_low  = row["low"]
                cum_vol += row["volume"]; day_high = max(day_high, row["high"]); day_low = min(day_low, row["low"])
            hh_cnt[i] = cur_hh; ll_cnt[i] = cur_ll
            day_vol[i] = cum_vol; hl_diff[i] = (day_high - day_low) if day_high != -np.inf else 0.0
        df["hh_daily"] = hh_cnt; df["ll_daily"] = ll_cnt
        df["daily_vol"] = day_vol; df["daily_hl_diff"] = hl_diff

    def _calc_vwap(self, df):
        hlc3 = (df["high"] + df["low"] + df["close"]).values / 3
        vol = df["volume"].values; sess = df["in_session"].values; n = len(df)
        vwap_arr = np.full(n, np.nan); cum_tp_vol = cum_vol = 0.0; prev_date = None
        for i in range(n):
            d = df.index[i].date()
            if d != prev_date: cum_tp_vol = cum_vol = 0.0; prev_date = d
            if sess[i]:
                cum_tp_vol += hlc3[i] * vol[i]; cum_vol += vol[i]
                vwap_arr[i] = cum_tp_vol / cum_vol if cum_vol > 0 else np.nan
            else:
                vwap_arr[i] = vwap_arr[i-1] if i > 0 else np.nan
        df["vwap"] = vwap_arr
        dev_raw = (df["close"].values - vwap_arr) / vwap_arr * 100
        dev = _sma(dev_raw, SMOOTH_LEN) if SMOOTH_LEN > 1 else dev_raw
        df["vwap_dev"] = dev; df["vwap_dev_line"] = vwap_arr - dev

    def _calc_donchian(self, df):
        high = df["high"].values; low = df["low"].values
        close = df["close"].values; sess = df["in_session"].values; n = len(df)
        def _don(length):
            trend = np.zeros(n, dtype=int)
            for i in range(length, n):
                if not sess[i]: trend[i] = trend[i-1]; continue
                hh = np.max(high[i-length:i]); ll = np.min(low[i-length:i])
                if   close[i] > hh: trend[i] = 1
                elif close[i] < ll: trend[i] = -1
                else:               trend[i] = trend[i-1]
            return trend
        main = _don(DONCHIAN_LEN); df["don_main"] = main
        sub = {}
        for offset in range(10):
            length = DONCHIAN_LEN - offset
            if length >= 1:
                t = _don(length); df[f"don_sub_{length}"] = t; sub[length] = t
        s30 = sub.get(DONCHIAN_LEN-5, np.zeros(n)); s35 = sub.get(DONCHIAN_LEN-6, np.zeros(n))
        s40 = sub.get(DONCHIAN_LEN-7, np.zeros(n))
        df["all_green"] = (main==1)  & (s30==1)  & (s35==1)  & (s40==1)
        df["all_red"]   = (main==-1) & (s30==-1) & (s35==-1) & (s40==-1)

    def _calc_gravity_speed(self, df):
        n = len(df); close = df["close"].values; sess = df["in_session"].values
        weights = np.arange(GRAVITY_LEN, 0, -1, dtype=float)
        gravity = np.full(n, np.nan); dist_avg = np.full(n, np.nan)
        speed_avg = np.full(n, np.nan); escape_arr = np.zeros(n, bool); crash_arr = np.zeros(n, bool)
        for i in range(GRAVITY_LEN, n):
            if not sess[i]:
                gravity[i]=gravity[i-1]; dist_avg[i]=dist_avg[i-1]
                speed_avg[i]=speed_avg[i-1]; escape_arr[i]=escape_arr[i-1]; crash_arr[i]=crash_arr[i-1]; continue
            window = close[i-GRAVITY_LEN+1:i+1]
            gravity[i] = np.dot(window, weights) / weights.sum()
            dist = close[i] - gravity[i]
            grav_win = gravity[max(0,i-GRAVITY_LEN+1):i+1]
            abs_dists = np.abs(close[max(0,i-GRAVITY_LEN+1):i+1] - grav_win)
            dist_avg[i] = np.nanmean(abs_dists)
            if i >= SPEED_LEN:
                speed = close[i] - close[i-SPEED_LEN]
                speed_win = close[max(0,i-GRAVITY_LEN):i+1]
                speeds = np.abs(np.diff(speed_win, n=SPEED_LEN))
                speed_avg[i] = np.nanmean(speeds) if len(speeds) else np.nan
                if not np.isnan(dist_avg[i]) and not np.isnan(speed_avg[i]):
                    escape_arr[i] = (abs(speed) > SPEED_MULT*speed_avg[i]) and (abs(dist) > dist_avg[i])
                    crash_arr[i]  = (abs(dist) > CRASH_MULT*dist_avg[i])   and (abs(speed) < speed_avg[i])
        df["gravity"]=gravity; df["dist_avg"]=dist_avg
        df["speed_avg"]=speed_avg; df["escape"]=escape_arr; df["crash_risk"]=crash_arr

    def _calc_majority_fall(self, df):
        log_ret = np.log(df["close"].values / df["open"].values)
        ret_mean = _sma(log_ret, WINDOW_FALL); ret_std = _rolling_std(log_ret, WINDOW_FALL)
        zscore = np.where(ret_std > 0, (log_ret - ret_mean) / ret_std, 0.0)
        df["zscore_fall"] = zscore; df["major_fall"] = zscore < -THRESHOLD_FALL

    def _calc_advanced_rise(self, df):
        close = df["close"].values; volume = df["volume"].values
        high = df["high"].values; n = len(df)
        log_ret = np.concatenate(([0.0], np.log(close[1:] / close[:-1])))
        ret_mean = _sma(log_ret, WINDOW_AD); ret_std = _rolling_std(log_ret, WINDOW_AD)
        zscore_ad = np.where(ret_std > 0.0001, (log_ret - ret_mean) / ret_std, 0.0)
        ret_std_c = np.nan_to_num(ret_std)
        eff_thr = np.array([
            THRESHOLD_AD * 1.3
            if np.sum(ret_std_c[:i+1] < ret_std_c[i]) / max(i,1) * 100 > 70
            else THRESHOLD_AD for i in range(n)
        ])
        vol_avg = _sma(volume.astype(float), WINDOW_AD)
        vol_ok  = (volume > vol_avg * VOL_MULTIPLIER) if USE_VOLUME else np.ones(n, bool)
        rsi     = _rsi(close, 14); rsi_ok = (rsi > RSI_THRESHOLD) if USE_RSI else np.ones(n, bool)
        rises   = (close > np.concatenate(([close[0]], close[:-1]))).astype(float)
        mom_sum = _sma(rises, 3); mom_ok = mom_sum >= (2/3)
        highest_20 = np.array([np.max(high[max(0,i-19):i+1]) for i in range(n)])
        breakout   = (close > np.concatenate(([highest_20[0]], highest_20[:-1]))) \
                     if DETECT_BREAKOUTS else np.zeros(n, bool)
        ret_std50  = _sma(ret_std_c, 50)
        was_consol = np.concatenate(([False], ret_std_c[:-1] < ret_std50[:-1]))
        breakout_consol = breakout & was_consol
        basic_rise     = (zscore_ad > eff_thr) & vol_ok
        confirmed_rise = basic_rise & mom_ok & rsi_ok
        explosive_rise = confirmed_rise & breakout_consol
        parabolic_rise = (zscore_ad > eff_thr * 1.5) & vol_ok
        df["zscore_ad"]=zscore_ad; df["basic_rise"]=basic_rise
        df["confirmed_rise"]=confirmed_rise; df["explosive_rise"]=explosive_rise
        df["parabolic_rise"]=parabolic_rise

    def _calc_flip_signals(self, df):
        vwap = df["vwap"].values; dev_line = df["vwap_dev_line"].values
        grav = df["gravity"].values; sess = df["in_session"].values
        ll = df["ll_daily"].values; hh = df["hh_daily"].values; n = len(df)
        cross_over = np.zeros(n, bool); cross_under = np.zeros(n, bool)
        for i in range(2, n):
            cross_over[i]  = (vwap[i-1] > dev_line[i-1]) and (vwap[i-2] <= dev_line[i-2])
            cross_under[i] = (vwap[i-1] < dev_line[i-1]) and (vwap[i-2] >= dev_line[i-2])
        major_fall = df["major_fall"].values
        b_rise=df["basic_rise"].values; c_rise=df["confirmed_rise"].values
        e_rise=df["explosive_rise"].values; p_rise=df["parabolic_rise"].values
        df["signal_green_to_red"] = (sess & (vwap>grav) & (vwap>dev_line) &
                                     cross_over & (ll>hh)) | major_fall
        df["signal_red_to_green"] = (sess & (vwap<grav) & (vwap<dev_line) &
                                     cross_under & (hh>ll)) | b_rise | c_rise | e_rise | p_rise

    def summary_dict(self):
        last = self.df.iloc[-1]
        def _f(key):
            v = last.get(key, np.nan)
            if isinstance(v, (bool, np.bool_)): return bool(v)
            try:
                fv = float(v)
                return None if np.isnan(fv) else round(fv, 4)
            except: return str(v)
        return {
            "datetime":          str(self.df.index[-1]),
            "close":             _f("close"),
            "fib0":              _f("fib0"),
            "fib50":             _f("fib50"),
            "fib100":            _f("fib100"),
            "zone_above_l1":     int(last.get("zone_above_l1", 0)),
            "zone_l1_l2":        int(last.get("zone_l1_l2", 0)),
            "zone_l2_l3":        int(last.get("zone_l2_l3", 0)),
            "zone_below_l3":     int(last.get("zone_below_l3", 0)),
            "hh_daily":          int(last.get("hh_daily", 0)),
            "ll_daily":          int(last.get("ll_daily", 0)),
            "daily_hl_diff":     _f("daily_hl_diff"),
            "daily_vol":         get_vol_str(last.get("daily_vol", 0)),
            "vwap":              _f("vwap"),
            "vwap_dev_pct":      _f("vwap_dev"),
            "gravity":           _f("gravity"),
            "don_main":          int(last.get("don_main", 0)),
            "all_green":         bool(last.get("all_green", False)),
            "all_red":           bool(last.get("all_red", False)),
            "major_fall":        bool(last.get("major_fall", False)),
            "basic_rise":        bool(last.get("basic_rise", False)),
            "confirmed_rise":    bool(last.get("confirmed_rise", False)),
            "explosive_rise":    bool(last.get("explosive_rise", False)),
            "parabolic_rise":    bool(last.get("parabolic_rise", False)),
            "escape":            bool(last.get("escape", False)),
            "crash_risk":        bool(last.get("crash_risk", False)),
            "signal_green_to_red": bool(last.get("signal_green_to_red", False)),
            "signal_red_to_green": bool(last.get("signal_red_to_green", False)),
        }


# ═══════════════════════════════════════════════════════════
# CHART GENERATOR
# ═══════════════════════════════════════════════════════════

def generate_chart(df: pd.DataFrame, ticker: str) -> io.BytesIO:
    """Build the 4-panel chart and return it as a PNG BytesIO buffer."""

    comb_vals   = df["combined_volume"].values
    smooth_vals = df["smoothed_volume"].values

    buy_signals  = [False]
    sell_signals = [False]
    for i in range(1, len(df)):
        buy_signals.append(comb_vals[i-1] <= smooth_vals[i-1] and comb_vals[i] > smooth_vals[i])
        sell_signals.append(comb_vals[i-1] >= smooth_vals[i-1] and comb_vals[i] < smooth_vals[i])
    buy_signals  = np.array(buy_signals)
    sell_signals = np.array(sell_signals)

    xs = np.arange(len(df))

    fig = plt.figure(figsize=(20, 14), facecolor=DARK_BG)
    gs  = GridSpec(4, 1, figure=fig, height_ratios=[6, 2.0, 1.8, 1.8],
                   hspace=0.06, left=0.06, right=0.97, top=0.93, bottom=0.08)
    ax_price  = fig.add_subplot(gs[0])
    ax_excess = fig.add_subplot(gs[1], sharex=ax_price)
    ax_vol    = fig.add_subplot(gs[2], sharex=ax_price)
    ax_dev    = fig.add_subplot(gs[3], sharex=ax_price)

    for ax in (ax_price, ax_excess, ax_vol, ax_dev):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.yaxis.label.set_color(TEXT_COL)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Panel 1: Candlesticks ──
    W_BODY = 0.5; W_WICK = 0.08
    for i, (_, row) in enumerate(df.iterrows()):
        bull = row["close"] >= row["open"]; col = UP_COL if bull else DN_COL
        body_b = min(row["open"], row["close"])
        body_h = abs(row["close"] - row["open"]) or 0.01
        ax_price.add_patch(mpatches.Rectangle((xs[i]-W_WICK/2, row["low"]), W_WICK, row["high"]-row["low"], color=col, zorder=2))
        ax_price.add_patch(mpatches.Rectangle((xs[i]-W_BODY/2, body_b),    W_BODY, body_h,                  color=col, zorder=3))

    ax_price.plot(xs, df["vwap"].values,    color=VWAP_COL, lw=1.6, label="VWAP",         zorder=5)
    ax_price.plot(xs, df["gravity"].values, color=GRAV_COL, lw=1.2, linestyle="--", label="Gravity (CoG)", zorder=5)

    for val, col, lbl in [
        (df["fib0"].iloc[-1],   FIB0_COL,   "ORB 0%"),
        (df["fib50"].iloc[-1],  FIB50_COL,  "ORB 50%"),
        (df["fib100"].iloc[-1], FIB100_COL, "ORB 100%"),
    ]:
        if not np.isnan(val):
            ax_price.axhline(val, color=col, lw=1.0, linestyle=":", alpha=0.85, zorder=4)
            ax_price.text(xs[-1]+0.4, val, f" {lbl}\n {val:.1f}", color=col, fontsize=7, va="center", zorder=6)

    g2r   = df["signal_green_to_red"].values.astype(bool)
    r2g   = df["signal_red_to_green"].values.astype(bool)
    mfall = df["major_fall"].values.astype(bool)

    if g2r.any():
        ax_price.scatter(xs[g2r], df["high"].values[g2r]*1.002, marker="v", color=DN_COL, s=80, zorder=7, label="G→R Flip")
    if r2g.any():
        ax_price.scatter(xs[r2g], df["low"].values[r2g]*0.998,  marker="^", color=UP_COL, s=80, zorder=7, label="R→G Flip")
    if mfall.any():
        for xi, yi in zip(xs[mfall], df["high"].values[mfall]):
            ax_price.text(xi, yi*1.003, "⧹⧹⧹", color=DN_COL, fontsize=7, ha="center", va="bottom", zorder=8)

    for col_name, col, sz in [
        ("basic_rise","#b3f5b3",40), ("confirmed_rise","#26a641",60),
        ("explosive_rise","#00e5ff",80), ("parabolic_rise","#ffd700",80),
    ]:
        m = df[col_name].values.astype(bool)
        if m.any():
            ax_price.scatter(xs[m], df["low"].values[m]*0.9975, marker="^", color=col, s=sz, zorder=7,
                             label=col_name.replace("_"," ").title())

    KEY_TIMES = {
        dtime(9,15):("Session Open","#ffd700","--"),
        dtime(9,20):("ORB End (5m)","#ff9f00",":"),
        dtime(9,30):("9:30",GRID_COL,":"),
        dtime(9,45):("9:45",GRID_COL,":"),
        dtime(10,0):("10:00","#ffd700","--"),
    }
    for kt, (label, kc, ls) in KEY_TIMES.items():
        idxs = [i for i, ts in enumerate(df.index) if ts.time() == kt]
        if idxs:
            for ax in (ax_price, ax_excess, ax_vol, ax_dev):
                ax.axvline(idxs[0], color=kc, lw=0.9, linestyle=ls, alpha=0.8, zorder=1)

    # ── Panel 2: Volume ──
    bull_mask = df["close"].values >= df["open"].values
    ax_vol.bar(xs, df["volume"].values, color=np.where(bull_mask, UP_COL, DN_COL), width=0.8, alpha=0.85)
    ax_vol.set_ylabel("Volume", color=TEXT_COL, fontsize=8)
    avg_vol = df["daily_vol"].iloc[-1] / max(1, len(df))
    ax_vol.axhline(avg_vol, color="#888888", lw=1.0, linestyle="--")

    # ── Panel 3: VWAP Dev % ──
    dev_vals = df["vwap_dev"].values
    ax_dev.fill_between(xs, dev_vals, 0, where=(dev_vals>=0), color=UP_COL, alpha=0.35)
    ax_dev.fill_between(xs, dev_vals, 0, where=(dev_vals<0),  color=DN_COL, alpha=0.35)
    ax_dev.plot(xs, dev_vals, color=VWAP_COL, lw=1.2, zorder=4)
    ax_dev.axhline(0, color=TEXT_COL, lw=0.8)
    ax_dev.set_ylabel("VWAP Dev %", color=TEXT_COL, fontsize=8)

    # ── Panel 4: Excess Combined Volume ──
    ax_excess.fill_between(xs, comb_vals, 0, where=(comb_vals>=0), color=UP_COL, alpha=0.25)
    ax_excess.fill_between(xs, comb_vals, 0, where=(comb_vals<0),  color=DN_COL, alpha=0.25)
    ax_excess.plot(xs, comb_vals,   color=COMB_COL,   lw=1.0, alpha=0.8, label="Combined Vol")
    ax_excess.plot(xs, smooth_vals, color=SMOOTH_COL, lw=1.5, label=f"Smoothed ({EXCESS_SMOOTH_BARS}b)")
    ax_excess.axhline(0, color=TEXT_COL, lw=0.8)

    if buy_signals.any():
        ax_excess.scatter(xs[buy_signals],  comb_vals[buy_signals],  marker="^", color=UP_COL, s=100, zorder=10, label="Buy")
    if sell_signals.any():
        ax_excess.scatter(xs[sell_signals], comb_vals[sell_signals], marker="v", color=DN_COL, s=100, zorder=10, label="Sell")

    ax_excess.set_ylabel("Excess Vol", color=TEXT_COL, fontsize=8)
    ax_excess.legend(loc="upper left", fontsize=6.5, facecolor="#161b22",
                     edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=4)

    # ── Axes / Ticks ──
    ax_price.set_xlim(-0.5, len(xs) - 0.5 + 4)
    ax_price.set_ylabel("Price (INR)", color=TEXT_COL, fontsize=9)
    plt.setp(ax_price.get_xticklabels(),  visible=False)
    plt.setp(ax_excess.get_xticklabels(), visible=False)
    plt.setp(ax_vol.get_xticklabels(),    visible=False)

    tick_pos = np.linspace(0, len(xs)-1, min(12, len(xs)), dtype=int)
    ax_dev.set_xticks(tick_pos)
    ax_dev.set_xticklabels(
        [df.index[i].strftime("%H:%M") for i in tick_pos],
        color=TEXT_COL, fontsize=8, rotation=45, ha="right")
    ax_dev.set_xlabel("Time (IST)", color=TEXT_COL, fontsize=9)

    legend_handles = [
        mlines.Line2D([], [], color=VWAP_COL,  lw=1.6,           label="VWAP"),
        mlines.Line2D([], [], color=GRAV_COL,  lw=1.2, ls="--",  label="Gravity"),
        mlines.Line2D([], [], color=FIB0_COL,  lw=1.0, ls=":",   label="ORB 0%"),
        mlines.Line2D([], [], color=FIB50_COL, lw=1.0, ls=":",   label="ORB 50%"),
        mlines.Line2D([], [], color=FIB100_COL,lw=1.0, ls=":",   label="ORB 100%"),
        mlines.Line2D([], [], marker="v", color=DN_COL, lw=0, ms=7, label="G→R Flip"),
        mlines.Line2D([], [], marker="^", color=UP_COL, lw=0, ms=7, label="R→G Flip"),
        mlines.Line2D([], [], marker="^", color=UP_COL, lw=0, ms=7, label="Buy Signal"),
        mlines.Line2D([], [], marker="v", color=DN_COL, lw=0, ms=7, label="Sell Signal"),
    ]
    ax_price.legend(handles=legend_handles, loc="upper left", fontsize=7.5,
                    facecolor="#161b22", edgecolor=GRID_COL, labelcolor=TEXT_COL, ncol=2)

    title_date = df.index[0].strftime("%d %b %Y")
    fig.suptitle(
        f"NSE · {ticker}  ·  {title_date}  ·  09:15–15:30 IST  ·  1-Minute Bars",
        color=TEXT_COL, fontsize=13, fontweight="bold", y=0.97
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════
# PIVOT DETECTION
# ═══════════════════════════════════════════════════════════

def pivotid(df1, l, n1, n2):
    """
    Returns:
        0 → not a pivot
        1 → pivot low
        2 → pivot high
        3 → both (inside bar / doji extreme)
    """
    if l - n1 < 0 or l + n2 >= len(df1):
        return 0
    pividlow = 1; pividhigh = 1
    for i in range(l - n1, l + n2 + 1):
        if df1["low"].iloc[l]  > df1["low"].iloc[i]:  pividlow  = 0
        if df1["high"].iloc[l] < df1["high"].iloc[i]: pividhigh = 0
    if pividlow and pividhigh: return 3
    elif pividlow:             return 1
    elif pividhigh:            return 2
    else:                      return 0


def generate_pivot_chart(raw_df: pd.DataFrame, ticker: str) -> io.BytesIO:
    """Compute pivots on raw_df and return a dark-themed histogram PNG."""
    df = raw_df.copy().reset_index(drop=False)
    df.index = range(len(df))

    # Use integer-safe column access (reset_index may have added datetime col)
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    df["pivot"] = [pivotid(df, i, 10, 10) for i in range(len(df))]

    high_values = df[df["pivot"] == 2]["high"]
    low_values  = df[df["pivot"] == 1]["low"]

    if high_values.empty and low_values.empty:
        # Return a simple "no pivots" placeholder
        fig, ax = plt.subplots(figsize=(12, 4), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)
        ax.text(0.5, 0.5, "No pivot points detected in this range",
                color=TEXT_COL, ha="center", va="center", fontsize=13, transform=ax.transAxes)
        ax.axis("off")
    else:
        all_prices = pd.concat([high_values, low_values])
        price_range = all_prices.max() - all_prices.min()
        bin_width   = max(price_range / 80, 0.001)   # adaptive bins, max 80
        bins = max(10, int(price_range / bin_width))

        fig, ax = plt.subplots(figsize=(12, 4), facecolor=DARK_BG)
        ax.set_facecolor(DARK_BG)

        if not high_values.empty:
            ax.hist(high_values, bins=bins, alpha=0.65, label=f"Pivot Highs ({len(high_values)})",
                    color=DN_COL, edgecolor="#00000033", linewidth=0.4)
        if not low_values.empty:
            ax.hist(low_values,  bins=bins, alpha=0.65, label=f"Pivot Lows  ({len(low_values)})",
                    color=VWAP_COL, edgecolor="#00000033", linewidth=0.4)

        # Mark pivot high/low means as vertical lines
        if not high_values.empty:
            ax.axvline(high_values.mean(), color=DN_COL,    lw=1.4, linestyle="--",
                       label=f"Avg High {high_values.mean():.2f}")
        if not low_values.empty:
            ax.axvline(low_values.mean(),  color=VWAP_COL,  lw=1.4, linestyle="--",
                       label=f"Avg Low  {low_values.mean():.2f}")

        ax.set_xlabel("Price (INR)", color=TEXT_COL, fontsize=10)
        ax.set_ylabel("Frequency",   color=TEXT_COL, fontsize=10)
        ax.tick_params(colors=TEXT_COL, labelsize=9)
        for spine in ax.spines.values(): spine.set_edgecolor(GRID_COL)
        ax.grid(color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, facecolor="#161b22", edgecolor=GRID_COL, labelcolor=TEXT_COL)

        fig.suptitle(
            f"Pivot Highs & Lows — {ticker}  (n1=n2=10 bars)",
            color=TEXT_COL, fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history/<ticker>")
def get_history(ticker):
    """Raw OHLCV. Params: range (default 1mo), interval (default 1d)."""
    try:
        range_   = request.args.get("range",    "1mo")
        interval = request.args.get("interval", "1d")
        df = fetch_ohlcv(ticker, interval=interval, range_=range_)

        records = []
        for ts, row in df.iterrows():
            records.append({
                "date":   ts.strftime("%Y-%m-%d %H:%M"),
                "open":   round(row["open"],   2),
                "high":   round(row["high"],   2),
                "low":    round(row["low"],    2),
                "close":  round(row["close"],  2),
                "volume": int(row["volume"]),
            })

        return jsonify({
            "ticker":   ticker.upper(),
            "range":    range_,
            "interval": interval,
            "count":    len(records),
            "data":     records,
        })

    except req_lib.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/screener/<ticker>")
def get_screener(ticker):
    """Run all indicators and return summary JSON. Params: range (default 1d), interval (default 1m)."""
    try:
        range_   = request.args.get("range",    "1d")
        interval = request.args.get("interval", "1m")
        raw_df = fetch_ohlcv(ticker, interval=interval, range_=range_)

        ind    = ScreenerIndicator(raw_df)
        result = ind.run()
        result = calculate_excess_combined_intraday_volume(result, smooth_bars=EXCESS_SMOOTH_BARS)

        return jsonify({
            "ticker":   ticker.upper(),
            "bars":     len(result),
            "summary":  ind.summary_dict(),
        })

    except req_lib.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chart/<ticker>")
def get_chart(ticker):
    """Generate 4-panel PNG chart. Params: range (default 1d), interval (default 1m)."""
    try:
        range_   = request.args.get("range",    "1d")
        interval = request.args.get("interval", "1m")
        raw_df   = fetch_ohlcv(ticker, interval=interval, range_=range_)

        ind    = ScreenerIndicator(raw_df)
        result = ind.run()
        result = calculate_excess_combined_intraday_volume(result, smooth_bars=EXCESS_SMOOTH_BARS)

        # Filter to last session only (9:15 – 15:30 IST)
        last_date = result.index[-1].date()
        mask = (
            (result.index.date == last_date) &
            (result.index.time >= dtime(9, 15)) &
            (result.index.time <= dtime(15, 30))
        )
        plot_df = result[mask]
        if plot_df.empty:
            return jsonify({"error": "No session bars to plot"}), 404

        buf = generate_chart(plot_df, ticker.upper())
        return send_file(buf, mimetype="image/png",
                         download_name=f"{ticker.upper()}_chart.png")

    except req_lib.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pivot/<ticker>")
def get_pivot(ticker):
    """Pivot histogram PNG. Params: range (default 1d), interval (default 1m)."""
    try:
        range_   = request.args.get("range",    "1d")
        interval = request.args.get("interval", "1m")
        raw_df   = fetch_ohlcv(ticker, interval=interval, range_=range_)
        buf      = generate_pivot_chart(raw_df, ticker.upper())
        return send_file(buf, mimetype="image/png",
                         download_name=f"{ticker.upper()}_pivots.png")
    except req_lib.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# RENKO ENGINE
# ═══════════════════════════════════════════════════════════

def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """RMA-based ATR — matches Pine Script ta.atr exactly."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_renko(df: pd.DataFrame,
                  use_atr: bool = True,
                  atr_len: int = 20,
                  fixed_brick: float = 1.0) -> pd.DataFrame:
    """
    Non-repaintable Renko engine.
    fixed_brick is treated as TRAD_LEN1 * 0.001 when use_atr=False
    (e.g. brick=1000 → 1000 * 0.001 = 1.0 price unit).
    """
    closes = df["close"].values
    n = len(closes)

    if use_atr:
        brick_sizes = compute_atr(df, atr_len).values
    else:
        # Match original: TRAD_LEN = TRAD_LEN1 * 0.001
        brick_sizes = np.full(n, fixed_brick * 0.001)

    renko_close = np.full(n, np.nan)
    renko_trend = np.zeros(n, dtype=int)
    rc = float(closes[0]); rt = 0

    for i in range(n):
        brick = brick_sizes[i]
        if np.isnan(brick) or brick <= 0:
            renko_close[i] = rc; renko_trend[i] = rt; continue
        price_move = closes[i] - rc
        if price_move >= brick:
            bricks = int(price_move / brick); rc += bricks * brick; rt = 1
        elif price_move <= -brick:
            bricks = int(abs(price_move) / brick); rc -= bricks * brick; rt = -1
        renko_close[i] = rc; renko_trend[i] = rt

    result = df.copy()
    result["brick_size"]  = brick_sizes
    result["renko_close"] = renko_close
    result["renko_trend"] = renko_trend
    return result


def check_renko_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """UP alert = first bar after trend flips to +1. DOWN = first bar after flip to -1."""
    df = df.copy()
    prev = df["renko_trend"].shift(1)
    df["alert_up"]   = (df["renko_trend"] ==  1) & (prev != 1)
    df["alert_down"] = (df["renko_trend"] == -1) & (prev != -1)
    return df


def _session_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 09:15–15:15 IST bars (matches original between_time call)."""
    ist = _to_ist(df.index)
    mask = np.array(
        [(t.time() >= dtime(9, 15) and t.time() <= dtime(15, 15)) for t in ist],
        dtype=bool,
    )
    return df[mask]


def generate_renko_chart(df: pd.DataFrame, ticker: str,
                         use_atr: bool = True, atr_len: int = 20,
                         fixed_brick: float = 1000.0,
                         max_bars: int = 300) -> io.BytesIO:
    """Filter to session, compute Renko, return 3-panel dark PNG."""

    # Session filter 09:15–15:15 IST (matches original)
    df_sess = _session_filter(df)
    if df_sess.empty:
        df_sess = df   # fallback — outside market hours

    result = compute_renko(df_sess, use_atr=use_atr,
                            atr_len=atr_len, fixed_brick=fixed_brick)
    result = check_renko_alerts(result)

    # Take tail and preserve datetime for annotations — convert to IST
    df_plot = result.tail(max_bars).copy()
    # Convert index to IST for correct time labels
    df_plot["original_datetime"] = _to_ist(df_plot.index)
    df_plot = df_plot.reset_index(drop=True)

    color_map = {1: "lime", -1: "red", 0: "gray"}
    colors = df_plot["renko_trend"].map(color_map).fillna("gray")

    fig, axes = plt.subplots(3, 1, figsize=(16, 9),
                             gridspec_kw={"height_ratios": [3, 1, 1]},
                             sharex=True)
    fig.patch.set_facecolor("#131722")
    for ax in axes:
        ax.set_facecolor("#131722")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    ax1 = axes[0]

    # Faint price line
    ax1.plot(df_plot["close"].values, color="#888888",
             linewidth=0.8, alpha=0.5, label="Price")

    # Renko coloured line + scatter dots
    for i in range(1, len(df_plot)):
        ax1.plot([i-1, i],
                 [df_plot["renko_close"].iloc[i-1], df_plot["renko_close"].iloc[i]],
                 color=colors.iloc[i], linewidth=2.5)
    ax1.scatter(range(len(df_plot)), df_plot["renko_close"],
                c=colors, s=15, zorder=4)

    # UP flip markers + time annotation
    up_bars   = df_plot[df_plot["alert_up"]]
    down_bars = df_plot[df_plot["alert_down"]]

    ax1.scatter(up_bars.index, up_bars["renko_close"],
                marker="^", color="lime", s=80, zorder=5, label="Renko UP ▲")
    for i, row in up_bars.iterrows():
        ax1.annotate(
            f"{row['original_datetime'].strftime('%H:%M')}\n₹{row['close']:.2f}",
            (i, row["renko_close"]),
            textcoords="offset points", xytext=(5, 5),
            ha="left", va="bottom", fontsize=7, color="lime",
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="lime", lw=0.5, alpha=0.8)
        )

    # DOWN flip markers + IST time + close price annotation
    ax1.scatter(down_bars.index, down_bars["renko_close"],
                marker="v", color="red", s=80, zorder=5, label="Renko DOWN ▼")
    for i, row in down_bars.iterrows():
        ax1.annotate(
            f"{row['original_datetime'].strftime('%H:%M')}\n₹{row['close']:.2f}",
            (i, row["renko_close"]),
            textcoords="offset points", xytext=(5, -28),
            ha="left", va="top", fontsize=7, color="#ff5555",
            bbox=dict(boxstyle="round,pad=0.2", fc="#0d1117", ec="red", lw=0.5, alpha=0.8)
        )

    ax1.set_ylabel("Price", color="white")
    brick_label = f"ATR-{atr_len}" if use_atr else f"Fixed-{fixed_brick * 0.001:.4g}"
    ax1.set_title(
        f"Non-Repaintable Renko  ·  {ticker}  ·  {brick_label}",
        color="white", fontsize=12
    )
    legend_patches = [
        mpatches.Patch(color="lime", label="Up"),
        mpatches.Patch(color="red",  label="Down"),
        mpatches.Patch(color="gray", label="Neutral"),
    ]
    ax1.legend(handles=legend_patches, facecolor="#1e222d",
               labelcolor="white", fontsize=8, loc="upper left")

    # Brick size panel
    ax2 = axes[1]
    ax2.plot(df_plot["brick_size"].values, color="#f0a500", linewidth=1.2)
    ax2.set_ylabel("Brick Sz", color="white", fontsize=8)
    ax2.yaxis.set_tick_params(labelcolor="white")

    # Trend direction bar panel
    ax3 = axes[2]
    trend_colors = [color_map.get(t, "gray") for t in df_plot["renko_trend"]]
    ax3.bar(range(len(df_plot)), df_plot["renko_trend"],
            color=trend_colors, width=1)
    ax3.axhline(0, color="white", linewidth=0.4)
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["DN", "—", "UP"], color="white", fontsize=7)
    ax3.set_ylabel("Trend", color="white", fontsize=8)

    axes[-1].set_xlabel("Bar Index", color="white")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/renko/<ticker>")
def get_renko(ticker):
    """
    Renko chart PNG.
    Params:
      range     : 1d 5d 1mo 3mo …    (default 1d)
      interval  : 1m 5m 15m 1d …    (default 1m)
      atr       : true / false        (default true)
      atr_len   : int                 (default 20)
      brick     : float (TRAD_LEN1)  (default 1000; actual = brick * 0.001)
      max_bars  : int                 (default 300)
    """
    try:
        range_      = request.args.get("range",    "1d")
        interval    = request.args.get("interval", "1m")
        use_atr     = request.args.get("atr",      "true").lower() == "true"
        atr_len     = int(request.args.get("atr_len", 20))
        fixed_brick = float(request.args.get("brick", 1000.0))
        max_bars    = int(request.args.get("max_bars", 300))

        raw_df = fetch_ohlcv(ticker, interval=interval, range_=range_)
        buf    = generate_renko_chart(raw_df, ticker.upper(),
                                      use_atr=use_atr, atr_len=atr_len,
                                      fixed_brick=fixed_brick, max_bars=max_bars)
        return send_file(buf, mimetype="image/png",
                         download_name=f"{ticker.upper()}_renko.png")

    except req_lib.HTTPError as e:
        return jsonify({"error": f"Yahoo Finance returned {e.response.status_code}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
