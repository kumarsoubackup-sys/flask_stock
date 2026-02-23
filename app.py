import datetime
import warnings
warnings.filterwarnings("ignore")

import pytz
import numpy as np
import pandas as pd
import yfinance as yf

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
IST           = pytz.timezone("Asia/Kolkata")
TICKER        = "DATAMATICS.NS"
ATR_LEN       = 20
IS_ATR        = True
TRAD_LEN1     = 1000
TRAD_LEN      = TRAD_LEN1 * 0.001

SHOW_BRICKS   = True
SHOW_TREND    = True
MAX_BARS      = 300
REFRESH_SEC   = 60        # advance 1 minute every N seconds

SESSION_START = (9, 15)
SESSION_END   = (15, 15)

DARK_BG   = "#131722"
SPINE_CLR = "#444444"
TEXT_CLR  = "white"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def fetch_all_data():
    raw = yf.download(TICKER, period="1d", interval="1m", progress=False)
    if raw.empty:
        return None
    raw.index = (
        raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index
    ).tz_convert(IST)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.rename(columns={"Open":"open","High":"high","Low":"low",
                         "Close":"close","Volume":"volume"}, inplace=True)
    raw = raw.between_time(
        f"{SESSION_START[0]:02d}:{SESSION_START[1]:02d}",
        f"{SESSION_END[0]:02d}:{SESSION_END[1]:02d}"
    )
    return raw[["open","high","low","close"]].copy()


def slice_up_to_minute(full_df, end_hour, end_minute):
    cutoff = datetime.time(end_hour, end_minute)
    mask   = [t <= cutoff for t in full_df.index.time]
    return full_df.loc[mask]


def compute_atr(df, period):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_renko(df):
    closes      = df["close"].values
    n           = len(closes)
    brick_sizes = compute_atr(df, ATR_LEN).values if IS_ATR else np.full(n, TRAD_LEN)

    renko_close = np.full(n, np.nan)
    renko_trend = np.zeros(n, dtype=int)
    rc, rt      = float(closes[0]), 0

    for i in range(n):
        brick = brick_sizes[i]
        if np.isnan(brick) or brick <= 0:
            renko_close[i] = rc
            renko_trend[i] = rt
            continue
        move = closes[i] - rc
        if move >= brick:
            rc += int(move / brick) * brick
            rt  = 1
        elif move <= -brick:
            rc -= int(abs(move) / brick) * brick
            rt  = -1
        renko_close[i] = rc
        renko_trend[i] = rt

    result                = df.copy()
    result["brick_size"]  = brick_sizes
    result["renko_close"] = renko_close
    result["renko_trend"] = renko_trend
    return result


def check_alerts(df):
    df   = df.copy()
    prev = df["renko_trend"].shift(1)
    df["alert_up"]   = (df["renko_trend"] ==  1) & (prev != 1)
    df["alert_down"] = (df["renko_trend"] == -1) & (prev != -1)
    return df


def build_figure(data, minute_offset, seconds_left):
    df = data.tail(MAX_BARS).copy()
    df["bar_idx"] = range(len(df))

    rows       = 3 if SHOW_TREND else 2
    row_heights = [0.6, 0.2, 0.2] if SHOW_TREND else [0.7, 0.3]
    subplot_titles = (
        ["Price / Renko", "Brick Size", "Trend"]
        if SHOW_TREND else ["Price / Renko", "Brick Size"]
    )

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    x = df["bar_idx"].values

    # ── Raw price ──
    fig.add_trace(go.Scatter(
        x=x, y=df["close"].values,
        mode="lines",
        line=dict(color="#888888", width=0.8),
        opacity=0.5, name="Close", showlegend=True
    ), row=1, col=1)

    # ── Renko bricks (coloured line segments) ──
    if SHOW_BRICKS:
        color_map = {1: "lime", -1: "red", 0: "gray"}
        for i in range(1, len(df)):
            trend = int(df["renko_trend"].iloc[i])
            clr   = color_map.get(trend, "gray")
            fig.add_trace(go.Scatter(
                x=[x[i-1], x[i]],
                y=[df["renko_close"].iloc[i-1], df["renko_close"].iloc[i]],
                mode="lines",
                line=dict(color=clr, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ), row=1, col=1)

        # Scatter dots on renko line
        dot_colors = df["renko_trend"].map(color_map).fillna("gray").tolist()
        fig.add_trace(go.Scatter(
            x=x, y=df["renko_close"].values,
            mode="markers",
            marker=dict(color=dot_colors, size=5),
            showlegend=False, name="Renko",
        ), row=1, col=1)

    # ── Alerts ──
    up_df   = df[df["alert_up"]]
    down_df = df[df["alert_down"]]

    if not up_df.empty:
        fig.add_trace(go.Scatter(
            x=up_df["bar_idx"].values,
            y=up_df["renko_close"].values,
            mode="markers+text",
            marker=dict(symbol="triangle-up", color="lime", size=12),
            text=[t.strftime("%H:%M") for t in up_df.index],
            textposition="top right",
            textfont=dict(color="white", size=9),
            name="Up Alert",
        ), row=1, col=1)

    if not down_df.empty:
        fig.add_trace(go.Scatter(
            x=down_df["bar_idx"].values,
            y=down_df["renko_close"].values,
            mode="markers+text",
            marker=dict(symbol="triangle-down", color="red", size=12),
            text=[t.strftime("%H:%M") for t in down_df.index],
            textposition="bottom right",
            textfont=dict(color="white", size=9),
            name="Down Alert",
        ), row=1, col=1)

    # ── Brick size ──
    fig.add_trace(go.Scatter(
        x=x, y=df["brick_size"].values,
        mode="lines",
        line=dict(color="#f0a500", width=1.2),
        name="Brick Size",
    ), row=2, col=1)

    # ── Trend bars ──
    if SHOW_TREND:
        trend_colors = df["renko_trend"].map(
            {1: "lime", -1: "red", 0: "gray"}
        ).fillna("gray").tolist()
        fig.add_trace(go.Bar(
            x=x, y=df["renko_trend"].values,
            marker_color=trend_colors,
            name="Trend",
        ), row=3, col=1)

    # ── Title / window string ──
    total_offset = SESSION_START[1] + minute_offset
    cur_h = SESSION_START[0] + total_offset // 60
    cur_m = total_offset % 60
    window_str = f"09:{SESSION_START[1]:02d} → {cur_h:02d}:{cur_m:02d}"
    brick_label = f"ATR-{ATR_LEN}" if IS_ATR else f"Fixed-{TRAD_LEN}"
    title = (
        f"Renko Replay  ·  {TICKER}  ·  {brick_label}  ·  "
        f"{window_str}  ·  next bar in {max(0, seconds_left)}s"
    )

    axis_style = dict(
        gridcolor=SPINE_CLR,
        color=TEXT_CLR,
        showline=True,
        linecolor=SPINE_CLR,
        zeroline=False,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_CLR, size=13), x=0.5),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_CLR),
        legend=dict(
            bgcolor="#1e222d",
            font=dict(color=TEXT_CLR, size=9),
            orientation="v",
            x=0, y=1,
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        hovermode="x unified",
    )

    # Apply axis styling to every axis
    for axis in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        fig.update_layout(**{axis: axis_style})

    # Fix y-tick labels for trend panel
    if SHOW_TREND:
        fig.update_layout(yaxis3=dict(
            **axis_style,
            tickvals=[-1, 0, 1],
            ticktext=["DN", "—", "UP"],
        ))

    return fig


# ─── APP DATA STORE ───────────────────────────────────────────────────────────
print("Fetching data …")
_full_data = fetch_all_data()
if _full_data is not None:
    print(f"Loaded {len(_full_data)} bars")
else:
    print("⚠ No data loaded")

# ─── DASH LAYOUT ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = f"Renko – {TICKER}"

app.layout = html.Div(
    style={"backgroundColor": DARK_BG, "height": "100vh", "padding": "0"},
    children=[
        # State stores
        dcc.Store(id="store-minute-offset", data=0),
        dcc.Store(id="store-elapsed",       data=0),

        # Control bar
        html.Div(
            style={
                "display": "flex", "alignItems": "center", "gap": "12px",
                "padding": "8px 16px", "backgroundColor": "#1e222d",
            },
            children=[
                html.Button(
                    "▶ Play", id="btn-play", n_clicks=0,
                    style={
                        "backgroundColor": "#2962ff", "color": "white",
                        "border": "none", "borderRadius": "4px",
                        "padding": "6px 16px", "cursor": "pointer",
                        "fontWeight": "bold",
                    }
                ),
                html.Button(
                    "⏸ Pause", id="btn-pause", n_clicks=0,
                    style={
                        "backgroundColor": "#444", "color": "white",
                        "border": "none", "borderRadius": "4px",
                        "padding": "6px 16px", "cursor": "pointer",
                    }
                ),
                html.Button(
                    "↺ Reset", id="btn-reset", n_clicks=0,
                    style={
                        "backgroundColor": "#444", "color": "white",
                        "border": "none", "borderRadius": "4px",
                        "padding": "6px 16px", "cursor": "pointer",
                    }
                ),
                html.Span("Speed (s/bar):", style={"color": TEXT_CLR, "fontSize": "13px"}),
                dcc.Slider(
                    id="slider-speed",
                    min=5, max=120, step=5, value=REFRESH_SEC,
                    marks={5:"5s", 30:"30s", 60:"1m", 120:"2m"},
                    tooltip={"placement": "bottom"},
                    updatemode="drag",
                    className="",
                    included=False,
                ),
                dcc.Store(id="store-paused", data=False),
                dcc.Store(id="store-speed",  data=REFRESH_SEC),
            ]
        ),

        # Chart
        dcc.Graph(
            id="renko-graph",
            style={"height": "calc(100vh - 80px)"},
            config={"displayModeBar": True, "scrollZoom": True},
        ),

        # Interval tick (1 second)
        dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=False),
    ]
)


# ─── CALLBACKS ────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-paused", "data"),
    Output("interval", "disabled"),
    Input("btn-play",  "n_clicks"),
    Input("btn-pause", "n_clicks"),
    State("store-paused", "data"),
    prevent_initial_call=True,
)
def toggle_pause(play_clicks, pause_clicks, paused):
    triggered = dash.ctx.triggered_id
    if triggered == "btn-play":
        return False, False
    elif triggered == "btn-pause":
        return True, True
    return paused, paused


@app.callback(
    Output("store-minute-offset", "data", allow_duplicate=True),
    Output("store-elapsed",       "data", allow_duplicate=True),
    Input("btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset(_):
    return 0, 0


@app.callback(
    Output("store-speed", "data"),
    Input("slider-speed", "value"),
)
def update_speed(val):
    return val or REFRESH_SEC


@app.callback(
    Output("store-minute-offset", "data"),
    Output("store-elapsed",       "data"),
    Output("renko-graph",         "figure"),
    Input("interval",             "n_intervals"),
    State("store-minute-offset",  "data"),
    State("store-elapsed",        "data"),
    State("store-paused",         "data"),
    State("store-speed",          "data"),
)
def tick(n, minute_offset, elapsed, paused, speed):
    if _full_data is None:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            title=dict(text="No data available", font=dict(color=TEXT_CLR)),
        )
        return minute_offset, elapsed, fig

    if paused:
        # Just redraw current state without advancing
        h, m      = _offset_to_hm(minute_offset)
        sliced    = slice_up_to_minute(_full_data, h, m)
        df        = check_alerts(compute_renko(sliced)) if not sliced.empty else None
        seconds_left = max(0, speed - elapsed)
        fig = build_figure(df, minute_offset, seconds_left) if df is not None else go.Figure()
        return minute_offset, elapsed, fig

    elapsed += 1
    if elapsed >= speed:
        elapsed = 0
        minute_offset += 1
        # Cap at session end
        end_total = SESSION_END[0] * 60 + SESSION_END[1]
        h, m      = _offset_to_hm(minute_offset)
        if h * 60 + m > end_total:
            h, m          = SESSION_END
            minute_offset = (SESSION_END[0] * 60 + SESSION_END[1]) - (SESSION_START[0] * 60 + SESSION_START[1])

    h, m   = _offset_to_hm(minute_offset)
    sliced = slice_up_to_minute(_full_data, h, m)

    if sliced.empty:
        return minute_offset, elapsed, go.Figure()

    df           = check_alerts(compute_renko(sliced))
    seconds_left = max(0, speed - elapsed)
    fig          = build_figure(df, minute_offset, seconds_left)
    return minute_offset, elapsed, fig


def _offset_to_hm(offset):
    total = SESSION_START[1] + offset
    return SESSION_START[0] + total // 60, total % 60


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
