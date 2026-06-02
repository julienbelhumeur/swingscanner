import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from indicators import compute_indicators


DARK_BG = "#0d0f14"
PANEL_BG = "#161923"
GRID = "#1e2230"
TEXT = "#94a3b8"
GREEN = "#4ade80"
RED = "#f87171"
BLUE = "#60a5fa"
AMBER = "#fbbf24"
PURPLE = "#a78bfa"


def build_chart(ticker: str) -> go.Figure | None:
    try:
        raw = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True, multi_level_index=False)
        if raw is None or len(raw) < 30:
            return None
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = compute_indicators(raw.dropna())
    except Exception:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=("", "RSI (14)", "MACD"),
    )

    # ── Candlestick ───────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
        name="Prix",
        line_width=1,
    ), row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], line=dict(color=BLUE, width=1.2), name="MA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], line=dict(color=AMBER, width=1.2), name="MA50"), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_upper"],
        line=dict(color=PURPLE, width=0.8, dash="dot"),
        name="BB sup", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_lower"],
        line=dict(color=PURPLE, width=0.8, dash="dot"),
        fill="tonexty", fillcolor="rgba(167,139,250,0.05)",
        name="BB inf", showlegend=False,
    ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_colors = [GREEN if float(df["Close"].iloc[i]) >= float(df["Open"].iloc[i]) else RED
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors,
        marker_opacity=0.5,
        name="Volume",
        showlegend=False,
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        line=dict(color=BLUE, width=1.5),
        name="RSI",
    ), row=2, col=1)

    for level, color in [(70, RED), (30, GREEN), (50, GRID)]:
        fig.add_hline(y=level, line=dict(color=color, width=0.8, dash="dot"), row=2, col=1)

    # ── MACD ──────────────────────────────────────────────────────────────────
    hist = df["MACD_hist"]
    hist_colors = [GREEN if v >= 0 else RED for v in hist]

    fig.add_trace(go.Bar(
        x=df.index, y=hist,
        marker_color=hist_colors,
        marker_opacity=0.7,
        name="Histogramme",
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color=BLUE, width=1.2), name="MACD"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color=AMBER, width=1.2), name="Signal"), row=3, col=1)

    # ── Style ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        height=520,
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(family="IBM Plex Mono", color=TEXT, size=11),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        margin=dict(l=8, r=8, t=32, b=8),
        title=dict(text=ticker, font=dict(size=14, color="#e2e8f0"), x=0.01),
    )

    axis_style = dict(
        gridcolor=GRID,
        gridwidth=0.5,
        zerolinecolor=GRID,
        tickfont=dict(size=10),
        showline=False,
    )

    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)

    # RSI subplot y-axis fix
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig
