import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Scanner de Swing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

from scanner import run_scanner, WATCHLISTS
from indicators import compute_indicators, score_setup
from charts import build_chart

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp { background: #0d0f14; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2230;
}

.metric-card {
    background: #161923;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.metric-label {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #f1f5f9;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 2px;
}

.signal-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}
.signal-bull { background: #0d2e1a; color: #4ade80; border: 1px solid #166534; }
.signal-bear { background: #2e0d0d; color: #f87171; border: 1px solid #991b1b; }
.signal-neut { background: #1a1a2e; color: #94a3b8; border: 1px solid #1e2230; }

.section-header {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e2230;
}

.score-bar-bg {
    background: #1e2230;
    border-radius: 3px;
    height: 6px;
    width: 100%;
    margin-top: 6px;
}
.score-bar-fill {
    height: 6px;
    border-radius: 3px;
    transition: width 0.4s ease;
}

div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

.stButton button {
    background: #1e3a5f;
    color: #93c5fd;
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 8px 20px;
    transition: all 0.15s;
}
.stButton button:hover {
    background: #1d4ed8;
    color: #ffffff;
}

.stSelectbox label, .stMultiSelect label, .stSlider label {
    color: #64748b !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.06em;
}

hr { border-color: #1e2230; }

.ticker-chip {
    display: inline-block;
    background: #161923;
    border: 1px solid #1e2230;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    color: #94a3b8;
    margin: 2px;
}

.setup-row {
    background: #161923;
    border: 1px solid #1e2230;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Scanner de Swing")
    st.markdown("<div class='section-header'>Marchés</div>", unsafe_allow_html=True)

    markets = st.multiselect(
        "Sélection",
        ["TSX", "NYSE/NASDAQ", "ETFs CA", "ETFs US"],
        default=["TSX", "NYSE/NASDAQ", "ETFs CA", "ETFs US"],
        label_visibility="collapsed"
    )

    st.markdown("<div class='section-header'>Filtres techniques</div>", unsafe_allow_html=True)

    min_score = st.slider("Score minimum", 1, 10, 6)
    min_volume = st.selectbox("Volume minimum", ["Aucun", "100K", "500K", "1M", "5M"], index=2)
    direction = st.radio("Direction", ["Toutes", "Haussières seulement", "Baissières seulement"], index=0)

    st.markdown("<div class='section-header'>Indicateurs actifs</div>", unsafe_allow_html=True)
    use_rsi = st.checkbox("RSI", value=True)
    use_macd = st.checkbox("MACD", value=True)
    use_ma = st.checkbox("Moyennes mobiles", value=True)
    use_bb = st.checkbox("Bollinger Bands", value=True)
    use_vol = st.checkbox("Volume anormal", value=True)

    st.markdown("---")
    scan_btn = st.button("🔍  Lancer le scan", use_container_width=True)
    st.markdown(f"<div style='font-size:11px; color:#475569; font-family:IBM Plex Mono; margin-top:8px;'>Dernière mise à jour<br>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)

# ── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# Scanner de Swing")
st.markdown("<div style='color:#64748b; font-size:14px; margin-bottom:24px;'>Détection automatique de setups techniques — TSX · NYSE/NASDAQ · ETFs</div>", unsafe_allow_html=True)

indicators_config = {
    "rsi": use_rsi,
    "macd": use_macd,
    "ma": use_ma,
    "bb": use_bb,
    "volume": use_vol,
}

vol_map = {"Aucun": 0, "100K": 100_000, "500K": 500_000, "1M": 1_000_000, "5M": 5_000_000}

if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

# ── SCAN ─────────────────────────────────────────────────────────────────────
if scan_btn:
    active_markets = []
    if "TSX" in markets: active_markets.append("TSX")
    if "NYSE/NASDAQ" in markets: active_markets.append("NYSE")
    if "ETFs CA" in markets: active_markets.append("ETF_CA")
    if "ETFs US" in markets: active_markets.append("ETF_US")

    with st.spinner("Scan en cours…"):
        df = run_scanner(
            markets=active_markets,
            min_score=min_score,
            min_volume=vol_map[min_volume],
            direction=direction,
            indicators=indicators_config,
        )
    st.session_state.results_df = df
    st.session_state.selected_ticker = None

# ── RÉSULTATS ────────────────────────────────────────────────────────────────
df = st.session_state.results_df

if df is None:
    st.markdown("""
    <div style='background:#161923; border:1px solid #1e2230; border-radius:12px; padding:48px; text-align:center; color:#475569;'>
        <div style='font-size:40px; margin-bottom:12px;'>🔍</div>
        <div style='font-size:15px; margin-bottom:6px; color:#64748b;'>Aucun scan lancé</div>
        <div style='font-size:13px;'>Configure les filtres dans la barre latérale et clique sur <strong style='color:#93c5fd;'>Lancer le scan</strong></div>
    </div>
    """, unsafe_allow_html=True)

elif df.empty:
    st.warning("Aucun setup trouvé avec ces critères. Essaie de baisser le score minimum.")

else:
    # Métriques summary
    bull = df[df["Direction"] == "↑ Hausse"]
    bear = df[df["Direction"] == "↓ Baisse"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Setups trouvés</div>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-sub'>sur {sum(len(v) for v in WATCHLISTS.values())} tickers scannés</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Haussiers</div>
            <div class='metric-value' style='color:#4ade80;'>{len(bull)}</div>
            <div class='metric-sub'>signaux d'achat potentiels</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Baissiers</div>
            <div class='metric-value' style='color:#f87171;'>{len(bear)}</div>
            <div class='metric-sub'>signaux de vente / short</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        avg_score = df["Score"].mean()
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Score moyen</div>
            <div class='metric-value'>{avg_score:.1f}<span style='font-size:16px; color:#64748b;'>/10</span></div>
            <div class='metric-sub'>qualité des setups</div>
        </div>""", unsafe_allow_html=True)

    # Table des setups
    st.markdown("<div class='section-header'>Setups détectés</div>", unsafe_allow_html=True)

    display_df = df[["Ticker", "Nom", "Direction", "Score", "Prix", "Variation %", "RSI", "Signal MACD", "Volume ratio", "Entrée suggérée", "Stop-loss", "Cible"]].copy()

    def color_direction(val):
        if "Hausse" in str(val):
            return "color: #4ade80"
        elif "Baisse" in str(val):
            return "color: #f87171"
        return "color: #94a3b8"

    def color_score(val):
        if val >= 8: return "color: #4ade80; font-weight: 600"
        if val >= 6: return "color: #fbbf24"
        return "color: #94a3b8"

    def color_rsi(val):
        try:
            v = float(val)
            if v <= 35: return "color: #4ade80"
            if v >= 65: return "color: #f87171"
        except: pass
        return "color: #94a3b8"

    styled = display_df.style\
        .map(color_direction, subset=["Direction"])\
        .map(color_score, subset=["Score"])\
        .map(color_rsi, subset=["RSI"])\
        .format({
            "Prix": "{:.2f}",
            "Variation %": "{:+.2f}%",
            "RSI": "{:.1f}",
            "Volume ratio": "{:.1f}x",
            "Entrée suggérée": "{:.2f}",
            "Stop-loss": "{:.2f}",
            "Cible": "{:.2f}",
        })\
        .set_properties(**{
            "background-color": "#161923",
            "color": "#e2e8f0",
            "border": "1px solid #1e2230",
            "font-family": "IBM Plex Mono, monospace",
            "font-size": "13px",
        })

    st.dataframe(styled, use_container_width=True, height=400)

    # Détail d'un ticker
    st.markdown("<div class='section-header'>Analyse détaillée</div>", unsafe_allow_html=True)

    ticker_options = df["Ticker"].tolist()
    selected = st.selectbox("Choisir un ticker pour l'analyse", ticker_options, label_visibility="visible")

    if selected:
        row = df[df["Ticker"] == selected].iloc[0]

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig = build_chart(selected)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col_right:
            dir_class = "signal-bull" if "Hausse" in row["Direction"] else "signal-bear"
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:12px;'>
                <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>
                    <span style='font-size:20px; font-weight:600; font-family:IBM Plex Mono;'>{selected}</span>
                    <span class='signal-badge {dir_class}'>{row["Direction"]}</span>
                </div>
                <div style='font-size:13px; color:#64748b; margin-bottom:16px;'>{row["Nom"]}</div>

                <div style='display:grid; grid-template-columns:1fr 1fr; gap:12px;'>
                    <div>
                        <div class='metric-label'>Prix actuel</div>
                        <div style='font-size:20px; font-weight:600; font-family:IBM Plex Mono;'>{row["Prix"]:.2f}</div>
                    </div>
                    <div>
                        <div class='metric-label'>Score setup</div>
                        <div style='font-size:20px; font-weight:600; font-family:IBM Plex Mono; color:{"#4ade80" if row["Score"]>=8 else "#fbbf24"};'>{row["Score"]}/10</div>
                        <div class='score-bar-bg'><div class='score-bar-fill' style='width:{row["Score"]*10}%; background:{"#4ade80" if row["Score"]>=8 else "#fbbf24"};'></div></div>
                    </div>
                </div>
            </div>

            <div class='metric-card' style='margin-bottom:12px;'>
                <div class='metric-label' style='margin-bottom:10px;'>Niveaux clés</div>
                <div style='display:grid; gap:8px; font-family:IBM Plex Mono; font-size:13px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Entrée suggérée</span>
                        <span style='color:#93c5fd;'>{row["Entrée suggérée"]:.2f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Stop-loss</span>
                        <span style='color:#f87171;'>{row["Stop-loss"]:.2f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Cible (TP)</span>
                        <span style='color:#4ade80;'>{row["Cible"]:.2f}</span>
                    </div>
                    <div style='border-top:1px solid #1e2230; padding-top:8px; display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Ratio R/R</span>
                        <span style='color:#fbbf24; font-weight:500;'>{row["Ratio R/R"]:.1f}:1</span>
                    </div>
                </div>
            </div>

            <div class='metric-card'>
                <div class='metric-label' style='margin-bottom:10px;'>Indicateurs</div>
                <div style='display:grid; gap:8px; font-family:IBM Plex Mono; font-size:13px;'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>RSI (14)</span>
                        <span style='color:{"#4ade80" if float(row["RSI"])<35 else "#f87171" if float(row["RSI"])>65 else "#e2e8f0"};'>{float(row["RSI"]):.1f}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>MACD</span>
                        <span style='color:#e2e8f0;'>{row["Signal MACD"]}</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Volume ratio</span>
                        <span style='color:{"#fbbf24" if row["Volume ratio"]>1.5 else "#e2e8f0"};'>{row["Volume ratio"]:.1f}x</span>
                    </div>
                    <div style='display:flex; justify-content:space-between;'>
                        <span style='color:#64748b;'>Variation</span>
                        <span style='color:{"#4ade80" if row["Variation %"]>0 else "#f87171"};'>{row["Variation %"]:+.2f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if row.get("Signaux", ""):
                st.markdown(f"""
                <div class='metric-card' style='margin-top:12px;'>
                    <div class='metric-label' style='margin-bottom:8px;'>Signaux déclenchés</div>
                    <div style='font-size:12px; color:#94a3b8; line-height:1.8;'>{row["Signaux"]}</div>
                </div>
                """, unsafe_allow_html=True)

    # Export
    st.markdown("<div class='section-header'>Export</div>", unsafe_allow_html=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇  Télécharger CSV",
        data=csv,
        file_name=f"swing_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
