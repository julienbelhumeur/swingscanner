import yfinance as yf
import pandas as pd
import numpy as np
from indicators import compute_indicators, score_setup

# ── WATCHLISTS ────────────────────────────────────────────────────────────────
WATCHLISTS = {
    "TSX": [
        "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO",
        "CNR.TO", "CP.TO", "ENB.TO", "TRP.TO", "SU.TO",
        "ABX.TO", "AEM.TO", "WPM.TO", "K.TO", "FNV.TO",
        "CNQ.TO", "CVE.TO", "MEG.TO", "VET.TO", "ARX.TO",
        "SHOP.TO", "CSU.TO", "MDA.TO", "DSGX.TO", "LSPD.TO",
        "ATD.TO", "MRU.TO", "L.TO", "DOL.TO", "EMP-A.TO",
        "BAM.TO", "BPY-UN.TO", "FSV.TO", "CAR-UN.TO", "REI-UN.TO",
        "BCE.TO", "T.TO", "RCI-B.TO",
        "CCO.TO", "LUN.TO", "IVN.TO", "FM.TO",
    ],
    "NYSE": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "BAC", "WFC", "GS", "MS",
        "JNJ", "UNH", "PFE", "MRK", "ABBV",
        "XOM", "CVX", "COP", "SLB",
        "HD", "LOW", "TGT", "WMT", "COST",
        "BA", "CAT", "MMM", "GE", "HON",
        "NEE", "DUK", "SO", "D",
        "AMT", "PLD", "SPG", "O",
        "V", "MA", "PYPL", "AXP",
    ],
    "ETF_CA": [
        "XIU.TO", "XEI.TO", "XIC.TO", "XFN.TO", "XEG.TO",
        "ZEB.TO", "ZRE.TO", "ZDV.TO", "XDIV.TO", "VCN.TO",
        "HXT.TO", "HXQ.TO", "HMAX.TO", "ETHY.TO",
        "XBB.TO", "ZAG.TO", "VAB.TO",
    ],
    "ETF_US": [
        "SPY", "QQQ", "IWM", "DIA", "VTI",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP",
        "GLD", "SLV", "GDX", "GDXJ",
        "TLT", "IEF", "HYG", "LQD",
        "EEM", "EFA", "VEA",
        "ARKK", "ARKG", "ARKW",
    ],
}


def fetch_ticker_data(ticker: str, period: str = "6mo") -> pd.DataFrame | None:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if df is None or len(df) < 50:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
        return df
    except Exception:
        return None


def get_ticker_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "avg_volume": info.get("averageVolume") or 0,
        }
    except Exception:
        return {"name": ticker, "avg_volume": 0}


def run_scanner(
    markets: list,
    min_score: int = 6,
    min_volume: int = 0,
    direction: str = "Toutes",
    indicators: dict = None,
) -> pd.DataFrame:

    if indicators is None:
        indicators = {"rsi": True, "macd": True, "ma": True, "bb": True, "volume": True}

    tickers_to_scan = []
    for market in markets:
        tickers_to_scan.extend(WATCHLISTS.get(market, []))

    results = []

    for ticker in tickers_to_scan:
        df = fetch_ticker_data(ticker)
        if df is None or df.empty:
            continue

        df = compute_indicators(df)
        if df is None or df.empty:
            continue

        score, direction_signal, signals, entry, stop, target = score_setup(df, indicators)

        if score < min_score:
            continue

        if direction == "Haussières seulement" and direction_signal != "↑ Hausse":
            continue
        if direction == "Baissières seulement" and direction_signal != "↓ Baisse":
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2]

        current_price = float(last["Close"])
        prev_price = float(prev["Close"])
        variation = ((current_price - prev_price) / prev_price) * 100

        avg_vol = float(df["Volume"].rolling(20).mean().iloc[-1])
        curr_vol = float(last["Volume"])
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0

        if min_volume > 0 and curr_vol < min_volume:
            continue

        info = get_ticker_info(ticker)

        rr = 0.0
        if entry and stop and target and abs(entry - stop) > 0:
            rr = abs(target - entry) / abs(entry - stop)

        results.append({
            "Ticker": ticker,
            "Nom": info["name"][:30],
            "Direction": direction_signal,
            "Score": score,
            "Prix": current_price,
            "Variation %": variation,
            "RSI": float(last.get("RSI", 50)),
            "Signal MACD": "Haussier" if float(last.get("MACD_hist", 0)) > 0 else "Baissier",
            "Volume ratio": vol_ratio,
            "Entrée suggérée": entry or current_price,
            "Stop-loss": stop or round(current_price * 0.97, 2),
            "Cible": target or round(current_price * 1.06, 2),
            "Ratio R/R": round(rr, 2),
            "Signaux": signals,
        })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("Score", ascending=False).reset_index(drop=True)
    return result_df
