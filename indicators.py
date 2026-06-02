import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # ── RSI ──────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # ── Moyennes mobiles ─────────────────────────────────────────────────────
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_mid"] = bb_mid
    bb_width = df["BB_upper"] - df["BB_lower"]
    df["BB_pct"] = (close - df["BB_lower"]) / bb_width.replace(0, np.nan)

    # ── ATR ──────────────────────────────────────────────────────────────────
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # ── Volume moyen ─────────────────────────────────────────────────────────
    df["Vol_MA20"] = volume.rolling(20).mean()

    # ── Supports / résistances (pivots simples) ───────────────────────────────
    df["Pivot"] = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
    df["R1"] = 2 * df["Pivot"] - low.shift(1)
    df["S1"] = 2 * df["Pivot"] - high.shift(1)

    return df.dropna(subset=["RSI", "MACD", "MA20", "BB_upper", "ATR"])


def score_setup(
    df: pd.DataFrame,
    indicators: dict,
) -> tuple:
    """
    Returns (score/10, direction, signal_description, entry, stop, target)
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    rsi = float(last["RSI"])
    macd_hist = float(last["MACD_hist"])
    prev_macd_hist = float(prev["MACD_hist"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    bb_pct = float(last["BB_pct"])
    atr = float(last["ATR"])
    volume = float(last["Volume"])
    vol_ma = float(last["Vol_MA20"])

    bull_points = 0
    bear_points = 0
    signals = []

    # ── RSI ──────────────────────────────────────────────────────────────────
    if indicators.get("rsi"):
        if rsi < 30:
            bull_points += 2
            signals.append("RSI survente (<30)")
        elif rsi < 40:
            bull_points += 1
            signals.append("RSI bas (30-40)")
        elif rsi > 70:
            bear_points += 2
            signals.append("RSI surachat (>70)")
        elif rsi > 60:
            bear_points += 1
            signals.append("RSI élevé (60-70)")

        # Divergence simplifiée
        price_up = close > float(df["Close"].iloc[-5])
        rsi_down = rsi < float(df["RSI"].iloc[-5])
        if price_up and rsi_down and rsi > 55:
            bear_points += 1
            signals.append("Divergence baissière RSI")

        price_down = close < float(df["Close"].iloc[-5])
        rsi_up = rsi > float(df["RSI"].iloc[-5])
        if price_down and rsi_up and rsi < 45:
            bull_points += 1
            signals.append("Divergence haussière RSI")

    # ── MACD ─────────────────────────────────────────────────────────────────
    if indicators.get("macd"):
        if prev_macd_hist < 0 and macd_hist > 0:
            bull_points += 2
            signals.append("Croisement MACD haussier")
        elif prev_macd_hist > 0 and macd_hist < 0:
            bear_points += 2
            signals.append("Croisement MACD baissier")
        elif macd_hist > 0 and macd_hist > prev_macd_hist:
            bull_points += 1
            signals.append("MACD momentum haussier")
        elif macd_hist < 0 and macd_hist < prev_macd_hist:
            bear_points += 1
            signals.append("MACD momentum baissier")

    # ── Moyennes mobiles ─────────────────────────────────────────────────────
    if indicators.get("ma"):
        if close > ma20 > ma50:
            bull_points += 1
            signals.append("Prix > MA20 > MA50")
        elif close < ma20 < ma50:
            bear_points += 1
            signals.append("Prix < MA20 < MA50")

        prev_close = float(prev["Close"])
        prev_ma20 = float(prev["MA20"])
        if prev_close < prev_ma20 and close > ma20:
            bull_points += 1
            signals.append("Croisement MA20 haussier")
        elif prev_close > prev_ma20 and close < ma20:
            bear_points += 1
            signals.append("Croisement MA20 baissier")

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if indicators.get("bb"):
        if bb_pct < 0.05:
            bull_points += 2
            signals.append("Prix sous bande BB inférieure")
        elif bb_pct < 0.2:
            bull_points += 1
            signals.append("Prix proche bande BB inférieure")
        elif bb_pct > 0.95:
            bear_points += 2
            signals.append("Prix sur bande BB supérieure")
        elif bb_pct > 0.8:
            bear_points += 1
            signals.append("Prix proche bande BB supérieure")

    # ── Volume anormal ───────────────────────────────────────────────────────
    if indicators.get("volume"):
        vol_ratio = volume / vol_ma if vol_ma > 0 else 1
        if vol_ratio > 2.0:
            if bull_points >= bear_points:
                bull_points += 1
                signals.append(f"Volume x{vol_ratio:.1f} (fort intérêt haussier)")
            else:
                bear_points += 1
                signals.append(f"Volume x{vol_ratio:.1f} (fort intérêt baissier)")
        elif vol_ratio > 1.5:
            signals.append(f"Volume x{vol_ratio:.1f} (légèrement élevé)")

    # ── Direction & score ────────────────────────────────────────────────────
    total = bull_points + bear_points
    if total == 0:
        return 0, "→ Neutre", "", None, None, None

    if bull_points > bear_points:
        direction = "↑ Hausse"
        raw_score = bull_points
    elif bear_points > bull_points:
        direction = "↓ Baisse"
        raw_score = bear_points
    else:
        direction = "→ Neutre"
        raw_score = 0

    score = min(10, int(round((raw_score / max(total, 1)) * 10 + raw_score * 0.4)))
    score = max(1, min(10, score))

    # ── Niveaux ──────────────────────────────────────────────────────────────
    if direction == "↑ Hausse":
        entry = round(close, 2)
        stop = round(close - 1.5 * atr, 2)
        target = round(close + 3.0 * atr, 2)
    elif direction == "↓ Baisse":
        entry = round(close, 2)
        stop = round(close + 1.5 * atr, 2)
        target = round(close - 3.0 * atr, 2)
    else:
        entry = stop = target = round(close, 2)

    signal_str = " · ".join(signals)
    return score, direction, signal_str, entry, stop, target
