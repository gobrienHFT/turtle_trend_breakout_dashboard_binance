from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd


def compute_levels_from_klines(klines_1d: List[List[Any]], lookbacks: Sequence[int] = (90, 180)) -> Tuple[float, float, float, float, int]:
    """
    Uses completed candles only (drops latest candle).
    Returns: high90, low90, high180, low180, n_days
    """
    if not klines_1d or len(klines_1d) < 90:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    highs = [float(k[2]) for k in klines_1d]
    lows = [float(k[3]) for k in klines_1d]

    if len(highs) >= 2:
        highs = highs[:-1]
        lows = lows[:-1]

    n_days = len(highs)
    if n_days < 90:
        return (np.nan, np.nan, np.nan, np.nan, n_days)

    w90h = highs[-90:]
    w90l = lows[-90:]
    h90 = float(max(w90h))
    l90 = float(min(w90l))

    if n_days >= 180:
        w180h = highs[-180:]
        w180l = lows[-180:]
        h180 = float(max(w180h))
        l180 = float(min(w180l))
    else:
        h180 = np.nan
        l180 = np.nan

    return (h90, l90, h180, l180, n_days)


def apply_breakout_flags(df: pd.DataFrame, buffer_pct: float = 0.0) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    b = float(buffer_pct) / 100.0

    out["break_90_high"] = out["price"] > out["high_90"] * (1.0 + b)
    out["break_90_low"] = out["price"] < out["low_90"] * (1.0 - b)
    out["break_180_high"] = out["price"] > out["high_180"] * (1.0 + b)
    out["break_180_low"] = out["price"] < out["low_180"] * (1.0 - b)

    return out


def summarize_breakouts(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {
            "symbols": 0,
            "break_90_high": 0,
            "break_90_low": 0,
            "break_180_high": 0,
            "break_180_low": 0,
        }

    return {
        "symbols": int(df["symbol"].nunique()),
        "break_90_high": int(df["break_90_high"].sum()),
        "break_90_low": int(df["break_90_low"].sum()),
        "break_180_high": int(df["break_180_high"].sum()),
        "break_180_low": int(df["break_180_low"].sum()),
    }
