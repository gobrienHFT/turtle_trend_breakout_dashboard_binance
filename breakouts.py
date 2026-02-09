#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timezone
import pandas as pd


def ms_to_utc(ms: int) -> str:
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


@dataclass(frozen=True)
class Levels:
    symbol: str
    quote_asset: str
    base_asset: str
    high_90: float
    low_90: float
    high_180: float
    low_180: float
    high_90_date: str
    low_90_date: str
    high_180_date: str
    low_180_date: str
    n_days: int


def _levels_from_klines(
    symbol: str,
    quote_asset: str,
    base_asset: str,
    klines_1d: List[List[Any]],
    lookbacks: Sequence[int] = (90, 180),
) -> Optional[Levels]:
    if not klines_1d or len(klines_1d) < 20:
        return None

    df = pd.DataFrame(
        {
            "open_time": [int(k[0]) for k in klines_1d],
            "high": [float(k[2]) for k in klines_1d],
            "low": [float(k[3]) for k in klines_1d],
            "close": [float(k[4]) for k in klines_1d],
        }
    )

    df = df.iloc[:-1].copy()
    if df.empty:
        return None

    def extreme(window: int, col: str, want_max: bool) -> Tuple[float, str]:
        if window <= 0:
            return (float("nan"), "")
        w = df.tail(window)
        if w.empty or len(w) < window:
            return (float("nan"), "")
        if want_max:
            idx = w[col].idxmax()
            v = float(w.loc[idx, col])
        else:
            idx = w[col].idxmin()
            v = float(w.loc[idx, col])
        date = ms_to_utc(int(df.loc[idx, "open_time"])) if idx in df.index else ""
        return (v, date)

    h90, h90d = extreme(int(lookbacks[0]), "high", True)
    l90, l90d = extreme(int(lookbacks[0]), "low", False)

    h180, h180d = extreme(int(lookbacks[1]), "high", True)
    l180, l180d = extreme(int(lookbacks[1]), "low", False)

    return Levels(
        symbol=symbol,
        quote_asset=quote_asset,
        base_asset=base_asset,
        high_90=h90,
        low_90=l90,
        high_180=h180,
        low_180=l180,
        high_90_date=h90d,
        low_90_date=l90d,
        high_180_date=h180d,
        low_180_date=l180d,
        n_days=int(len(df)),
    )


def compute_levels(rows: List[Tuple[str, str, str, List[List[Any]]]], lookbacks: Sequence[int] = (90, 180)) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    for sym, quote, base, kl in rows:
        lv = _levels_from_klines(sym, quote, base, kl, lookbacks=lookbacks)
        if not lv:
            continue
        out.append(
            {
                "symbol": lv.symbol,
                "base": lv.base_asset,
                "quote": lv.quote_asset,
                "high_90": lv.high_90,
                "low_90": lv.low_90,
                "high_180": lv.high_180,
                "low_180": lv.low_180,
                "high_90_date": lv.high_90_date,
                "low_90_date": lv.low_90_date,
                "high_180_date": lv.high_180_date,
                "low_180_date": lv.low_180_date,
                "n_days": lv.n_days,
            }
        )
    return pd.DataFrame(out)


def detect_breakouts(
    df_levels: pd.DataFrame,
    df_prices: pd.DataFrame,
    df_ticker24: Optional[pd.DataFrame] = None,
    *,
    buffer_pct: float = 0.0,
) -> pd.DataFrame:
    if df_levels is None or df_levels.empty:
        return pd.DataFrame()
    if df_prices is None or df_prices.empty:
        return pd.DataFrame()

    out = df_levels.merge(df_prices, on="symbol", how="left")

    if df_ticker24 is not None and not df_ticker24.empty:
        out = out.merge(df_ticker24, on="symbol", how="left")

    out["price"] = pd.to_numeric(out.get("price"), errors="coerce")
    for c in ("high_90", "low_90", "high_180", "low_180"):
        out[c] = pd.to_numeric(out.get(c), errors="coerce")

    buf = float(buffer_pct) / 100.0
    out["break_high_90"] = out["price"] > out["high_90"] * (1.0 + buf)
    out["break_low_90"] = out["price"] < out["low_90"] * (1.0 - buf)
    out["break_high_180"] = out["price"] > out["high_180"] * (1.0 + buf)
    out["break_low_180"] = out["price"] < out["low_180"] * (1.0 - buf)

    out["dist_high_90_pct"] = (out["price"] / out["high_90"] - 1.0) * 100.0
    out["dist_low_90_pct"] = (1.0 - out["price"] / out["low_90"]) * 100.0
    out["dist_high_180_pct"] = (out["price"] / out["high_180"] - 1.0) * 100.0
    out["dist_low_180_pct"] = (1.0 - out["price"] / out["low_180"]) * 100.0

    out.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
    return out
