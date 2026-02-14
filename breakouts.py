from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BreakoutRow:
    symbol: str
    base_asset: str
    last_price: float
    high_24h: float
    low_24h: float
    high_90d: float
    low_90d: float
    high_180d: float
    low_180d: float
    broke_high_90d: bool
    broke_high_180d: bool
    broke_low_90d: bool
    broke_low_180d: bool


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def levels_from_klines(klines: list[list[Any]]) -> tuple[float, float, float, float]:
    """Compute 90d and 180d highs/lows from *closed* daily candles only."""
    if len(klines) < 181:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    closed_only = klines[:-1]
    highs = [_to_float(k[2]) for k in closed_only if len(k) > 4]
    lows = [_to_float(k[3]) for k in closed_only if len(k) > 4]

    if len(highs) < 180 or len(lows) < 180:
        return (float("nan"), float("nan"), float("nan"), float("nan"))

    high_90 = max(highs[-90:])
    low_90 = min(lows[-90:])
    high_180 = max(highs[-180:])
    low_180 = min(lows[-180:])
    return (high_90, low_90, high_180, low_180)
