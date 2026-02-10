from __future__ import annotations

import os
import re
import time
import threading
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "10"))
REQUESTS_PER_SEC = float(os.environ.get("REQUESTS_PER_SEC", "2.0"))
LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
LEVELS_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "30"))
BREAKOUT_BUFFER_PCT_ENV = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))
MIN_QUOTE_VOL_24H_ENV = float(os.environ.get("MIN_QUOTE_VOL_24H", "20000000"))
QUOTE_ASSETS_ENV = os.environ.get("QUOTE_ASSETS", "USDT").strip()
BINANCE_FAPI_BASE = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")

CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_SNAPSHOT_TTL_SEC = int(os.environ.get("CACHE_SNAPSHOT_TTL_SEC", "30"))
CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))

_HTTP_REQ_LOCK = threading.Lock()
_LAST_HTTP_REQ_AT = 0.0

st.set_page_config(page_title="Binance Perp 90D/180D Breakouts", layout="wide")
st.title("Binance Perpetual Futures — 90D / 180D High-Low Breakouts")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def extract_ban_ms(error_text: str) -> Optional[int]:
    m = re.search(r"banned until\s+(\d{10,16})", str(error_text))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


class BinanceHTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:400]}")
        self.status = int(status)
        self.url = url
        self.body = body[:2000]


def paced_get(url: str, params: Optional[dict], timeout: int, headers: dict) -> requests.Response:
    global _LAST_HTTP_REQ_AT
    if REQUESTS_PER_SEC > 0:
        with _HTTP_REQ_LOCK:
            min_interval = 1.0 / REQUESTS_PER_SEC
            now = time.monotonic()
            wait = min_interval - (now - _LAST_HTTP_REQ_AT)
            if wait > 0:
                time.sleep(wait)
            _LAST_HTTP_REQ_AT = time.monotonic()
    return requests.get(url, params=params or {}, timeout=timeout, headers=headers)


def fapi_get(path: str, params: Optional[dict] = None, timeout: Optional[int] = None, retries: int = 2):
    timeout = int(timeout or HTTP_TIMEOUT)
    url = f"{BINANCE_FAPI_BASE}{path}"
    headers = {
        "User-Agent": "BreakoutScanner/2.0 (+streamlit)",
        "Accept": "application/json",
    }

    backoff = 1.0
    last_exc: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        try:
            r = paced_get(url, params, timeout, headers)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429, 500, 502, 503, 504):
                last_exc = BinanceHTTPError(r.status_code, url, r.text)
                time.sleep(backoff)
                backoff = min(8.0, backoff * 1.8)
                continue
            raise BinanceHTTPError(r.status_code, url, r.text)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(8.0, backoff * 1.8)

    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request failure")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=CACHE_SYMBOLS_TTL_SEC, show_spinner=False)
def get_perp_symbols() -> pd.DataFrame:
    info = fapi_get("/fapi/v1/exchangeInfo", retries=1)
    rows = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if not sym:
            continue
        rows.append(
            {
                "symbol": sym,
                "baseAsset": s.get("baseAsset", ""),
                "quoteAsset": s.get("quoteAsset", ""),
            }
        )
    df = pd.DataFrame(rows, columns=["symbol", "baseAsset", "quoteAsset"])
    return df.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True) if not df.empty else df


@st.cache_data(ttl=CACHE_SNAPSHOT_TTL_SEC, show_spinner=False)
def get_snapshot_last_24h() -> pd.DataFrame:
    data = fapi_get("/fapi/v1/ticker/24hr", retries=1)
    rows = []
    for x in data:
        sym = x.get("symbol")
        if not sym:
            continue
        rows.append(
            {
                "symbol": sym,
                "price": safe_float(x.get("lastPrice")),
                "quoteVolume": safe_float(x.get("quoteVolume")),
                "priceChangePercent": safe_float(x.get("priceChangePercent")),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", "price", "quoteVolume", "priceChangePercent"])


def fetch_klines_1d(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        data = fapi_get(
            "/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1d", "limit": int(limit)},
            timeout=min(HTTP_TIMEOUT, 8),
            retries=1,
        )
        if not isinstance(data, list) or len(data) < 10:
            return None
        df = pd.DataFrame(data, columns=[
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "quoteVolume", "numTrades", "takerBaseVol", "takerQuoteVol", "ignore",
        ])
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        return df[["high", "low"]]
    except Exception:
        return None


@st.cache_data(ttl=CACHE_LEVELS_TTL_SEC, show_spinner=False)
def compute_levels(symbols: list[str], fetch_days: int) -> pd.DataFrame:
    out = []
    for sym in symbols:
        d = fetch_klines_1d(sym, max(182, int(fetch_days)))
        if d is None or d.empty:
            continue
        d = d.iloc[:-1].copy() if len(d) >= 2 else d.copy()
        if len(d) < 90:
            continue

        high_90 = float(d.iloc[-90:]["high"].max())
        low_90 = float(d.iloc[-90:]["low"].min())
        high_180 = float(d.iloc[-180:]["high"].max()) if len(d) >= 180 else np.nan
        low_180 = float(d.iloc[-180:]["low"].min()) if len(d) >= 180 else np.nan

        out.append(
            {
                "symbol": sym,
                "high_90": high_90,
                "low_90": low_90,
                "high_180": high_180,
                "low_180": low_180,
                "hist_days": int(len(d)),
            }
        )
    df = pd.DataFrame(out)
    return df.sort_values("symbol").reset_index(drop=True) if not df.empty else df


def apply_flags(df: pd.DataFrame, buffer_pct: float) -> pd.DataFrame:
    out = df.copy()
    b = float(buffer_pct) / 100.0
    out["thr_90_high"] = out["high_90"] * (1 + b)
    out["thr_90_low"] = out["low_90"] * (1 - b)
    out["thr_180_high"] = out["high_180"] * (1 + b)
    out["thr_180_low"] = out["low_180"] * (1 - b)

    out["break_90_high"] = (out["price"] > out["thr_90_high"]) & np.isfinite(out["thr_90_high"])
    out["break_90_low"] = (out["price"] < out["thr_90_low"]) & np.isfinite(out["thr_90_low"])
    out["break_180_high"] = (out["price"] > out["thr_180_high"]) & np.isfinite(out["thr_180_high"])
    out["break_180_low"] = (out["price"] < out["thr_180_low"]) & np.isfinite(out["thr_180_low"])
    return out


with st.sidebar:
    st.markdown("## Scan settings")
    quote_assets_default = [q.strip().upper() for q in QUOTE_ASSETS_ENV.split(",") if q.strip()]
    quote_assets_default = quote_assets_default or ["USDT"]

    quote_assets_input = st.text_input("Quote assets (comma-separated)", value=",".join(quote_assets_default))
    min_qv = st.number_input("Min 24h quote volume (USDT)", min_value=0.0, value=float(MIN_QUOTE_VOL_24H_ENV), step=5_000_000.0)
    max_symbols = st.slider("Max symbols to scan", min_value=10, max_value=100, value=int(LEVELS_MAX_SYMBOLS), step=5)
    buffer_pct = st.number_input("Breakout buffer (%)", min_value=0.0, max_value=5.0, value=float(BREAKOUT_BUFFER_PCT_ENV), step=0.01, format="%.2f")
    show_only_breakouts = st.checkbox("Show only breakouts", value=True)

    run_scan = st.button("Run scan", type="primary")

# Run-once semantics: only scan when button clicked, or when no prior results exist.
if run_scan or "scan_df" not in st.session_state:
    try:
        syms = get_perp_symbols()
    except Exception as e:
        ban_ms = extract_ban_ms(str(e))
        if ban_ms:
            now_ms = int(time.time() * 1000)
            rem_sec = max(0, int((ban_ms - now_ms) / 1000))
            st.error(
                "Binance IP temporary ban detected. "
                f"Banned until {to_utc_str(ban_ms)} UTC (~{rem_sec // 60}m {rem_sec % 60}s remaining).\n\n"
                f"Raw error: {e}"
            )
        else:
            st.error(f"Failed to load exchangeInfo. Error: {e}")
        st.stop()

    qset = {q.strip().upper() for q in quote_assets_input.split(",") if q.strip()}
    syms = syms[syms["quoteAsset"].isin(qset)].copy()

    try:
        snap = get_snapshot_last_24h()
    except Exception as e:
        ban_ms = extract_ban_ms(str(e))
        if ban_ms:
            now_ms = int(time.time() * 1000)
            rem_sec = max(0, int((ban_ms - now_ms) / 1000))
            st.error(
                "Binance IP temporary ban detected. "
                f"Banned until {to_utc_str(ban_ms)} UTC (~{rem_sec // 60}m {rem_sec % 60}s remaining).\n\n"
                f"Raw error: {e}"
            )
        else:
            st.error(f"Failed to load ticker/24hr snapshot. Error: {e}")
        st.stop()

    snap = snap[snap["symbol"].isin(syms["symbol"])].copy()
    if min_qv > 0:
        snap = snap[snap["quoteVolume"].fillna(0) >= float(min_qv)].copy()

    snap = snap.sort_values(["quoteVolume", "symbol"], ascending=[False, True]).head(int(max_symbols)).copy()
    symbols = snap["symbol"].tolist()

    if not symbols:
        st.warning("No symbols matched filters.")
        st.stop()

    levels = compute_levels(symbols, fetch_days=LEVELS_FETCH_DAYS)
    if levels.empty:
        st.warning("No level data could be computed for selected symbols.")
        st.stop()

    df = snap.merge(levels, on="symbol", how="left")
    df = df.merge(syms[["symbol", "baseAsset", "quoteAsset"]], on="symbol", how="left")
    df = apply_flags(df, buffer_pct)

    st.session_state["scan_df"] = df
    st.session_state["scan_at"] = utc_now().strftime("%Y-%m-%d %H:%M:%S")

if "scan_df" not in st.session_state:
    st.info("Click 'Run scan' to load breakout opportunities.")
    st.stop()

out = st.session_state["scan_df"].copy()
if show_only_breakouts:
    out = out[out["break_90_high"] | out["break_90_low"] | out["break_180_high"] | out["break_180_low"]].copy()

n_symbols = int(out["symbol"].nunique()) if not out.empty else 0
n90h = int(out["break_90_high"].sum()) if not out.empty else 0
n90l = int(out["break_90_low"].sum()) if not out.empty else 0
n180h = int(out["break_180_high"].sum()) if not out.empty else 0
n180l = int(out["break_180_low"].sum()) if not out.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Symbols", n_symbols)
c2.metric("Break > 90D High", n90h)
c3.metric("Break < 90D Low", n90l)
c4.metric("Break > 180D High", n180h)
c5.metric("Break < 180D Low", n180l)
st.caption(f"Scan time (UTC): {st.session_state.get('scan_at', '')}")

show_cols = [
    "symbol", "baseAsset", "quoteAsset", "price", "quoteVolume", "priceChangePercent",
    "high_90", "low_90", "high_180", "low_180", "break_90_high", "break_90_low", "break_180_high", "break_180_low",
]
show_cols = [c for c in show_cols if c in out.columns]

if out.empty:
    st.info("No breakout opportunities matched current filters.")
else:
    st.dataframe(out[show_cols].sort_values(["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"], ascending=[False, False, False, False, True]), use_container_width=True)
