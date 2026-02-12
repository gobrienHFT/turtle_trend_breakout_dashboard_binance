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

# ---- config ----
BINANCE_FAPI_BASE = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "10"))
REQUESTS_PER_SEC = float(os.environ.get("REQUESTS_PER_SEC", "2.0"))

CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_SNAPSHOT_TTL_SEC = int(os.environ.get("CACHE_SNAPSHOT_TTL_SEC", "30"))
CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))

LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
DEFAULT_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "25"))
DEFAULT_MIN_QV = float(os.environ.get("MIN_QUOTE_VOL_24H", "20000000"))
DEFAULT_QUOTES = os.environ.get("QUOTE_ASSETS", "USDT")
DEFAULT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))

_HTTP_REQ_LOCK = threading.Lock()
_LAST_HTTP_REQ_AT = 0.0

APP_BUILD = "manual-scan-2026-02-12"
st.set_page_config(page_title="Binance 90D/180D Breakouts", layout="wide")
st.title("Binance Perpetual Futures — 90D / 180D High-Low Breakouts")
st.caption(f"Build: {APP_BUILD} | Manual one-shot scan only. No auto-refresh loop.")


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


def fapi_get(path: str, params: Optional[dict] = None, retries: int = 1):
    url = f"{BINANCE_FAPI_BASE}{path}"
    headers = {"User-Agent": "SimpleBreakoutScanner/1.0", "Accept": "application/json"}

    backoff = 1.0
    last_exc: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        try:
            r = paced_get(url, params, HTTP_TIMEOUT, headers)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429, 500, 502, 503, 504):
                last_exc = BinanceHTTPError(r.status_code, url, r.text)
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)
                continue
            raise BinanceHTTPError(r.status_code, url, r.text)
        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(8.0, backoff * 2.0)

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
        if s.get("contractType") != "PERPETUAL" or s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if sym:
            rows.append({"symbol": sym, "baseAsset": s.get("baseAsset", ""), "quoteAsset": s.get("quoteAsset", "")})
    df = pd.DataFrame(rows, columns=["symbol", "baseAsset", "quoteAsset"])
    return df.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True) if not df.empty else df


@st.cache_data(ttl=CACHE_SNAPSHOT_TTL_SEC, show_spinner=False)
def get_snapshot_24h() -> pd.DataFrame:
    data = fapi_get("/fapi/v1/ticker/24hr", retries=1)
    rows = []
    for x in data:
        sym = x.get("symbol")
        if sym:
            rows.append({
                "symbol": sym,
                "price": safe_float(x.get("lastPrice")),
                "quoteVolume": safe_float(x.get("quoteVolume")),
                "priceChangePercent": safe_float(x.get("priceChangePercent")),
            })
    return pd.DataFrame(rows, columns=["symbol", "price", "quoteVolume", "priceChangePercent"])


def fetch_klines_1d(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        data = fapi_get("/fapi/v1/klines", params={"symbol": symbol, "interval": "1d", "limit": int(limit)}, retries=1)
        if not isinstance(data, list) or len(data) < 90:
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
def compute_levels(symbols: tuple[str, ...], fetch_days: int) -> pd.DataFrame:
    out = []
    for sym in symbols:
        d = fetch_klines_1d(sym, max(182, int(fetch_days)))
        if d is None or d.empty:
            continue

        d = d.iloc[:-1].copy() if len(d) >= 2 else d.copy()
        if len(d) < 90:
            continue

        out.append({
            "symbol": sym,
            "high_90": float(d.iloc[-90:]["high"].max()),
            "low_90": float(d.iloc[-90:]["low"].min()),
            "high_180": float(d.iloc[-180:]["high"].max()) if len(d) >= 180 else np.nan,
            "low_180": float(d.iloc[-180:]["low"].min()) if len(d) >= 180 else np.nan,
        })

    df = pd.DataFrame(out)
    return df.sort_values("symbol").reset_index(drop=True) if not df.empty else df


def apply_flags(df: pd.DataFrame, buffer_pct: float) -> pd.DataFrame:
    out = df.copy()
    b = float(buffer_pct) / 100.0
    out["break_90_high"] = out["price"] > out["high_90"] * (1 + b)
    out["break_90_low"] = out["price"] < out["low_90"] * (1 - b)
    out["break_180_high"] = out["price"] > out["high_180"] * (1 + b)
    out["break_180_low"] = out["price"] < out["low_180"] * (1 - b)
    return out


with st.sidebar:
    st.markdown("## Scan settings")
    st.caption(f"Build: {APP_BUILD}")
    clear_prev = st.button("Clear previous results")
    if clear_prev:
        st.session_state.pop("scan_df", None)
        st.session_state.pop("scan_at", None)
    with st.form("scan_form"):
        quote_assets = st.text_input("Quote assets (comma-separated)", value=DEFAULT_QUOTES)
        min_qv = st.number_input("Min 24h quote volume (USDT)", min_value=0.0, value=DEFAULT_MIN_QV, step=5_000_000.0)
        max_symbols = st.slider("Max symbols to scan", min_value=10, max_value=100, value=DEFAULT_MAX_SYMBOLS, step=5)
        buffer_pct = st.number_input("Breakout buffer (%)", min_value=0.0, max_value=5.0, value=DEFAULT_BUFFER_PCT, step=0.01, format="%.2f")
        show_only_breakouts = st.checkbox("Show only breakouts", value=True)
        run_scan = st.form_submit_button("Run scan", type="primary")

if run_scan:
    try:
        syms = get_perp_symbols()
        snap = get_snapshot_24h()
    except Exception as e:
        ban_ms = extract_ban_ms(str(e))
        if ban_ms:
            now_ms = int(time.time() * 1000)
            rem = max(0, int((ban_ms - now_ms) / 1000))
            st.error(
                "Binance IP temporary ban detected. "
                f"Banned until {to_utc_str(ban_ms)} UTC (~{rem // 60}m {rem % 60}s remaining).\n\n{e}"
            )
        else:
            st.error(f"Scan failed: {e}")
        st.stop()

    qset = {q.strip().upper() for q in quote_assets.split(",") if q.strip()}
    syms = syms[syms["quoteAsset"].isin(qset)].copy()

    snap = snap[snap["symbol"].isin(syms["symbol"])].copy()
    snap = snap[snap["quoteVolume"].fillna(0) >= float(min_qv)].copy()
    snap = snap.sort_values(["quoteVolume", "symbol"], ascending=[False, True]).head(int(max_symbols)).copy()

    symbols = tuple(snap["symbol"].tolist())
    levels = compute_levels(symbols, fetch_days=LEVELS_FETCH_DAYS)

    if levels.empty:
        st.warning("No symbols had enough 1D history for 90D levels with current filters.")
        st.stop()

    result = snap.merge(levels, on="symbol", how="inner").merge(syms[["symbol", "baseAsset", "quoteAsset"]], on="symbol", how="left")
    result = apply_flags(result, buffer_pct=buffer_pct)

    st.session_state["scan_df"] = result
    st.session_state["scan_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["show_only_breakouts"] = bool(show_only_breakouts)

if "scan_df" not in st.session_state:
    st.info("Click **Run scan** in the sidebar. The app does not auto-refresh or auto-scan.")
    st.stop()

out = st.session_state["scan_df"].copy()
if st.session_state.get("show_only_breakouts", True):
    out = out[out["break_90_high"] | out["break_90_low"] | out["break_180_high"] | out["break_180_low"]].copy()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Symbols", int(out["symbol"].nunique()) if not out.empty else 0)
m2.metric("Break > 90D High", int(out["break_90_high"].sum()) if not out.empty else 0)
m3.metric("Break < 90D Low", int(out["break_90_low"].sum()) if not out.empty else 0)
m4.metric("Break > 180D High", int(out["break_180_high"].sum()) if not out.empty else 0)
m5.metric("Break < 180D Low", int(out["break_180_low"].sum()) if not out.empty else 0)
st.caption(f"Last scan (UTC): {st.session_state.get('scan_at', '')}")

cols = [
    "symbol", "baseAsset", "quoteAsset", "price", "quoteVolume", "priceChangePercent",
    "high_90", "low_90", "high_180", "low_180",
    "break_90_high", "break_90_low", "break_180_high", "break_180_low",
]
cols = [c for c in cols if c in out.columns]

if out.empty:
    st.info("No breakout opportunities matched your filters.")
else:
    st.dataframe(
        out[cols].sort_values(["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"], ascending=[False, False, False, False, True]),
        use_container_width=True,
    )
