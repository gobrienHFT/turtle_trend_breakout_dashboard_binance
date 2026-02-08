from __future__ import annotations

import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List

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
REFRESH_SECONDS = int(os.environ.get("REFRESH_SECONDS", "5"))

CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))
CACHE_SNAPSHOT_TTL_SEC = int(os.environ.get("CACHE_SNAPSHOT_TTL_SEC", "5"))
CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_INTRADAY_TTL_SEC = int(os.environ.get("CACHE_INTRADAY_TTL_SEC", "120"))

LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
LEVELS_THREADS = int(os.environ.get("LEVELS_THREADS", "12"))
MAX_TIMING_SYMBOLS = int(os.environ.get("MAX_TIMING_SYMBOLS", "120"))
LEVELS_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "250"))
RATE_LIMIT_REQ_PER_SEC = float(os.environ.get("RATE_LIMIT_REQ_PER_SEC", "12"))
MIN_REFRESH_WITH_TIMES_SEC = int(os.environ.get("MIN_REFRESH_WITH_TIMES_SEC", "15"))

QUOTE_ASSETS_ENV = os.environ.get("QUOTE_ASSETS", "").strip()
MIN_QUOTE_VOL_24H_ENV = float(os.environ.get("MIN_QUOTE_VOL_24H", "0"))
BREAKOUT_BUFFER_PCT_ENV = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))

BREAKOUT_TIME_INTERVAL = os.environ.get("BREAKOUT_TIME_INTERVAL", "15m").strip()
BREAKOUT_TIME_LOOKBACK_HOURS = int(os.environ.get("BREAKOUT_TIME_LOOKBACK_HOURS", "168"))
RECENT_WINDOW_DEFAULT_HOURS = int(os.environ.get("RECENT_WINDOW_DEFAULT_HOURS", "24"))

DEFAULT_BASE_URL = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")

_HTTP_REQ_LOCK = threading.Lock()
_LAST_HTTP_REQ_AT = 0.0

st.set_page_config(page_title="Binance Perp 90D/180D Breakouts", layout="wide")
st.title("Binance Perpetual Futures — 90D / 180D High-Low Breakouts")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fmt_ago(hours: float) -> str:
    if np.isnan(hours):
        return ""
    if hours < 1:
        return f"{hours*60:.0f}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours/24:.1f}d"


class BinanceHTTPError(RuntimeError):
    def __init__(self, status: int, url: str, body: str):
        super().__init__(f"HTTP {status} for {url}: {body[:300]}")
        self.status = status
        self.url = url
        self.body = body[:2000]


def fapi_get(
    base_url: str,
    path: str,
    params: Optional[dict] = None,
    timeout: Optional[int] = None,
    retries: int = 4,
):
    global _LAST_HTTP_REQ_AT
    if timeout is None:
        if threading.current_thread() is threading.main_thread():
            timeout = int(st.session_state.get("http_timeout", HTTP_TIMEOUT))
        else:
            timeout = HTTP_TIMEOUT
    else:
        timeout = int(timeout)

    url = f"{base_url}{path}"
    headers = {
        "User-Agent": "TrendBreakoutDashboard/1.0 (+streamlit)",
        "Accept": "application/json",
    }

    backoff = 0.6
    last_exc: Optional[Exception] = None

    for _ in range(1, retries + 1):
        try:
            if RATE_LIMIT_REQ_PER_SEC > 0:
                with _HTTP_REQ_LOCK:
                    min_interval = 1.0 / RATE_LIMIT_REQ_PER_SEC
                    now = time.monotonic()
                    wait = min_interval - (now - _LAST_HTTP_REQ_AT)
                    if wait > 0:
                        time.sleep(wait)
                    _LAST_HTTP_REQ_AT = time.monotonic()

            r = requests.get(url, params=params or {}, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r.json()

            if r.status_code in (418, 429, 500, 502, 503, 504):
                last_exc = BinanceHTTPError(r.status_code, url, r.text)
                time.sleep(backoff)
                backoff = min(6.0, backoff * 1.8)
                continue

            raise BinanceHTTPError(r.status_code, url, r.text)

        except requests.RequestException as e:
            last_exc = e
            time.sleep(backoff)
            backoff = min(6.0, backoff * 1.8)

    if isinstance(last_exc, Exception):
        raise last_exc
    raise RuntimeError("Unknown HTTP failure")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


@st.cache_data(ttl=CACHE_SYMBOLS_TTL_SEC, show_spinner=False)
def get_perp_symbols(base_url: str) -> pd.DataFrame:
    info = fapi_get(base_url, "/fapi/v1/exchangeInfo")
    rows = []
    for s in info.get("symbols", []):
        if s.get("contractType") != "PERPETUAL":
            continue
        if s.get("status") != "TRADING":
            continue
        sym = s.get("symbol")
        if not sym:
            continue
        rows.append({"symbol": sym, "baseAsset": s.get("baseAsset", ""), "quoteAsset": s.get("quoteAsset", "")})

    df = pd.DataFrame(rows, columns=["symbol", "baseAsset", "quoteAsset"])
    if df.empty:
        return df
    return df.drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)


@st.cache_data(ttl=CACHE_SNAPSHOT_TTL_SEC, show_spinner=False)
def get_snapshot_mark_price(base_url: str) -> pd.DataFrame:
    data = fapi_get(base_url, "/fapi/v1/premiumIndex")
    if isinstance(data, dict):
        data = [data]
    rows = []
    for x in data:
        sym = x.get("symbol")
        if not sym:
            continue
        rows.append({
            "symbol": sym,
            "price": safe_float(x.get("markPrice")),
            "indexPrice": safe_float(x.get("indexPrice")),
            "fundingRate": safe_float(x.get("lastFundingRate")),
            "time": int(x.get("time", 0)) if str(x.get("time", "")).isdigit() else np.nan,
        })
    return pd.DataFrame(rows, columns=["symbol", "price", "indexPrice", "fundingRate", "time"])


@st.cache_data(ttl=CACHE_SNAPSHOT_TTL_SEC, show_spinner=False)
def get_snapshot_last_24h(base_url: str) -> pd.DataFrame:
    data = fapi_get(base_url, "/fapi/v1/ticker/24hr")
    rows = []
    for x in data:
        sym = x.get("symbol")
        if not sym:
            continue
        rows.append({
            "symbol": sym,
            "price": safe_float(x.get("lastPrice")),
            "quoteVolume": safe_float(x.get("quoteVolume")),
            "priceChangePercent": safe_float(x.get("priceChangePercent")),
        })
    return pd.DataFrame(rows, columns=["symbol", "price", "quoteVolume", "priceChangePercent"])


def fetch_klines_1d(base_url: str, symbol: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        data = fapi_get(base_url, "/fapi/v1/klines", params={"symbol": symbol, "interval": "1d", "limit": int(limit)})
        if not isinstance(data, list) or len(data) < 10:
            return None
        df = pd.DataFrame(data, columns=[
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "quoteVolume", "numTrades", "takerBaseVol", "takerQuoteVol", "ignore",
        ])
        df["openTime"] = df["openTime"].astype("int64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["quoteVolume"] = df["quoteVolume"].astype("float64")
        return df[["openTime", "high", "low", "close", "quoteVolume"]]
    except Exception:
        return None


def _bucket_ms(ts_ms: int, bucket_ms: int) -> int:
    if bucket_ms <= 0:
        return int(ts_ms)
    ts_ms = int(ts_ms)
    return ts_ms - (ts_ms % int(bucket_ms))


@st.cache_data(ttl=CACHE_INTRADAY_TTL_SEC, show_spinner=False)
def fetch_klines_intraday_cached(base_url: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> Optional[pd.DataFrame]:
    return fetch_klines_intraday(base_url, symbol, interval, start_ms, end_ms)


@st.cache_data(ttl=CACHE_LEVELS_TTL_SEC, show_spinner=False)
def compute_levels_all(base_url: str, symbols: List[str], fetch_days: int, threads: int) -> pd.DataFrame:
    out_rows = []

    for sym in symbols:
        df = fetch_klines_1d(base_url, sym, limit=max(fetch_days, 182))
        if df is None or df.empty:
            continue

        df2 = df.iloc[:-1].copy() if len(df) >= 2 else df.copy()
        if len(df2) < 10:
            continue

        h90 = l90 = h180 = l180 = np.nan
        if len(df2) >= 90:
            w = df2.iloc[-90:]
            h90 = float(np.max(w["high"].values))
            l90 = float(np.min(w["low"].values))
        if len(df2) >= 180:
            w = df2.iloc[-180:]
            h180 = float(np.max(w["high"].values))
            l180 = float(np.min(w["low"].values))

        out_rows.append({
            "symbol": sym,
            "high_90": h90,
            "low_90": l90,
            "high_180": h180,
            "low_180": l180,
            "hist_days": int(len(df2)),
        })

    df_out = pd.DataFrame(out_rows, columns=["symbol", "high_90", "low_90", "high_180", "low_180", "hist_days"])
    return df_out.sort_values("symbol").reset_index(drop=True) if not df_out.empty else df_out


def fetch_klines_intraday(base_url: str, symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1500) -> Optional[pd.DataFrame]:
    rows = []
    cur = start_ms
    try:
        while True:
            data = fapi_get(base_url, "/fapi/v1/klines", params={
                "symbol": symbol,
                "interval": interval,
                "startTime": int(cur),
                "endTime": int(end_ms),
                "limit": int(limit),
            })
            if not isinstance(data, list) or len(data) == 0:
                break
            rows.extend(data)
            last_open = int(data[-1][0])
            if last_open <= cur:
                break
            cur = last_open + 1
            if int(data[-1][6]) >= end_ms:
                break
            if len(rows) > 60000:
                break

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=[
            "openTime", "open", "high", "low", "close", "volume",
            "closeTime", "quoteVolume", "numTrades", "takerBaseVol", "takerQuoteVol", "ignore",
        ])
        df["openTime"] = df["openTime"].astype("int64")
        df["closeTime"] = df["closeTime"].astype("int64")
        df["close"] = df["close"].astype("float64")
        return df[["openTime", "closeTime", "close"]].sort_values("openTime").reset_index(drop=True)
    except Exception:
        return None


def last_cross_time(df: pd.DataFrame, level: float, direction: str) -> Optional[int]:
    if df is None or df.empty or not np.isfinite(level):
        return None
    c = df["close"].values
    ct = df["closeTime"].values
    if len(c) < 2:
        return None

    if direction == "up":
        idx = np.where((c[:-1] <= level) & (c[1:] > level))[0]
    else:
        idx = np.where((c[:-1] >= level) & (c[1:] < level))[0]
    if idx.size == 0:
        return None
    return int(ct[int(idx[-1] + 1)])


def only_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    mask = df["break_90_high"] | df["break_90_low"] | df["break_180_high"] | df["break_180_low"]
    return df[mask].copy()


@st.cache_data(ttl=CACHE_INTRADAY_TTL_SEC, show_spinner=False)
def enrich_breakout_times(base_url: str, df: pd.DataFrame, interval: str, lookback_hours: int, buffer_pct: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    now = utc_now()
    raw_end_ms = int(now.timestamp() * 1000)
    end_ms = _bucket_ms(raw_end_ms, 60_000)
    start_ms = int(end_ms - int(lookback_hours) * 3600 * 1000)

    out = df.copy()

    def high_thr(x): return x * (1.0 + buffer_pct/100.0)
    def low_thr(x): return x * (1.0 - buffer_pct/100.0)

    b90h_at = []
    b90l_at = []
    b180h_at = []
    b180l_at = []
    for _, r in out.iterrows():
        sym = r["symbol"]
        df_i = fetch_klines_intraday_cached(base_url, sym, interval, start_ms, end_ms)
        if df_i is None:
            b90h_at.append(np.nan); b90l_at.append(np.nan); b180h_at.append(np.nan); b180l_at.append(np.nan)
            continue

        h90 = r.get("high_90", np.nan); l90 = r.get("low_90", np.nan)
        h180 = r.get("high_180", np.nan); l180 = r.get("low_180", np.nan)

        t90h = last_cross_time(df_i, high_thr(h90), "up") if np.isfinite(h90) else None
        t90l = last_cross_time(df_i, low_thr(l90), "down") if np.isfinite(l90) else None
        t180h = last_cross_time(df_i, high_thr(h180), "up") if np.isfinite(h180) else None
        t180l = last_cross_time(df_i, low_thr(l180), "down") if np.isfinite(l180) else None

        b90h_at.append(float(t90h) if t90h else np.nan)
        b90l_at.append(float(t90l) if t90l else np.nan)
        b180h_at.append(float(t180h) if t180h else np.nan)
        b180l_at.append(float(t180l) if t180l else np.nan)

    out["b_90h_at"] = b90h_at; out["b_90l_at"] = b90l_at; out["b_180h_at"] = b180h_at; out["b_180l_at"] = b180l_at

    now_ms = end_ms
    for col in ["b_90h_at", "b_90l_at", "b_180h_at", "b_180l_at"]:
        out[col.replace("_at", "_ago_h")] = (now_ms - out[col]) / 1000.0 / 3600.0

    ago_cols = ["b_90h_ago_h", "b_90l_ago_h", "b_180h_ago_h", "b_180l_ago_h"]
    out["b_last_ago_h"] = out[ago_cols].min(axis=1, skipna=True)
    out["b_last_at"] = np.where(np.isfinite(out["b_last_ago_h"]), now_ms - out["b_last_ago_h"]*3600.0*1000.0, np.nan)

    for c in ["b_90h_at", "b_90l_at", "b_180h_at", "b_180l_at", "b_last_at"]:
        out[c + "_utc"] = out[c].apply(lambda x: "" if not np.isfinite(x) else to_utc_str(int(x)))
    for c in ["b_90h_ago_h", "b_90l_ago_h", "b_180h_ago_h", "b_180l_ago_h", "b_last_ago_h"]:
        out[c.replace("_ago_h", "_ago")] = out[c].apply(lambda x: "" if not np.isfinite(x) else fmt_ago(float(x)))

    return out


def apply_breakout_flags(df: pd.DataFrame, buffer_pct: float) -> pd.DataFrame:
    out = df.copy()
    buf = float(buffer_pct) / 100.0
    out["thr_90_high"] = out["high_90"] * (1.0 + buf)
    out["thr_90_low"] = out["low_90"] * (1.0 - buf)
    out["thr_180_high"] = out["high_180"] * (1.0 + buf)
    out["thr_180_low"] = out["low_180"] * (1.0 - buf)

    out["break_90_high"] = (out["price"] > out["thr_90_high"]) & np.isfinite(out["thr_90_high"])
    out["break_90_low"] = (out["price"] < out["thr_90_low"]) & np.isfinite(out["thr_90_low"])
    out["break_180_high"] = (out["price"] > out["thr_180_high"]) & np.isfinite(out["thr_180_high"])
    out["break_180_low"] = (out["price"] < out["thr_180_low"]) & np.isfinite(out["thr_180_low"])
    return out


with st.sidebar:
    st.markdown("## Settings")

    base_url_input = st.text_input("Binance Futures Base URL", value=DEFAULT_BASE_URL, key="base_url")
    base_url = (base_url_input or DEFAULT_BASE_URL).rstrip("/")

    st.number_input("HTTP timeout (sec)", min_value=3, max_value=60, value=HTTP_TIMEOUT, step=1, key="http_timeout")
    refresh_seconds = st.slider("Auto refresh (seconds)", min_value=2, max_value=60, value=REFRESH_SECONDS, step=1)
    price_source = st.selectbox("Live price source", options=["MARK_PRICE", "LAST_24H"], index=0)

    st.markdown("## Universe filters")
    try:
        df_syms = get_perp_symbols(base_url)
    except Exception as e:
        st.error(f"Failed to load exchangeInfo from {base_url}. Error: {e}")
        st.stop()

    quote_assets_all = sorted(df_syms["quoteAsset"].dropna().unique().tolist())
    default_quotes = [q.strip().upper() for q in (QUOTE_ASSETS_ENV.split(",") if QUOTE_ASSETS_ENV else []) if q.strip()]
    if not default_quotes:
        default_quotes = ["USDT"] if "USDT" in quote_assets_all else quote_assets_all[:1]

    quote_assets_sel = st.multiselect("Quote assets", quote_assets_all, default=default_quotes)
    min_qv = st.number_input("Min 24h quote volume (USDT)", min_value=0.0, value=float(MIN_QUOTE_VOL_24H_ENV), step=10_000_000.0)
    buffer_pct = st.number_input("Breakout buffer (%)", min_value=0.0, max_value=5.0, value=float(BREAKOUT_BUFFER_PCT_ENV), step=0.01, format="%.2f")

    st.markdown("## Table view")
    show_only_breakouts = st.checkbox("Show only breakouts", value=True)
    show_90 = st.checkbox("Show 90D columns", value=True)
    show_180 = st.checkbox("Show 180D columns", value=True)
    show_times = st.checkbox("Show breakout timestamps + recency", value=True)

    st.markdown("## Recent breakouts")
    recent_hours = st.slider("Recent breakout window (hours)", min_value=1, max_value=168, value=int(RECENT_WINDOW_DEFAULT_HOURS), step=1)
    only_recent = st.checkbox("Filter: only breakouts inside recent window", value=False)

    st.markdown("## Intraday timing")
    intraday_interval = st.selectbox(
        "Intraday interval",
        options=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"],
        index=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"].index(BREAKOUT_TIME_INTERVAL)
        if BREAKOUT_TIME_INTERVAL in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h"] else 3,
    )
    intraday_lookback = st.slider("Intraday lookback (hours)", min_value=24, max_value=720, value=int(BREAKOUT_TIME_LOOKBACK_HOURS), step=24)

effective_refresh_seconds = int(refresh_seconds)
if show_times and effective_refresh_seconds < MIN_REFRESH_WITH_TIMES_SEC:
    effective_refresh_seconds = MIN_REFRESH_WITH_TIMES_SEC
    st.info(
        f"Auto refresh increased to {effective_refresh_seconds}s while breakout timestamps are enabled "
        "to prevent endless reruns before intraday computation finishes."
    )

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=effective_refresh_seconds * 1000, key="auto_refresh")
except Exception:
    pass

syms_df = df_syms[df_syms["quoteAsset"].isin(quote_assets_sel)].copy()
symbols = syms_df["symbol"].tolist()

snap = None
snap_err = None

if price_source == "MARK_PRICE":
    try:
        snap = get_snapshot_mark_price(base_url)
        t24 = get_snapshot_last_24h(base_url)
        snap = snap.merge(t24[["symbol", "quoteVolume", "priceChangePercent"]], on="symbol", how="left")
    except Exception as e:
        snap_err = e
        try:
            snap = get_snapshot_last_24h(base_url)
            snap["indexPrice"] = np.nan
            snap["fundingRate"] = np.nan
        except Exception as e2:
            st.error(f"Both MARK_PRICE and LAST_24H failed.\n\nMARK_PRICE error:\n{e}\n\nLAST_24H error:\n{e2}")
            st.stop()
else:
    try:
        snap = get_snapshot_last_24h(base_url)
        snap["indexPrice"] = np.nan
        snap["fundingRate"] = np.nan
    except Exception as e:
        st.error(f"Failed to load ticker/24hr from {base_url}. Error: {e}")
        st.stop()

if snap_err is not None:
    st.warning(f"MARK_PRICE failed (likely 403/451/429). Fell back to LAST_24H. Error: {snap_err}")

snap = snap[snap["symbol"].isin(symbols)].copy()

for _c, _default in (("quoteVolume", np.nan), ("priceChangePercent", np.nan), ("fundingRate", np.nan), ("indexPrice", np.nan)):
    if _c not in snap.columns:
        snap[_c] = _default

if min_qv > 0:
    snap = snap[(snap["quoteVolume"].fillna(0) >= min_qv)].copy()

symbols = snap["symbol"].tolist()

if LEVELS_MAX_SYMBOLS > 0 and len(symbols) > LEVELS_MAX_SYMBOLS:
    st.warning(
        f"Universe capped to top {LEVELS_MAX_SYMBOLS} symbols by 24h quote volume for level computation/performance. "
        f"Increase LEVELS_MAX_SYMBOLS to scan more symbols."
    )
    snap = snap.sort_values(["quoteVolume", "symbol"], ascending=[False, True]).head(LEVELS_MAX_SYMBOLS).copy()
    symbols = snap["symbol"].tolist()

if snap.empty:
    st.warning("No symbols matched the current filters (quote assets / volume), or Binance returned an empty snapshot.")
    st.stop()

levels_df = compute_levels_all(base_url, symbols, fetch_days=LEVELS_FETCH_DAYS, threads=LEVELS_THREADS)

df = snap.merge(levels_df, on="symbol", how="left")
df = df.merge(syms_df[["symbol", "baseAsset", "quoteAsset"]], on="symbol", how="left")

df = apply_breakout_flags(df, buffer_pct=buffer_pct)

n_symbols = int(df["symbol"].nunique())
n_90h = int(df["break_90_high"].sum())
n_90l = int(df["break_90_low"].sum())
n_180h = int(df["break_180_high"].sum())
n_180l = int(df["break_180_low"].sum())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Symbols", f"{n_symbols}")
m2.metric("Break > 90D High", f"{n_90h}")
m3.metric("Break < 90D Low", f"{n_90l}")
m4.metric("Break > 180D High", f"{n_180h}")
m5.metric("Break < 180D Low", f"{n_180l}")

st.caption(f"Last update (UTC): {utc_now().strftime('%Y-%m-%d %H:%M:%S')}")

breakout_df = only_breakouts(df)
df_view = breakout_df.copy() if show_only_breakouts else df.copy()

if show_times and not df_view.empty:
    # Avoid requesting intraday candles for the full universe at 5s autorefresh.
    # Default is to enrich only symbols actually shown in table; plus hard cap.
    if len(df_view) > MAX_TIMING_SYMBOLS:
        st.warning(
            f"Timing enrichment capped at {MAX_TIMING_SYMBOLS} symbols to avoid Binance rate-limit issues. "
            f"Narrow filters (quote asset / volume / show only breakouts) for full timing coverage."
        )
        df_view = df_view.sort_values(["quoteVolume", "symbol"], ascending=[False, True]).head(MAX_TIMING_SYMBOLS).copy()

    with st.spinner("Computing breakout timestamps (intraday)…"):
        df_view = enrich_breakout_times(base_url, df_view, intraday_interval, intraday_lookback, buffer_pct)

if show_times and only_recent and "b_last_ago_h" in df_view.columns:
    df_view = df_view[np.isfinite(df_view["b_last_ago_h"]) & (df_view["b_last_ago_h"] <= float(recent_hours))].copy()

if show_times:
    st.markdown(f"## Breakouts in the past {recent_hours} hours")
    recent_df = df_view.copy() if show_only_breakouts else only_breakouts(df_view)
    if "b_last_ago_h" in recent_df.columns:
        recent_df = recent_df[np.isfinite(recent_df["b_last_ago_h"]) & (recent_df["b_last_ago_h"] <= float(recent_hours))].copy()

    if recent_df.empty:
        st.info("No breakouts detected in the recent window.")
    else:
        cols = ["symbol", "quoteAsset", "price", "break_90_high", "break_90_low", "break_180_high", "break_180_low", "b_last_at_utc", "b_last_ago"]
        cols_existing = [c for c in cols if c in recent_df.columns]
        st.dataframe(recent_df[cols_existing].sort_values(["b_last_ago_h", "symbol"], ascending=[True, True]), use_container_width=True)

st.markdown("## Breakouts right now")

base_cols = ["symbol", "baseAsset", "quoteAsset", "price", "quoteVolume", "priceChangePercent", "fundingRate"]
cols_90 = ["high_90", "low_90", "thr_90_high", "thr_90_low", "break_90_high", "break_90_low"]
cols_180 = ["high_180", "low_180", "thr_180_high", "thr_180_low", "break_180_high", "break_180_low"]
time_cols = [
    "b_90h_at_utc", "b_90h_ago", "b_90l_at_utc", "b_90l_ago",
    "b_180h_at_utc", "b_180h_ago", "b_180l_at_utc", "b_180l_ago", "b_last_at_utc", "b_last_ago",
]

show_cols = base_cols.copy()
if show_90:
    show_cols += cols_90
if show_180:
    show_cols += cols_180
if show_times:
    show_cols += time_cols
show_cols = [c for c in show_cols if c in df_view.columns]

if show_times and "b_last_ago_h" in df_view.columns:
    df_view = df_view.sort_values(
        by=["break_180_high", "break_180_low", "break_90_high", "break_90_low", "b_last_ago_h", "symbol"],
        ascending=[False, False, False, False, True, True],
    )
else:
    df_view = df_view.sort_values(
        by=["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"],
        ascending=[False, False, False, False, True],
    )

st.dataframe(df_view[show_cols], use_container_width=True)

with st.expander("Show all symbols snapshot"):
    # Keep this light: do not trigger a second full intraday enrichment pass.
    st.dataframe(df.sort_values("symbol")[base_cols + cols_90 + cols_180], use_container_width=True)

st.markdown("## Symbol detail")
if df.empty:
    st.info("No symbols available with the current filters.")
else:
    sel = st.selectbox("Select symbol", options=sorted(df["symbol"].unique().tolist()))
    row = df[df["symbol"] == sel].head(1)
    if not row.empty:
        r = row.iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"{r.get('price', np.nan):.6g}")
        c2.metric("90D High / Low", f"{r.get('high_90', np.nan):.6g} / {r.get('low_90', np.nan):.6g}")
        c3.metric("180D High / Low", f"{r.get('high_180', np.nan):.6g} / {r.get('low_180', np.nan):.6g}")
        c4.metric("24h Quote Vol", f"{r.get('quoteVolume', np.nan):.6g}")

        if show_times:
            df_one = df[df["symbol"] == sel].copy()
            df_one = enrich_breakout_times(base_url, df_one, intraday_interval, intraday_lookback, buffer_pct)
            rr = df_one.iloc[0].to_dict()
            st.write({
                "90H break at": rr.get("b_90h_at_utc", ""),
                "90H ago": rr.get("b_90h_ago", ""),
                "90L break at": rr.get("b_90l_at_utc", ""),
                "90L ago": rr.get("b_90l_ago", ""),
                "180H break at": rr.get("b_180h_at_utc", ""),
                "180H ago": rr.get("b_180h_ago", ""),
                "180L break at": rr.get("b_180l_at_utc", ""),
                "180L ago": rr.get("b_180l_ago", ""),
                "last break at": rr.get("b_last_at_utc", ""),
                "last break ago": rr.get("b_last_ago", ""),
            })
