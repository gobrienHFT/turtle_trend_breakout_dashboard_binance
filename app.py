from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from binance_futures import BinanceFuturesPublic, BinanceHTTPError
from breakouts import apply_breakout_flags, compute_levels_from_klines, summarize_breakouts

try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

APP_BUILD = "daily-breakouts-v4-2026-02-13"
BASE_URL = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "10"))
REQUESTS_PER_SEC = float(os.environ.get("REQUESTS_PER_SEC", "1.0"))
RETRIES = int(os.environ.get("RETRIES", "1"))

CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_PRICES_TTL_SEC = int(os.environ.get("CACHE_PRICES_TTL_SEC", "20"))
CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))

LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
DEFAULT_QUOTES = [q.strip().upper() for q in os.environ.get("QUOTE_ASSETS", "USDT").split(",") if q.strip()]
DEFAULT_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "10"))
DEFAULT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))

st.set_page_config(page_title="Binance Perp Daily 90/180D Breakouts", layout="wide")
st.title("Binance Perp Daily 90/180D Breakouts")
st.caption(f"Build: {APP_BUILD}")
st.warning("Manual one-shot scan only. No auto-refresh. No intraday timing.")
st.caption(f"Running file: {Path(__file__).resolve()}")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_error(err: Exception) -> str:
    text = str(err)
    ban = BinanceFuturesPublic.parse_ban_info(text)
    if ban is None:
        return text
    now_ms = int(time.time() * 1000)
    remain_s = max(0, int((ban.banned_until_ms - now_ms) / 1000))
    banned_until = datetime.fromtimestamp(ban.banned_until_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"IP temporarily banned until {banned_until} UTC (~{remain_s // 60}m {remain_s % 60}s remaining)."


@st.cache_data(ttl=CACHE_SYMBOLS_TTL_SEC, show_spinner=False)
def load_symbols(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    client = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = client.perpetual_symbols()
    return pd.DataFrame([{"symbol": r.symbol, "baseAsset": r.base_asset, "quoteAsset": r.quote_asset} for r in rows])


@st.cache_data(ttl=CACHE_PRICES_TTL_SEC, show_spinner=False)
def load_prices(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    client = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = client.ticker_price()
    out = []
    for row in rows:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        try:
            out.append({"symbol": symbol, "price": float(row.get("price", "nan"))})
        except Exception:
            continue
    return pd.DataFrame(out)


@st.cache_data(ttl=CACHE_LEVELS_TTL_SEC, show_spinner=False)
def load_levels(base_url: str, timeout: int, rps: float, retries: int, symbols: tuple[str, ...], fetch_days: int) -> pd.DataFrame:
    client = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    out = []
    for symbol in symbols:
        klines = client.klines_1d(symbol, limit=max(182, int(fetch_days)))
        h90, l90, h180, l180, n_days = compute_levels_from_klines(klines)
        if pd.isna(h90) or pd.isna(l90):
            continue
        out.append(
            {
                "symbol": symbol,
                "high_90": h90,
                "low_90": l90,
                "high_180": h180,
                "low_180": l180,
                "n_days": n_days,
            }
        )
    return pd.DataFrame(out)


with st.form("scan_form"):
    c1, c2, c3 = st.columns(3)
    quotes = c1.multiselect("Quote assets", ["USDT", "USDC", "BTC", "ETH", "BUSD"], default=DEFAULT_QUOTES or ["USDT"])
    max_symbols = c2.slider("Max symbols", min_value=5, max_value=80, value=DEFAULT_MAX_SYMBOLS, step=5)
    buffer_pct = c3.number_input("Breakout buffer (%)", min_value=0.0, max_value=3.0, value=DEFAULT_BUFFER_PCT, step=0.01, format="%.2f")
    show_only_breakouts = st.checkbox("Show only current breakouts", value=True)
    st.caption("Request budget per run: 2 fixed calls (exchangeInfo + ticker/price) + 1 daily kline call per symbol.")
    run_scan = st.form_submit_button("Run one-time scan", type="primary")

if st.button("Clear results"):
    st.session_state.pop("scan_df", None)
    st.session_state.pop("scan_ts", None)

if run_scan:
    try:
        with st.spinner("Running scan..."):
            universe = load_symbols(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
            if universe.empty:
                st.error("No perpetual symbols returned by exchangeInfo.")
                st.stop()

            qset = {q.upper() for q in quotes}
            if qset:
                universe = universe[universe["quoteAsset"].isin(qset)].copy()
            universe = universe.sort_values("symbol").head(int(max_symbols)).copy()
            if universe.empty:
                st.warning("No symbols matched selected quote assets.")
                st.stop()

            prices = load_prices(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
            prices = prices[prices["symbol"].isin(universe["symbol"])].copy()
            if prices.empty:
                st.warning("No prices returned for selected symbols.")
                st.stop()

            levels = load_levels(
                BASE_URL,
                HTTP_TIMEOUT,
                REQUESTS_PER_SEC,
                RETRIES,
                tuple(universe["symbol"].tolist()),
                LEVELS_FETCH_DAYS,
            )
            if levels.empty:
                st.warning("No symbols had enough completed daily candles for 90D/180D levels.")
                st.stop()

            result = universe.merge(prices, on="symbol", how="inner").merge(levels, on="symbol", how="inner")
            result = apply_breakout_flags(result, buffer_pct=float(buffer_pct))

            st.session_state["scan_df"] = result
            st.session_state["scan_ts"] = utc_now_str()
            st.session_state["show_only_breakouts"] = bool(show_only_breakouts)
    except BinanceHTTPError as e:
        st.error(format_error(e))
        st.stop()
    except Exception as e:
        st.error(f"Scan failed: {format_error(e)}")
        st.stop()

if "scan_df" not in st.session_state:
    st.info("Press 'Run one-time scan' to fetch current daily 90/180D breakouts.")
    st.stop()

output = st.session_state["scan_df"].copy()
if st.session_state.get("show_only_breakouts", True):
    output = output[
        output["break_90_high"] | output["break_90_low"] | output["break_180_high"] | output["break_180_low"]
    ].copy()

stats = summarize_breakouts(output)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Symbols", stats["symbols"])
m2.metric("> 90D High", stats["break_90_high"])
m3.metric("< 90D Low", stats["break_90_low"])
m4.metric("> 180D High", stats["break_180_high"])
m5.metric("< 180D Low", stats["break_180_low"])
st.caption(f"Last scan UTC: {st.session_state.get('scan_ts', '')}")

if output.empty:
    st.warning("No active breakouts in this scan.")
    st.stop()

cols = [
    "symbol",
    "baseAsset",
    "quoteAsset",
    "price",
    "high_90",
    "low_90",
    "high_180",
    "low_180",
    "break_90_high",
    "break_90_low",
    "break_180_high",
    "break_180_low",
]

st.dataframe(
    output[cols].sort_values(
        ["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"],
        ascending=[False, False, False, False, True],
    ),
    use_container_width=True,
)
