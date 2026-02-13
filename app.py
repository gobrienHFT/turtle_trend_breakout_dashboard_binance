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

APP_BUILD = "daily-breakouts-v3-2026-02-13"
BASE_URL = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "10"))
REQUESTS_PER_SEC = float(os.environ.get("REQUESTS_PER_SEC", "1.0"))
RETRIES = int(os.environ.get("RETRIES", "1"))

CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_PRICES_TTL_SEC = int(os.environ.get("CACHE_PRICES_TTL_SEC", "20"))
CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))

LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
DEFAULT_QUOTES = [x.strip().upper() for x in os.environ.get("QUOTE_ASSETS", "USDT").split(",") if x.strip()]
DEFAULT_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "10"))
DEFAULT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))

st.set_page_config(page_title="Binance Perp Daily 90/180D Breakouts", layout="wide")
st.title("Binance Perp Daily 90/180D Breakouts")
st.caption(f"Build: {APP_BUILD}")
st.warning(
    "This app is one-time/manual only. If you still see auto-refresh or intraday controls, you are running an old app.py."
)
st.caption(f"Running file: {Path(__file__).resolve()}")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def fmt_error(err: Exception) -> str:
    text = str(err)
    ban = BinanceFuturesPublic.parse_ban_info(text)
    if not ban:
        return text
    now_ms = int(time.time() * 1000)
    remaining = max(0, int((ban.banned_until_ms - now_ms) / 1000))
    banned_until = datetime.fromtimestamp(ban.banned_until_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"IP temporarily banned until {banned_until} UTC (~{remaining // 60}m {remaining % 60}s remaining)."


@st.cache_data(ttl=CACHE_SYMBOLS_TTL_SEC, show_spinner=False)
def get_symbols(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = c.perpetual_symbols()
    return pd.DataFrame([{"symbol": r.symbol, "baseAsset": r.base_asset, "quoteAsset": r.quote_asset} for r in rows])


@st.cache_data(ttl=CACHE_PRICES_TTL_SEC, show_spinner=False)
def get_prices(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = c.ticker_price()
    out = []
    for x in rows:
        sym = str(x.get("symbol", "")).upper()
        if not sym:
            continue
        try:
            out.append({"symbol": sym, "price": float(x.get("price", "nan"))})
        except Exception:
            continue
    return pd.DataFrame(out)


@st.cache_data(ttl=CACHE_LEVELS_TTL_SEC, show_spinner=False)
def get_levels(base_url: str, timeout: int, rps: float, retries: int, symbols: tuple[str, ...], fetch_days: int) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = []
    for sym in symbols:
        kl = c.klines_1d(sym, limit=max(182, int(fetch_days)))
        h90, l90, h180, l180, n_days = compute_levels_from_klines(kl)
        if pd.isna(h90) or pd.isna(l90):
            continue
        rows.append(
            {
                "symbol": sym,
                "high_90": h90,
                "low_90": l90,
                "high_180": h180,
                "low_180": l180,
                "n_days": n_days,
            }
        )
    return pd.DataFrame(rows)


with st.form("run_scan"):
    c1, c2, c3 = st.columns(3)
    quotes = c1.multiselect("Quote assets", options=["USDT", "USDC", "BTC", "ETH", "BUSD"], default=DEFAULT_QUOTES or ["USDT"])
    max_symbols = c2.slider("Max symbols", min_value=5, max_value=80, value=DEFAULT_MAX_SYMBOLS, step=5)
    buffer_pct = c3.number_input("Breakout buffer (%)", min_value=0.0, max_value=3.0, value=DEFAULT_BUFFER_PCT, step=0.01, format="%.2f")
    only_breakouts = st.checkbox("Show only current breakouts", value=True)
    st.caption("Per run request budget: 2 base calls + N daily-kline calls, where N = selected symbols.")
    run_scan = st.form_submit_button("Run one-time scan", type="primary")

if st.button("Clear results"):
    st.session_state.pop("scan_df", None)
    st.session_state.pop("scan_ts", None)

if run_scan:
    try:
        with st.spinner("Scanning..."):
            syms = get_symbols(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
            if syms.empty:
                st.error("No perpetual symbols returned by Binance exchangeInfo.")
                st.stop()

            qset = {q.upper() for q in quotes}
            syms = syms[syms["quoteAsset"].isin(qset)].copy() if qset else syms.copy()
            syms = syms.sort_values("symbol").head(int(max_symbols)).copy()
            if syms.empty:
                st.warning("No symbols matched selected quote assets.")
                st.stop()

            prices = get_prices(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
            prices = prices[prices["symbol"].isin(syms["symbol"])].copy()
            if prices.empty:
                st.warning("No prices returned for selected symbols.")
                st.stop()

            levels = get_levels(
                BASE_URL,
                HTTP_TIMEOUT,
                REQUESTS_PER_SEC,
                RETRIES,
                tuple(syms["symbol"].tolist()),
                LEVELS_FETCH_DAYS,
            )
            if levels.empty:
                st.warning("No symbols had enough completed daily candles for 90/180D levels.")
                st.stop()

            out = syms.merge(prices, on="symbol", how="inner").merge(levels, on="symbol", how="inner")
            out = apply_breakout_flags(out, buffer_pct=float(buffer_pct))

            st.session_state["scan_df"] = out
            st.session_state["scan_ts"] = utc_now_str()
            st.session_state["only_breakouts"] = bool(only_breakouts)

    except BinanceHTTPError as e:
        st.error(fmt_error(e))
        st.stop()
    except Exception as e:
        st.error(f"Scan failed: {fmt_error(e)}")
        st.stop()

if "scan_df" not in st.session_state:
    st.info("Press 'Run one-time scan' to fetch current daily 90/180D breakouts.")
    st.stop()

out = st.session_state["scan_df"].copy()
if st.session_state.get("only_breakouts", True):
    out = out[out["break_90_high"] | out["break_90_low"] | out["break_180_high"] | out["break_180_low"]].copy()

stats = summarize_breakouts(out)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Symbols", stats["symbols"])
c2.metric("> 90D High", stats["break_90_high"])
c3.metric("< 90D Low", stats["break_90_low"])
c4.metric("> 180D High", stats["break_180_high"])
c5.metric("< 180D Low", stats["break_180_low"])
st.caption(f"Last scan (UTC): {st.session_state.get('scan_ts', '')}")

if out.empty:
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
    out[cols].sort_values(
        ["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"],
        ascending=[False, False, False, False, True],
    ),
    use_container_width=True,
)
