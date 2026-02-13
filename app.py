from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from binance_futures import BinanceFuturesPublic, BinanceHTTPError
from breakouts import compute_levels_from_klines, apply_breakout_flags, summarize_breakouts

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ---------- Config ----------
APP_BUILD = "overhaul-minimal-2026-02-12"
BASE_URL = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com").rstrip("/")
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "10"))
REQUESTS_PER_SEC = float(os.environ.get("REQUESTS_PER_SEC", "2.0"))
RETRIES = int(os.environ.get("RETRIES", "1"))

CACHE_SYMBOLS_TTL_SEC = int(os.environ.get("CACHE_SYMBOLS_TTL_SEC", "86400"))
CACHE_SNAPSHOT_TTL_SEC = int(os.environ.get("CACHE_SNAPSHOT_TTL_SEC", "30"))
CACHE_LEVELS_TTL_SEC = int(os.environ.get("CACHE_LEVELS_TTL_SEC", "21600"))

LEVELS_FETCH_DAYS = int(os.environ.get("LEVELS_FETCH_DAYS", "210"))
DEFAULT_QUOTES = os.environ.get("QUOTE_ASSETS", "USDT")
DEFAULT_MIN_QV = float(os.environ.get("MIN_QUOTE_VOL_24H", "20000000"))
DEFAULT_MAX_SYMBOLS = int(os.environ.get("LEVELS_MAX_SYMBOLS", "25"))
DEFAULT_BUFFER_PCT = float(os.environ.get("BREAKOUT_BUFFER_PCT", "0.00"))

st.set_page_config(page_title="Perp 90/180 Breakouts", layout="wide")
st.title("Perp Breakouts Scanner (90D / 180D)")
st.caption(f"Build: {APP_BUILD} • Manual scan only • No auto-refresh")


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def format_ban_message(err: Exception) -> str:
    text = str(err)
    ban = BinanceFuturesPublic.parse_ban_info(text)
    if not ban:
        return text
    now_ms = int(time.time() * 1000)
    remaining = max(0, int((ban.banned_until_ms - now_ms) / 1000))
    banned_until_utc = datetime.fromtimestamp(ban.banned_until_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return f"IP temporarily banned until {banned_until_utc} UTC (~{remaining//60}m {remaining%60}s remaining). Raw: {text}"


def build_client() -> BinanceFuturesPublic:
    return BinanceFuturesPublic(
        base_url=BASE_URL,
        timeout=HTTP_TIMEOUT,
        requests_per_sec=REQUESTS_PER_SEC,
        retries=RETRIES,
    )


@st.cache_data(ttl=CACHE_SYMBOLS_TTL_SEC, show_spinner=False)
def fetch_symbols_cached(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = c.perpetual_symbols()
    return pd.DataFrame([{"symbol": x.symbol, "quoteAsset": x.quote_asset, "baseAsset": x.base_asset} for x in rows])


@st.cache_data(ttl=CACHE_SNAPSHOT_TTL_SEC, show_spinner=False)
def fetch_ticker_cached(base_url: str, timeout: int, rps: float, retries: int) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    data = c.ticker_24hr()
    out = []
    for x in data:
        sym = str(x.get("symbol", "")).upper()
        if not sym:
            continue
        try:
            out.append(
                {
                    "symbol": sym,
                    "price": float(x.get("lastPrice", "nan")),
                    "quoteVolume": float(x.get("quoteVolume", "nan")),
                    "priceChangePercent": float(x.get("priceChangePercent", "nan")),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(out)


@st.cache_data(ttl=CACHE_LEVELS_TTL_SEC, show_spinner=False)
def fetch_levels_cached(
    base_url: str,
    timeout: int,
    rps: float,
    retries: int,
    symbols: tuple[str, ...],
    fetch_days: int,
) -> pd.DataFrame:
    c = BinanceFuturesPublic(base_url=base_url, timeout=timeout, requests_per_sec=rps, retries=retries)
    rows = []
    for sym in symbols:
        kl = c.klines_1d(sym, limit=max(182, int(fetch_days)))
        h90, l90, h180, l180, n_days = compute_levels_from_klines(kl)
        if not np.isfinite(h90) or not np.isfinite(l90):
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


with st.sidebar:
    st.markdown("## Scanner")
    st.caption("Single pass scan only")

    with st.form("scan_form", clear_on_submit=False):
        quote_assets = st.text_input("Quote assets", value=DEFAULT_QUOTES, help="Comma-separated, e.g. USDT,USDC")
        min_qv = st.number_input("Min 24h quote volume", min_value=0.0, value=DEFAULT_MIN_QV, step=5_000_000.0)
        max_symbols = st.slider("Max symbols", min_value=10, max_value=120, value=DEFAULT_MAX_SYMBOLS, step=5)
        buffer_pct = st.number_input("Breakout buffer (%)", min_value=0.0, max_value=5.0, value=DEFAULT_BUFFER_PCT, step=0.01, format="%.2f")
        only_breakouts = st.checkbox("Only show breakouts", value=True)
        run = st.form_submit_button("Run scan", type="primary")

    if st.button("Clear results"):
        st.session_state.pop("scan_df", None)
        st.session_state.pop("scan_ts", None)

if run:
    try:
        syms = fetch_symbols_cached(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
        if syms.empty:
            st.error("No perp symbols available from exchangeInfo")
            st.stop()

        qset = {q.strip().upper() for q in quote_assets.split(",") if q.strip()}
        syms = syms[syms["quoteAsset"].isin(qset)].copy()
        if syms.empty:
            st.warning("No symbols matched quote-asset filter")
            st.stop()

        tkr = fetch_ticker_cached(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES)
        tkr = tkr[tkr["symbol"].isin(syms["symbol"])].copy()
        tkr = tkr[tkr["quoteVolume"].fillna(0) >= float(min_qv)].copy()
        tkr = tkr.sort_values(["quoteVolume", "symbol"], ascending=[False, True]).head(int(max_symbols)).copy()

        selected = tuple(tkr["symbol"].tolist())
        if not selected:
            st.warning("No symbols matched volume/scope filters")
            st.stop()

        levels = fetch_levels_cached(BASE_URL, HTTP_TIMEOUT, REQUESTS_PER_SEC, RETRIES, selected, LEVELS_FETCH_DAYS)
        if levels.empty:
            st.warning("No symbols had enough daily history for 90D/180D levels")
            st.stop()

        df = tkr.merge(levels, on="symbol", how="inner").merge(syms[["symbol", "baseAsset", "quoteAsset"]], on="symbol", how="left")
        df = apply_breakout_flags(df, buffer_pct=buffer_pct)

        st.session_state["scan_df"] = df
        st.session_state["scan_ts"] = utc_now_str()
        st.session_state["only_breakouts"] = bool(only_breakouts)

    except BinanceHTTPError as e:
        st.error(format_ban_message(e))
        st.stop()
    except Exception as e:
        st.error(f"Scan failed: {format_ban_message(e)}")
        st.stop()

if "scan_df" not in st.session_state:
    st.info("Click **Run scan** to fetch breakout opportunities.")
    st.stop()

out = st.session_state["scan_df"].copy()
if st.session_state.get("only_breakouts", True):
    out = out[out["break_90_high"] | out["break_90_low"] | out["break_180_high"] | out["break_180_low"]].copy()

stats = summarize_breakouts(out)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Symbols", stats["symbols"])
c2.metric("Break > 90D High", stats["break_90_high"])
c3.metric("Break < 90D Low", stats["break_90_low"])
c4.metric("Break > 180D High", stats["break_180_high"])
c5.metric("Break < 180D Low", stats["break_180_low"])
st.caption(f"Last scan UTC: {st.session_state.get('scan_ts', '')}")

if out.empty:
    st.warning("No breakout opportunities found for current filters.")
    st.stop()

show_cols = [
    "symbol", "baseAsset", "quoteAsset",
    "price", "quoteVolume", "priceChangePercent",
    "high_90", "low_90", "high_180", "low_180",
    "break_90_high", "break_90_low", "break_180_high", "break_180_low",
]
show_cols = [c for c in show_cols if c in out.columns]

st.dataframe(
    out[show_cols].sort_values(
        ["break_180_high", "break_180_low", "break_90_high", "break_90_low", "symbol"],
        ascending=[False, False, False, False, True],
    ),
    use_container_width=True,
)
