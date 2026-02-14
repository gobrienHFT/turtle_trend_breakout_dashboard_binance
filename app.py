from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from binance_futures import BinanceFuturesPublic
from breakouts import BreakoutRow, levels_from_klines

BASE_URL = os.environ.get("BINANCE_FAPI_BASE", "https://fapi.binance.com")
TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "12"))
REQUESTS_PER_SECOND = float(os.environ.get("REQUESTS_PER_SECOND", "4.0"))
RETRIES = int(os.environ.get("HTTP_RETRIES", "3"))
MAX_SYMBOLS_TO_SCAN = int(os.environ.get("MAX_SYMBOLS_TO_SCAN", "45"))

st.set_page_config(page_title="Simple Binance Breakouts", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0b1020 0%, #111827 100%); color: #f3f4f6; }
    h1, h2, h3, p, label, .stMarkdown, .stCaption { color: #f9fafb !important; }
    .card { background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Binance USDT Perp 90D/180D Breakout Dashboard")
st.caption("Single-click scan with request pacing. No sliders. No auto-refresh.")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@st.cache_data(ttl=180)
def run_scan() -> tuple[pd.DataFrame, pd.DataFrame]:
    client = BinanceFuturesPublic(
        base_url=BASE_URL,
        timeout=TIMEOUT,
        requests_per_second=REQUESTS_PER_SECOND,
        retries=RETRIES,
    )

    symbols = {s.symbol: s.base_asset for s in client.perpetual_usdt_symbols()}
    ticker = pd.DataFrame(client.ticker_24hr())
    if ticker.empty:
        return pd.DataFrame(), pd.DataFrame()

    ticker["symbol"] = ticker["symbol"].astype(str).str.upper()
    ticker = ticker[ticker["symbol"].isin(symbols.keys())].copy()

    for col in ("lastPrice", "highPrice", "lowPrice", "quoteVolume"):
        ticker[col] = pd.to_numeric(ticker.get(col), errors="coerce")

    ticker = ticker.dropna(subset=["lastPrice", "highPrice", "lowPrice", "quoteVolume"])
    ticker = ticker.sort_values("quoteVolume", ascending=False).head(MAX_SYMBOLS_TO_SCAN)

    rows: list[BreakoutRow] = []
    for _, t in ticker.iterrows():
        symbol = str(t["symbol"])
        klines = client.klines_1d(symbol, limit=181)
        h90, l90, h180, l180 = levels_from_klines(klines)
        if any(math.isnan(x) for x in (h90, l90, h180, l180)):
            continue

        high_24h = float(t["highPrice"])
        low_24h = float(t["lowPrice"])
        rows.append(
            BreakoutRow(
                symbol=symbol,
                base_asset=symbols[symbol],
                last_price=float(t["lastPrice"]),
                high_24h=high_24h,
                low_24h=low_24h,
                high_90d=h90,
                low_90d=l90,
                high_180d=h180,
                low_180d=l180,
                broke_high_90d=high_24h > h90,
                broke_high_180d=high_24h > h180,
                broke_low_90d=low_24h < l90,
                broke_low_180d=low_24h < l180,
            )
        )

    all_df = pd.DataFrame([r.__dict__ for r in rows]).sort_values("symbol")
    high_breaks_df = all_df[(all_df["broke_high_90d"]) | (all_df["broke_high_180d"])].copy()
    high_breaks_df = high_breaks_df.sort_values(["broke_high_180d", "broke_high_90d", "symbol"], ascending=False)
    return high_breaks_df, all_df


if st.button("Scan now", type="primary"):
    with st.spinner("Scanning Binance futures..."):
        highs_df, all_df = run_scan()

    st.caption(f"Last scan: {_now_utc()} | Universe scanned: up to top {MAX_SYMBOLS_TO_SCAN} USDT perps by 24h quote volume")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pairs scanned", int(len(all_df)))
    c2.metric("High breakouts (24h)", int(len(highs_df)))
    c3.metric("180D high breaks", int(highs_df["broke_high_180d"].sum()) if not highs_df.empty else 0)

    st.subheader("Pairs that broke 90D/180D highs in the last 24h")
    if highs_df.empty:
        st.info("No 90D/180D upside breakouts detected in the scanned universe.")
    else:
        display_cols = [
            "symbol",
            "base_asset",
            "last_price",
            "high_24h",
            "high_90d",
            "high_180d",
            "broke_high_90d",
            "broke_high_180d",
        ]
        st.dataframe(highs_df[display_cols], use_container_width=True, hide_index=True)

    with st.expander("Show full scanned table"):
        st.dataframe(all_df, use_container_width=True, hide_index=True)
else:
    st.markdown('<div class="card">Click <b>Scan now</b> to fetch Binance data once and show 24h high breakouts.</div>', unsafe_allow_html=True)
