from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import threading
import re

import requests


class BinanceHTTPError(RuntimeError):
    def __init__(self, status: int, payload: Any, url: str):
        super().__init__(f"HTTP {status} {payload}")
        self.status = int(status)
        self.payload = payload
        self.url = url


@dataclass(frozen=True)
class FuturesSymbol:
    symbol: str
    quote_asset: str
    base_asset: str


@dataclass(frozen=True)
class BanInfo:
    banned_until_ms: int


class BinanceFuturesPublic:
    """
    Low-churn Binance USD-M public client.
    - request pacing via requests_per_sec
    - light retries only
    - helpers for exchange info, 24h ticker, and daily klines
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout: int = 10,
        requests_per_sec: float = 2.0,
        retries: int = 1,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = int(timeout)
        self.requests_per_sec = float(requests_per_sec)
        self.retries = max(1, int(retries))

        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "simple-perp-breakouts/3.0"})

        self._lock = threading.Lock()
        self._last_req_at = 0.0

    def _pace(self) -> None:
        if self.requests_per_sec <= 0:
            return
        with self._lock:
            min_interval = 1.0 / self.requests_per_sec
            now = time.monotonic()
            wait = min_interval - (now - self._last_req_at)
            if wait > 0:
                time.sleep(wait)
            self._last_req_at = time.monotonic()

    @staticmethod
    def parse_ban_info(err_text: str) -> Optional[BanInfo]:
        m = re.search(r"banned until\s+(\d{10,16})", str(err_text))
        if not m:
            return None
        try:
            return BanInfo(banned_until_ms=int(m.group(1)))
        except Exception:
            return None

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base_url + path
        params = params or {}

        backoff = 1.0
        last_exc: Optional[Exception] = None
        for _ in range(self.retries):
            try:
                self._pace()
                r = self.s.get(url, params=params, timeout=self.timeout)

                if r.status_code == 200:
                    if not r.text:
                        return None
                    try:
                        return r.json()
                    except Exception:
                        return r.text

                payload: Any
                try:
                    payload = r.json()
                except Exception:
                    payload = {"text": r.text[:400]}

                if r.status_code in (418, 429, 500, 502, 503, 504):
                    last_exc = BinanceHTTPError(r.status_code, payload, url)
                    time.sleep(backoff)
                    backoff = min(8.0, backoff * 2.0)
                    continue

                raise BinanceHTTPError(r.status_code, payload, url)

            except requests.RequestException as e:
                last_exc = e
                time.sleep(backoff)
                backoff = min(8.0, backoff * 2.0)

        if last_exc:
            raise last_exc
        raise RuntimeError("request failed")

    def exchange_info(self) -> Dict[str, Any]:
        x = self._get("/fapi/v1/exchangeInfo")
        return x if isinstance(x, dict) else {}

    def ticker_24hr(self) -> List[Dict[str, Any]]:
        x = self._get("/fapi/v1/ticker/24hr")
        return x if isinstance(x, list) else []

    def klines_1d(self, symbol: str, limit: int = 210) -> List[List[Any]]:
        x = self._get("/fapi/v1/klines", {"symbol": symbol, "interval": "1d", "limit": int(limit)})
        return x if isinstance(x, list) else []

    def perpetual_symbols(self) -> List[FuturesSymbol]:
        ex = self.exchange_info()
        out: List[FuturesSymbol] = []
        for s in ex.get("symbols", []) or []:
            try:
                if str(s.get("contractType", "")).upper() != "PERPETUAL":
                    continue
                if str(s.get("status", "")).upper() != "TRADING":
                    continue
                sym = str(s.get("symbol", "")).upper()
                if not sym:
                    continue
                out.append(
                    FuturesSymbol(
                        symbol=sym,
                        quote_asset=str(s.get("quoteAsset", "")).upper(),
                        base_asset=str(s.get("baseAsset", "")).upper(),
                    )
                )
            except Exception:
                continue
        return out
