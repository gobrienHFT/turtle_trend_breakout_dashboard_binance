#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import random
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
    contract_type: str
    status: str
    quote_asset: str
    base_asset: str


class BinanceFuturesPublic:
    def __init__(self, base_url: str = "https://fapi.binance.com", timeout: int = 10):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = int(timeout)
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "pf-breakouts/1.0"})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, tries: int = 6) -> Any:
        url = self.base_url + path
        params = params or {}
        last_err: Optional[Exception] = None

        for a in range(1, tries + 1):
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    sleep_s = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(20.0, 0.8 * (1.8 ** (a - 1)))
                    time.sleep(sleep_s)
                    continue
                if r.status_code >= 400:
                    try:
                        payload = r.json()
                    except Exception:
                        payload = {"text": r.text[:500]}
                    raise BinanceHTTPError(r.status_code, payload, url)
                if not r.text:
                    return None
                try:
                    return r.json()
                except Exception:
                    return r.text
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                if a >= tries:
                    break
                dly = min(15.0, 0.5 * (1.8 ** (a - 1))) + random.uniform(0.0, 0.25)
                time.sleep(dly)
            except BinanceHTTPError as e:
                last_err = e
                if e.status in (418, 500, 502, 503, 504) and a < tries:
                    dly = min(15.0, 0.5 * (1.8 ** (a - 1))) + random.uniform(0.0, 0.25)
                    time.sleep(dly)
                    continue
                raise

        if last_err:
            raise last_err
        raise RuntimeError("request failed")

    def exchange_info(self) -> Dict[str, Any]:
        return self._get("/fapi/v1/exchangeInfo")

    def ticker_24hr(self) -> List[Dict[str, Any]]:
        x = self._get("/fapi/v1/ticker/24hr")
        return x if isinstance(x, list) else []

    def premium_index(self) -> List[Dict[str, Any]]:
        x = self._get("/fapi/v1/premiumIndex")
        return x if isinstance(x, list) else []

    def klines(self, symbol: str, interval: str = "1d", limit: int = 200) -> List[List[Any]]:
        p = {"symbol": symbol, "interval": interval, "limit": int(limit)}
        x = self._get("/fapi/v1/klines", p)
        return x if isinstance(x, list) else []

    def list_perpetuals(self) -> List[FuturesSymbol]:
        ex = self.exchange_info()
        out: List[FuturesSymbol] = []
        for s in ex.get("symbols", []) or []:
            try:
                sym = str(s.get("symbol", "")).upper()
                if not sym:
                    continue
                out.append(
                    FuturesSymbol(
                        symbol=sym,
                        contract_type=str(s.get("contractType", "")).upper(),
                        status=str(s.get("status", "")).upper(),
                        quote_asset=str(s.get("quoteAsset", "")).upper(),
                        base_asset=str(s.get("baseAsset", "")).upper(),
                    )
                )
            except Exception:
                continue
        return out
