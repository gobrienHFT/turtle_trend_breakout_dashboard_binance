from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


class BinanceHTTPError(RuntimeError):
    def __init__(self, status_code: int, payload: Any, url: str):
        self.status_code = int(status_code)
        self.payload = payload
        self.url = url
        super().__init__(f"Binance HTTP {self.status_code} on {url}: {payload}")


@dataclass(frozen=True)
class FuturesSymbol:
    symbol: str
    base_asset: str
    quote_asset: str


class BinanceFuturesPublic:
    """Small public-only client with gentle pacing to avoid bans."""

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout: int = 12,
        requests_per_second: float = 4.0,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = int(timeout)
        self.retries = int(max(1, retries))
        self.min_gap = 1.0 / max(0.5, float(requests_per_second))
        self._last_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "simple-breakout-dashboard/1.0"})

    def _pace(self) -> None:
        now = time.monotonic()
        gap = now - self._last_at
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)
        self._last_at = time.monotonic()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                response = self.session.get(url, params=params or {}, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json() if response.text else None

                payload: Any
                try:
                    payload = response.json()
                except Exception:
                    payload = {"text": response.text[:400]}

                if response.status_code in (418, 429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                    continue
                raise BinanceHTTPError(response.status_code, payload, url)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(min(8.0, 0.8 * (2 ** (attempt - 1))))
                else:
                    break

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Request failed without exception")

    def perpetual_usdt_symbols(self) -> list[FuturesSymbol]:
        info = self._get("/fapi/v1/exchangeInfo")
        rows: list[FuturesSymbol] = []
        for item in info.get("symbols", []):
            if str(item.get("contractType", "")).upper() != "PERPETUAL":
                continue
            if str(item.get("status", "")).upper() != "TRADING":
                continue
            if str(item.get("quoteAsset", "")).upper() != "USDT":
                continue

            symbol = str(item.get("symbol", "")).upper()
            if symbol:
                rows.append(
                    FuturesSymbol(
                        symbol=symbol,
                        base_asset=str(item.get("baseAsset", "")).upper(),
                        quote_asset="USDT",
                    )
                )
        return rows

    def ticker_24hr(self) -> list[dict[str, Any]]:
        data = self._get("/fapi/v1/ticker/24hr")
        return data if isinstance(data, list) else []

    def klines_1d(self, symbol: str, limit: int = 200) -> list[list[Any]]:
        data = self._get(
            "/fapi/v1/klines",
            {"symbol": symbol, "interval": "1d", "limit": int(limit)},
        )
        return data if isinstance(data, list) else []
