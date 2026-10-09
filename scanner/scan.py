#!/usr/bin/env python3
"""
ta-screener — Phase 1: personal crypto TA screener (standalone).

Pipeline:
  1. GET /api/v3/exchangeInfo          -> USDT spot, TRADING symbols
  2. GET /api/v3/ticker/24hr          -> price + 24h change (1 call)
  3. GET /api/v3/klines x (symbols x 6 TF) -> 250 candles each
  4. Indicators per symbol x timeframe -> composite score -100..+100
  5. Write data/screener.json

Score design (documented weights):
  Per-indicator signals are normalized to [-1, +1] (+1 = bullish, -1 = bearish):
    RSI(14)        w=0.20  score = (50 - rsi) / 50            (oversold -> +)
    MACD(12,26,9)  w=0.20  score = clip(hist / max|hist|_50, -1, 1)
    ADX(14) dir    w=0.15  score = sign(+DI - -DI) * min(adx,50)/50
    Bollinger(20,2)w=0.15  score = clip((0.5 - %B) * 2, -1, 1) (near lower -> +)
    Supertrend(10,3) w=0.15 score = +1 if close >= st else -1
    StochRSI(14)   w=0.10  score = (50 - %K) / 50
    OBV slope      w=0.05  score = sign(slope_10) * min(|slope|/ref, 1)
  Composite = sum(w * s) * 100  ->  [-100, +100]

  MTF (multi-timeframe) weights:
    15m: 0.10, 1h: 0.15, 4h: 0.20, 1d: 0.25, 3d: 0.15, 1w: 0.15

Scores are screening references only, not trading advice.
"""

import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- config
BASE_URL = "https://data-api.binance.vision"
TIMEFRAMES = ["15m", "1h", "4h", "1d", "3d", "1w"]
KLINES_LIMIT = 250
MIN_BARS = 60          # skip symbols with fewer bars than this
MAX_WORKERS = 10
TARGET_RPS = 10        # pacing to stay under public weight limits
MAX_RETRIES = 4

INDICATOR_WEIGHTS = {
    "rsi": 0.20,
    "macd": 0.20,
    "adx_dir": 0.15,
    "bollinger": 0.15,
    "supertrend": 0.15,
    "stochrsi": 0.10,
    "obv": 0.05,
}
MTF_WEIGHTS = {"15m": 0.10, "1h": 0.15, "4h": 0.20, "1d": 0.25, "3d": 0.15, "1w": 0.15}

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "screener.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ta-screener")

# ---------------------------------------------------------------- rate limiting
_api_calls = 0
_api_lock = threading.Lock()
_last_req = [0.0]
_req_lock = threading.Lock()


def _pace():
    with _req_lock:
        wait = (1.0 / TARGET_RPS) - (time.time() - _last_req[0])
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


def api_get(path, params=None):
    """GET with pacing + exponential-backoff retry. Returns parsed JSON."""
    global _api_calls
    url = BASE_URL + path
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        _pace()
        try:
            with _api_lock:
                _api_calls += 1
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 418) or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else backoff
                log.warning("HTTP %s on %s (attempt %d/%d), sleeping %.1fs",
                            r.status_code, path, attempt, MAX_RETRIES, sleep_s)
                time.sleep(sleep_s)
                backoff *= 2
                continue
            r.raise_for_status()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            log.warning("request error %s (attempt %d/%d), sleeping %.1fs",
                        e, attempt, MAX_RETRIES, backoff)
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"GET {path} failed after {MAX_RETRIES} attempts")


# ---------------------------------------------------------------- indicators
def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    rs = _wilder(gain, n) / _wilder(loss, n).replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=signal, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n).replace(0, np.nan)
    plus_di = 100 * _wilder(plus_dm, n) / atr
    minus_di = 100 * _wilder(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


def bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = ma + k * sd, ma - k * sd
    return (close - lower) / (upper - lower).replace(0, np.nan)


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               n: int = 10, mult: float = 3.0) -> pd.Series:
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    hl2 = (high + low) / 2
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    st = pd.Series(index=close.index, dtype=float)
    direction = True  # True = up
    prev_ub = prev_lb = prev_st = np.nan
    for i in range(len(close)):
        c = close.iloc[i]
        cub, clb = ub.iloc[i], lb.iloc[i]
        if np.isnan(cub):
            continue
        if np.isnan(prev_ub):
            prev_ub, prev_lb = cub, clb
            prev_st = clb if c >= clb else cub
            direction = c >= clb
        else:
            cub = min(cub, prev_ub) if not np.isnan(prev_ub) else cub
            clb = max(clb, prev_lb) if not np.isnan(prev_lb) else clb
            if direction:  # was up
                if c <= clb:
                    direction = False
                    prev_st = cub
                else:
                    prev_st = clb
            else:  # was down
                if c >= cub:
                    direction = True
                    prev_st = clb
                else:
                    prev_st = cub
        st.iloc[i] = prev_st
        prev_ub, prev_lb = cub, clb
    return st


def stochrsi_k(close: pd.Series, n: int = 14) -> pd.Series:
    r = rsi(close, n)
    lo = r.rolling(n).min()
    hi = r.rolling(n).max()
    return 100 * (r - lo) / (hi - lo).replace(0, np.nan)


def obv_slope(close: pd.Series, volume: pd.Series, lookback: int = 10) -> float:
    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    if len(obv) <= lookback:
        return 0.0
    slope = obv.iloc[-1] - obv.iloc[-1 - lookback]
    ref = obv.diff().abs().rolling(lookback).mean().iloc[-1]
    if not np.isfinite(ref) or ref == 0:
        return 0.0
    return float(np.clip(slope / (ref * lookback), -1, 1))


def clip01(x):
    return max(-1.0, min(1.0, x))


def score_timeframe(df: pd.DataFrame):
    """Return (composite_score, indicator_dict) for one symbol x timeframe."""
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    r = rsi(close).iloc[-1]
    h = macd_hist(close)
    hist = h.iloc[-1]
    hist_ref = h.tail(50).abs().max()
    a, pdi, mdi = adx(high, low, close)
    adx_v, pdi_v, mdi_v = a.iloc[-1], pdi.iloc[-1], mdi.iloc[-1]
    pctb = bollinger_pctb(close).iloc[-1]
    st = supertrend(high, low, close).iloc[-1]
    k = stochrsi_k(close).iloc[-1]
    vol_ratio = float(vol.iloc[-1] / vol.iloc[-21:-1].mean()) if len(vol) > 21 else 1.0

    s_rsi = clip01((50 - r) / 50) if np.isfinite(r) else 0.0
    s_macd = clip01(hist / hist_ref) if np.isfinite(hist) and hist_ref and hist_ref > 0 else 0.0
    if np.isfinite(adx_v) and np.isfinite(pdi_v) and np.isfinite(mdi_v):
        s_adx = (1.0 if pdi_v > mdi_v else -1.0) * min(adx_v, 50) / 50
    else:
        s_adx = 0.0
    s_bb = clip01((0.5 - pctb) * 2) if np.isfinite(pctb) else 0.0
    s_st = 1.0 if np.isfinite(st) and close.iloc[-1] >= st else -1.0
    s_sk = clip01((50 - k) / 50) if np.isfinite(k) else 0.0
    s_obv = obv_slope(close, vol)

    parts = {"rsi": s_rsi, "macd": s_macd, "adx_dir": s_adx, "bollinger": s_bb,
             "supertrend": s_st, "stochrsi": s_sk, "obv": s_obv}
    composite = sum(INDICATOR_WEIGHTS[k] * v for k, v in parts.items()) * 100
    if not np.isfinite(composite):
        raise ValueError("non-finite composite")
    ind = {
        "rsi": round(float(r), 2) if np.isfinite(r) else None,
        "macd_h": round(float(hist), 6) if np.isfinite(hist) else None,
        "adx": round(float(adx_v), 2) if np.isfinite(adx_v) else None,
        "bb_pctb": round(float(pctb), 3) if np.isfinite(pctb) else None,
        "st_dir": "up" if s_st > 0 else "down",
        "stochrsi_k": round(float(k), 2) if np.isfinite(k) else None,
        "vol_ratio": round(vol_ratio, 2),
    }
    return round(float(composite), 1), ind


# ---------------------------------------------------------------- scan
def get_symbols():
    info = api_get("/api/v3/exchangeInfo")
    out = []
    for s in info["symbols"]:
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING" or not s.get("isSpotTradingAllowed"):
            continue
        sym = s["symbol"]
        if sym.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
            continue  # leveraged tokens
        out.append(sym)
    return sorted(out)


def klines_df(symbol, tf):
    raw = api_get("/api/v3/klines",
                  {"symbol": symbol, "interval": tf, "limit": KLINES_LIMIT})
    if len(raw) < MIN_BARS:
        raise ValueError(f"only {len(raw)} bars")
    df = pd.DataFrame(raw, columns=[
        "ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "x"])
    return pd.DataFrame({
        "close": df["c"].astype(float),
        "high": df["h"].astype(float),
        "low": df["l"].astype(float),
        "volume": df["v"].astype(float),
    })


def scan_symbol(symbol):
    tf_scores, tf_inds = {}, {}
    for tf in TIMEFRAMES:
        df = klines_df(symbol, tf)
        score, ind = score_timeframe(df)
        tf_scores[tf] = score
        tf_inds[tf] = ind
    mtf = round(sum(MTF_WEIGHTS[tf] * tf_scores[tf] for tf in TIMEFRAMES), 1)
    return tf_scores, tf_inds, mtf


def main():
    t0 = time.time()
    log.info("fetching symbols...")
    symbols = get_symbols()
    log.info("%d USDT spot symbols", len(symbols))

    log.info("fetching 24h tickers...")
    tickers = {t["symbol"]: t for t in api_get("/api/v3/ticker/24hr")
               if t["symbol"] in set(symbols)}

    results, failures = [], []
    done = [0]

    def work(sym):
        try:
            tf_scores, tf_inds, mtf = scan_symbol(sym)
            t = tickers.get(sym, {})
            results.append({
                "s": sym,
                "p": float(t.get("lastPrice", 0)) or None,
                "c24": round(float(t.get("priceChangePercent", 0)), 2),
                "tf": {tf: {"score": tf_scores[tf], **tf_inds[tf]} for tf in TIMEFRAMES},
                "mtf": mtf,
            })
        except Exception as e:  # one symbol must never kill the scan
            failures.append({"symbol": sym, "error": str(e)[:120]})
        done[0] += 1
        if done[0] % 50 == 0:
            log.info("progress %d/%d (%.1f%%)", done[0], len(symbols),
                     100 * done[0] / len(symbols))

    log.info("scanning %d symbols x %d timeframes...", len(symbols), len(TIMEFRAMES))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(work, symbols))

    results.sort(key=lambda r: r["mtf"], reverse=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exchange": "binance",
        "market": "spot_usdt",
        "timeframes": TIMEFRAMES,
        "indicator_weights": INDICATOR_WEIGHTS,
        "mtf_weights": MTF_WEIGHTS,
        "symbol_count": len(results),
        "failed_count": len(failures),
        "failures": failures[:50],
        "scan_duration_sec": round(time.time() - t0, 1),
        "api_calls": _api_calls,
        "disclaimer": "Scores are screening references only, not trading advice. "
                      "점수는 참고용 스크리닝 점수이며 매매 권유가 아닙니다.",
        "symbols": results,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)
    dur = time.time() - t0
    log.info("done: %d ok, %d failed, %d api calls, %.1fs",
             len(results), len(failures), _api_calls, dur)
    if failures:
        log.warning("sample failures: %s", failures[:5])
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
