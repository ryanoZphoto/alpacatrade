import os
import uuid
import math
import asyncio
import logging
import statistics
from collections import deque
from typing import Deque, Dict, List, Optional
from datetime import datetime, timezone, date, timedelta

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# -------------------------------------------------
# Env
# -------------------------------------------------
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER_URL = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
CB_THRESHOLD_PCT = float(os.getenv("CIRCUIT_BREAKER_PCT", "0.7"))

if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    raise RuntimeError("Missing Alpaca credentials – edit .env")

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.FileHandler("ladder_bot.log"), logging.StreamHandler()],
)
log = logging.getLogger("ladder")

# -------------------------------------------------
# App
# -------------------------------------------------
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):  # type: ignore[override]
        resp = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers["Cache-Control"] = "no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp

app = FastAPI(title="Alpaca Ladder Bot UI")
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Models
# -------------------------------------------------
class BarParams(BaseModel):
    symbols: str = Field(..., description="comma-separated, e.g. BTCUSD")
    timeframe: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    limit: Optional[int] = 1000
    sort: Optional[str] = "asc"

class LadderConfig(BaseModel):
    symbol: str
    direction: str
    steps: int = Field(..., ge=1, le=20)
    interval: float = Field(..., gt=0)
    size: float = Field(..., gt=0)
    max_exposure: float = Field(..., gt=0)

    @validator("direction")
    def dir_must_be(cls, v):
        v = v.upper()
        if v not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        return v

class AutopilotConfig(BaseModel):
    symbol: str = Field(..., description="e.g. BTC/USD")
    fast_window: int = Field(12, ge=3, le=60)
    slow_window: int = Field(26, ge=5, le=240)
    rsi_window: int = Field(14, ge=5, le=240)
    overbought: float = Field(70.0, gt=50.0, lt=100.0)
    oversold: float = Field(30.0, gt=0.0, lt=50.0)
    base_interval: float = Field(150.0, gt=0.0)
    base_steps: int = Field(7, ge=3, le=20)
    rung_notional: float = Field(..., gt=0.0)
    max_notional: float = Field(..., gt=0.0)
    volatility_lookback: int = Field(60, ge=10, le=500)
    risk_multiplier: float = Field(1.0, gt=0.1, le=5.0)
    poll_seconds: float = Field(30.0, ge=10.0, le=120.0)

    @validator("slow_window")
    def slow_gt_fast(cls, v, values):
        fast = values.get("fast_window")
        if fast and v <= fast:
            raise ValueError("slow_window must be greater than fast_window")
        return v

    @validator("overbought")
    def ob_gt_os(cls, v, values):
        os_ = values.get("oversold")
        if os_ and v <= os_:
            raise ValueError("overbought must exceed oversold")
        return v

class NudgeRequest(BaseModel):
    direction: str

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def normalize_symbol(symbol: str) -> str:
    s = symbol.replace("-", "/").upper()
    if "/" not in s and len(s) > 3:
        s = f"{s[:-3]}/{s[-3:]}"
    return s

def _pick_series(symbol: str, container) -> List[dict]:
    if isinstance(container, dict):
        normalized = normalize_symbol(symbol)
        candidates = [
            normalized,
            symbol,
            symbol.replace("-", "/"),
            normalized.replace("/", ""),
            normalized.replace("/", "-"),
        ]
        for k in candidates:
            if k in container:
                return container[k]
        return next(iter(container.values()), [])
    return container

async def fetch_crypto_bars(
    client: httpx.AsyncClient,
    symbol: str,
    *,
    limit: int,
    timeframe: str = "1Min",
) -> List[Dict[str, float]]:
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    params = {"symbols": normalize_symbol(symbol), "timeframe": timeframe, "limit": limit, "sort": "desc"}
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    # simple retry/backoff for transient errors / 429
    last_exc = None
    for attempt in range(3):
        r = await client.get(endpoint, params=params, headers=headers)
        if r.status_code == 200:
            break
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(0.5 * (2 ** attempt))
            last_exc = HTTPException(status_code=r.status_code, detail=r.text)
            continue
    r.raise_for_status()
    if r.status_code != 200:
        raise last_exc or HTTPException(status_code=r.status_code, detail=r.text)
    data = r.json()
    series = _pick_series(symbol, data.get("bars", []))
    if not series:
        raise RuntimeError(f"No bar data for {symbol}")
    out: List[Dict[str, float]] = []
    for raw in reversed(series):
        out.append({
            "t": raw.get("t") or raw.get("timestamp"),
            "o": float(raw.get("o", raw.get("open"))),
            "h": float(raw.get("h", raw.get("high", raw.get("c", 0)))),
            "l": float(raw.get("l", raw.get("low", raw.get("c", 0)))),
            "c": float(raw.get("c", raw.get("close"))),
        })
    return out[-limit:]

async def get_latest_price(client: httpx.AsyncClient, symbol: str) -> float:
    bars = await fetch_crypto_bars(client, symbol, limit=1)
    return float(bars[-1]["c"])

async def get_latest_bar(client: httpx.AsyncClient, symbol: str) -> dict:
    bars = await fetch_crypto_bars(client, symbol, limit=1)
    return bars[-1]

def compute_ema(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    k = 2 / (window + 1)
    ema = values[0]
    for p in values[1:]:
        ema = p * k + ema * (1 - k)
    return ema

def compute_rsi(values: List[float], window: int) -> float:
    if len(values) <= window:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(values)):
        d = values[i] - values[i - 1]
        if d > 0:
            gains.append(d); losses.append(0.0)
        else:
            gains.append(0.0); losses.append(-d)
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_pct_volatility(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    rets = []
    for a, b in zip(values[:-1], values[1:]):
        if a != 0:
            rets.append((b - a) / a)
    if not rets:
        return 0.0
    return statistics.pstdev(rets)

def ladder_configs_close(a: LadderConfig, b: LadderConfig) -> bool:
    return (
        a.symbol == b.symbol
        and a.direction == b.direction
        and math.isclose(a.interval, b.interval, rel_tol=0.05, abs_tol=0.5)
        and math.isclose(a.size, b.size, rel_tol=0.05, abs_tol=1e-6)
        and a.steps == b.steps
        and math.isclose(a.max_exposure, b.max_exposure, rel_tol=0.05, abs_tol=1e-6)
    )

def compute_step_prices(cfg: LadderConfig, price: float) -> List[float]:
    sign = -1 if cfg.direction == "BUY" else 1
    return [round(price + sign * i * cfg.interval, 2) for i in range(cfg.steps)]

# -------------------------------------------------
# Alpaca ops
# -------------------------------------------------
async def list_open_orders(client: httpx.AsyncClient, symbol: str) -> List[dict]:
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    params = {"status": "open", "symbol": normalize_symbol(symbol), "asset_class": "crypto"}
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    for attempt in range(3):
        r = await client.get(endpoint, params=params, headers=headers)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            await asyncio.sleep(0.4 * (2 ** attempt))
            continue
    r.raise_for_status()
    # if still failing, return current snapshot
    return bot_manager.open_orders

async def fetch_account(client: httpx.AsyncClient) -> Dict[str, object]:
    endpoint = f"{ALPACA_PAPER_URL}/v2/account"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    r = await client.get(endpoint, headers=headers)
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except Exception:
        return {}

async def cancel_order(client: httpx.AsyncClient, order_id: str):
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders/{order_id}"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    r = await client.delete(endpoint, headers=headers)
    r.raise_for_status()

async def cancel_all_open_orders(symbol: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        orders = await list_open_orders(client, symbol)
        for o in orders:
            await cancel_order(client, o["id"])
            log.info("Cancelled %s %s@%s", o.get("side"), o.get("qty"), o.get("limit_price"))

async def submit_limit_order(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    side: str,
    qty: float,
    limit_price: float,
    client_order_id: Optional[str] = None,
) -> dict:
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    payload = {
        "symbol": normalize_symbol(symbol),
        "qty": f"{qty:.8f}",
        "side": side.lower(),
        "type": "limit",
        "limit_price": f"{limit_price:.2f}",
        "time_in_force": "gtc",
        "asset_class": "crypto",
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id
    r = await client.post(endpoint, headers=headers, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

async def submit_market_order(
    client: httpx.AsyncClient, *, symbol: str, side: str, qty: float
) -> dict:
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    payload = {
        "symbol": normalize_symbol(symbol),
        "qty": f"{qty:.8f}",
        "side": side.lower(),
        "type": "market",
        "time_in_force": "gtc",
        "asset_class": "crypto",
    }
    r = await client.post(endpoint, headers=headers, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

async def submit_stop_order(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    side: str,
    qty: float,
    stop_price: float,
    client_order_id: Optional[str] = None,
    limit_price: Optional[float] = None,
) -> dict:
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    if limit_price is None:
        limit_price = stop_price
    payload = {
        "symbol": normalize_symbol(symbol),
        "qty": f"{qty:.8f}",
        "side": side.lower(),
        # Alpaca crypto expects stop_limit for protective stop on crypto
        "type": "stop_limit",
        "stop_price": f"{stop_price:.2f}",
        "limit_price": f"{float(limit_price):.2f}",
        "time_in_force": "gtc",
        "asset_class": "crypto",
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id
    r = await client.post(endpoint, headers=headers, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()

async def get_position(client: httpx.AsyncClient, symbol: str) -> float:
    endpoint = f"{ALPACA_PAPER_URL}/v2/positions/{normalize_symbol(symbol)}"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    r = await client.get(endpoint, headers=headers)
    if r.status_code == 404:
        return 0.0
    r.raise_for_status()
    data = r.json()
    qty = float(data.get("qty", 0))
    # Crypto positions return qty as positive for long, negative for short
    return qty

# -------------------------------------------------
# Bot
# -------------------------------------------------
class BotManager:
    def __init__(self):
        self.task: Optional[asyncio.Task] = None
        self.cfg: Optional[LadderConfig] = None
        self.position_qty: float = 0.0
        self.avg_price: float = 0.0
        self.pnl: float = 0.0
        self.open_orders: List[dict] = []
        self.filled_orders: List[dict] = []
        self._lock = asyncio.Lock()
        self.last_error: Optional[str] = None
        self.manual_price: Optional[float] = None
        self.last_action: Optional[str] = None

    async def start(self, cfg: LadderConfig):
        async with self._lock:
            if self.task and not self.task.done():
                raise RuntimeError("Bot already running")
            self.cfg = cfg
            self.last_error = None
            self.last_action = "Ladder started"
            self.task = asyncio.create_task(self._run())

    async def stop(self):
        async with self._lock:
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            self.task = None
        if self.cfg:
            await cancel_all_open_orders(self.cfg.symbol)

    async def _run(self):
        assert self.cfg is not None
        cfg = self.cfg
        client = httpx.AsyncClient(timeout=10.0)
        try:
            while True:
                # 1) circuit breaker on 1m range
                bar = await get_latest_bar(client, cfg.symbol)
                price = float(bar["c"]); high = float(bar["h"]); low = float(bar["l"])
                pct_range = ((high - low) / max(price, 1e-9)) * 100.0
                if pct_range >= CB_THRESHOLD_PCT:
                    await cancel_all_open_orders(cfg.symbol)
                    self.last_error = f"Circuit breaker: 1m range {pct_range:.2f}% ≥ {CB_THRESHOLD_PCT:.2f}%"
                    self.last_action = "Circuit breaker tripped; stopped and cancelled rungs"
                    log.info("Circuit breaker tripped: price=%.2f high=%.2f low=%.2f range=%.2f%% symbol=%s", price, high, low, pct_range, cfg.symbol)
                    break

                # 2) sync ladder
                center = self.manual_price if self.manual_price is not None else price
                log.info("price %s: %.3f", cfg.symbol, price)
                await sync_ladder(client, cfg, center)

                # 3) fills and position
                await update_fills_and_position(client, cfg)

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self.last_error = str(exc)
            log.exception("Bot crashed")
            raise
        finally:
            await client.aclose()

bot_manager = BotManager()

async def sync_ladder(client: httpx.AsyncClient, cfg: LadderConfig, center_price: float):
    desired_prices = compute_step_prices(cfg, center_price)
    side = "buy" if cfg.direction == "BUY" else "sell"
    # sizing
    # cfg.size is asset qty per rung. enforce max_exposure
    current_pos = bot_manager.position_qty
    max_exposure = cfg.max_exposure
    if side == "sell":
        # Spot-only: only place SELL rungs against existing inventory
        remaining = max(0.0, current_pos)
    else:
        remaining = max(0.0, max_exposure - max(0.0, current_pos))
    rung_qty = min(cfg.size, remaining) if remaining > 0 else 0.0

    # Enforce Alpaca crypto min order size and 9dp precision
    min_qty = max(1.0 / max(center_price, 1e-12), 0.000000002)
    rung_qty = math.floor(max(rung_qty, 0.0) * 1e9) / 1e9
    if rung_qty < min_qty:
        # Nothing to do; below minimum tradable increment
        return

    # For BUYs, ensure sufficient non-marginable buying power
    if side == "buy":
        try:
            acct = await fetch_account(client)
            nmbp = float(acct.get("non_marginable_buying_power", 0) or 0)
            needed = rung_qty * center_price
            if needed > nmbp:
                # reduce to fit buying power
                rung_qty = math.floor(max(nmbp / max(center_price, 1e-12), 0.0) * 1e9) / 1e9
                if rung_qty < min_qty:
                    return
        except Exception:
            pass

    open_orders = await list_open_orders(client, cfg.symbol)
    bot_manager.open_orders = open_orders

    # Map by price to keep one per rung
    existing_by_price: Dict[float, dict] = {}
    for o in open_orders:
        # ignore non-ladder-side orders and take-profit/stop-loss orders
        try:
            cid = str(o.get("client_order_id", ""))
            if cid.startswith("TP-") or cid.startswith("SL-"):
                continue
            if str(o.get("side", "")).lower() != side:
                continue
            lp = o.get("limit_price") or o.get("filled_avg_price")
            p = round(float(lp), 2)
            if p not in existing_by_price:
                existing_by_price[p] = o
        except Exception:
            continue

    # Place or keep desired rungs
    for p in desired_prices:
        if rung_qty <= 0:
            break
        if p in existing_by_price:
            continue
        try:
            await submit_limit_order(client, symbol=cfg.symbol, side=side, qty=rung_qty, limit_price=p)
            log.info("Placed %s %f @ %.2f", side.upper(), rung_qty, p)
        except Exception as exc:
            msg = str(getattr(exc, "detail", exc))
            # price band protection: move price slightly toward last trade and retry once
            if "price band" in msg.lower():
                try:
                    adjust = round(center_price * 0.001, 2)  # ~0.1%
                    p2 = round(min(p + adjust, center_price) if side == "buy" else max(p - adjust, center_price), 2)
                    await submit_limit_order(client, symbol=cfg.symbol, side=side, qty=rung_qty, limit_price=p2)
                    log.info("Repriced due to band: %s %f @ %.2f (from %.2f)", side.upper(), rung_qty, p2, p)
                except Exception as exc2:
                    log.warning("Place failed after band retry @ %.2f: %s", p2, exc2)
            else:
                log.warning("Place failed @ %.2f: %s", p, exc)

    # Cancel strays
    desired_set = set(desired_prices)
    for o in open_orders:
        try:
            cid = str(o.get("client_order_id", ""))
            if cid.startswith("TP-") or cid.startswith("SL-"):
                continue
            if str(o.get("side", "")).lower() != side:
                continue
            p = round(float(o.get("limit_price")), 2)
            if p not in desired_set:
                try:
                    await cancel_order(client, o["id"])
                    log.info("Cancelled stray %s", o["id"])
                except Exception:
                    log.exception("Cancel stray failed")
        except Exception:
            continue
    # Capital footprint snapshot
    try:
        working_notional = sum(
            (float(o.get("qty", 0)) if isinstance(o.get("qty"), (int, float, str)) else 0.0)
            * float(o.get("limit_price") or 0.0) for o in bot_manager.open_orders
        )
        position_notional = abs(bot_manager.position_qty) * center_price
        net_deployed = working_notional + position_notional
        cap = cfg.steps * cfg.size * center_price
        usage = (net_deployed / cap * 100.0) if cap else 0.0
        log.info("Ladder capital: working=%.2f position=%.2f net=%.2f target=%.2f usage=%.2f%%",
                 working_notional, position_notional, net_deployed, cap, usage)
    except Exception:
        pass

async def update_fills_and_position(client: httpx.AsyncClient, cfg: LadderConfig):
    # snapshot position from Alpaca
    qty = await get_position(client, cfg.symbol)
    if qty != bot_manager.position_qty:
        log.info("Position update %s: %.8f", cfg.symbol, qty)
    bot_manager.position_qty = qty

    # pull recently closed orders to infer fills
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    params = {"status": "closed", "symbols": normalize_symbol(cfg.symbol), "limit": 50, "nested": True}
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    r = await client.get(endpoint, params=params, headers=headers)
    if r.status_code == 200:
        closed = r.json()
    else:
        closed = []

    seen_ids = {o.get("id") for o in bot_manager.filled_orders}
    new_fills = [o for o in closed if o.get("filled_avg_price") and o.get("id") not in seen_ids]
    for o in new_fills:
        try:
            side = str(o.get("side", "")).upper()
            qty = float(o.get("filled_qty", o.get("qty", 0)) or 0)
            fill_price = float(o.get("filled_avg_price", o.get("limit_price", 0)) or 0)
        except Exception:
            continue

        prev_pnl = bot_manager.pnl
        q0 = bot_manager.position_qty
        ap0 = bot_manager.avg_price
        realized_delta = 0.0

        if side == "BUY":
            if q0 < 0:  # covering short
                cover = min(qty, -q0)
                realized_delta += (ap0 - fill_price) * cover
                remaining = qty - cover
                new_qty = q0 + cover
                # if still short after covering, avg remains; if flip to long, start new avg at fill_price
                if remaining > 0:
                    new_qty = remaining  # now long
                    bot_manager.avg_price = fill_price
                else:
                    bot_manager.avg_price = ap0 if new_qty < 0 else 0.0
                bot_manager.position_qty = new_qty
                # if additional remaining handled above
            else:  # increasing or opening long
                new_qty = q0 + qty
                if new_qty > 0:
                    bot_manager.avg_price = ((ap0 * q0) + (fill_price * qty)) / max(new_qty, 1e-12)
                else:
                    bot_manager.avg_price = 0.0
                bot_manager.position_qty = new_qty
        elif side == "SELL":
            if q0 > 0:  # closing long
                close = min(qty, q0)
                realized_delta += (fill_price - ap0) * close
                remaining = qty - close
                new_qty = q0 - close
                if remaining > 0:
                    # flip to short with remaining
                    new_qty = -remaining
                    bot_manager.avg_price = fill_price
                else:
                    bot_manager.avg_price = ap0 if new_qty > 0 else 0.0
                bot_manager.position_qty = new_qty
            else:  # increasing or opening short
                new_qty = q0 - qty
                if new_qty < 0:
                    # weighted avg over absolute quantities
                    prev_abs = abs(q0)
                    new_abs = abs(new_qty)
                    bot_manager.avg_price = ((ap0 * prev_abs) + (fill_price * qty)) / max(new_abs, 1e-12)
                else:
                    bot_manager.avg_price = 0.0
                bot_manager.position_qty = new_qty
        else:
            # unknown side; skip accounting
            pass

        bot_manager.pnl += realized_delta
        try:
            o["realized_delta"] = realized_delta
        except Exception:
            pass

        bot_manager.filled_orders.append(o)
        log.info(
            "Processed fill %s: %s %s@%s (realized delta=%.2f)",
            o.get("id"), side, qty, fill_price, bot_manager.pnl - prev_pnl,
        )

    # Refresh open orders snapshot post-fills
    try:
        bot_manager.open_orders = await list_open_orders(client, cfg.symbol)
    except Exception:
        pass

    # Maintain a single take-profit order against the net position
    try:
        await ensure_take_profit(client, cfg)
    except Exception:
        log.exception("ensure_take_profit failed")

    # Maintain a single stop-loss (trailing) against the net position
    try:
        await ensure_stop_loss(client, cfg)
    except Exception:
        log.exception("ensure_stop_loss failed")

async def ensure_take_profit(client: httpx.AsyncClient, cfg: LadderConfig):
    pos = bot_manager.position_qty
    # cancel any existing TP when flat
    if abs(pos) < 1e-8:
        for o in bot_manager.open_orders:
            try:
                if str(o.get("client_order_id", "")).startswith("TP-"):
                    await cancel_order(client, o["id"])
                    log.info("Cancelled TP %s (flat)", o.get("id"))
            except Exception:
                log.exception("Cancel TP failed")
        return

    side = "sell" if pos > 0 else "buy"
    tp_price = round(bot_manager.avg_price + (cfg.interval if pos > 0 else -cfg.interval), 2)
    qty = round(abs(pos), 8)

    # see if an appropriate TP exists
    existing = []
    for o in bot_manager.open_orders:
        try:
            if not str(o.get("client_order_id", "")).startswith("TP-"):
                continue
            existing.append(o)
        except Exception:
            continue

    chosen = None
    for o in existing:
        try:
            ok_side = str(o.get("side", "")).lower() == side
            ok_price = abs(float(o.get("limit_price")) - tp_price) < 0.01
            ok_qty = abs(float(o.get("qty")) - qty) < 1e-8
            if ok_side and ok_price and ok_qty:
                chosen = o
                break
        except Exception:
            continue

    # cancel stale TPs
    for o in existing:
        if o is chosen:
            continue
        try:
            await cancel_order(client, o["id"])
            log.info("Cancelled stale TP %s", o.get("id"))
        except Exception:
            log.exception("Cancel stale TP failed")

    # place TP if missing
    if not chosen:
        await submit_limit_order(
            client,
            symbol=cfg.symbol,
            side=side,
            qty=qty,
            limit_price=tp_price,
            client_order_id=f"TP-{normalize_symbol(cfg.symbol)}-{uuid.uuid4().hex[:8]}",
        )
        log.info("Placed TP %s %s @ %.2f", side.upper(), f"{qty:.8f}", tp_price)

async def ensure_stop_loss(client: httpx.AsyncClient, cfg: LadderConfig):
    pos = bot_manager.position_qty
    if abs(pos) < 1e-8:
        # cancel any existing SL when flat
        for o in bot_manager.open_orders:
            try:
                if str(o.get("client_order_id", "")).startswith("SL-"):
                    await cancel_order(client, o["id"])
                    log.info("Cancelled SL %s (flat)", o.get("id"))
            except Exception:
                log.exception("Cancel SL failed")
        return

    # Trailing stop: ratchet in favorable direction based on latest price
    last = await get_latest_price(client, cfg.symbol)
    sl_mult = 1.5  # trailing distance in intervals
    if pos > 0:
        side = "sell"
        candidate = round(last - sl_mult * cfg.interval, 2)
    else:
        side = "buy"
        candidate = round(last + sl_mult * cfg.interval, 2)

    qty = round(abs(pos), 8)
    existing = []
    for o in bot_manager.open_orders:
        try:
            if not str(o.get("client_order_id", "")).startswith("SL-"):
                continue
            existing.append(o)
        except Exception:
            continue

    chosen = None
    for o in existing:
        try:
            ok_side = str(o.get("side", "")).lower() == side
            ok_qty = abs(float(o.get("qty")) - qty) < 1e-8
            if not ok_side or not ok_qty:
                continue
            # stop orders expose stop or stop_price
            sp = float(o.get("stop_price") or o.get("stop", 0.0) or 0.0)
            if pos > 0:
                # never lower the stop for longs
                candidate = max(candidate, sp) if sp else candidate
            else:
                # never raise the stop for shorts
                candidate = min(candidate, sp) if sp else candidate
            chosen = o
            break
        except Exception:
            continue

    # cancel stale SLs (wrong side/qty)
    for o in existing:
        if o is chosen:
            continue
        try:
            await cancel_order(client, o["id"])
            log.info("Cancelled stale SL %s", o.get("id"))
        except Exception:
            log.exception("Cancel stale SL failed")

    # place or update SL
    if chosen:
        # If candidate tightened, replace
        try:
            sp = float(chosen.get("stop_price") or chosen.get("stop", 0.0) or 0.0)
        except Exception:
            sp = 0.0
        need_update = (pos > 0 and candidate > sp + 1e-6) or (pos < 0 and candidate < sp - 1e-6)
        if need_update:
            try:
                await cancel_order(client, chosen["id"])
            except Exception:
                log.exception("Cancel old SL failed")
            await submit_stop_order(
                client,
                symbol=cfg.symbol,
                side=side,
                qty=qty,
                stop_price=candidate,
                client_order_id=f"SL-{normalize_symbol(cfg.symbol)}-{uuid.uuid4().hex[:8]}",
            )
            log.info("Updated SL %s %s @ %.2f", side.upper(), f"{qty:.8f}", candidate)
    else:
        await submit_stop_order(
            client,
            symbol=cfg.symbol,
            side=side,
            qty=qty,
            stop_price=candidate,
            client_order_id=f"SL-{normalize_symbol(cfg.symbol)}-{uuid.uuid4().hex[:8]}",
        )
        log.info("Placed SL %s %s @ %.2f", side.upper(), f"{qty:.8f}", candidate)

# -------------------------------------------------
# Autopilot
# -------------------------------------------------
class AutopilotManager:
    def __init__(self):
        self.task: Optional[asyncio.Task] = None
        self.cfg: Optional[AutopilotConfig] = None
        self.last_signal: Optional[str] = None
        self.last_decision: Dict[str, float] = {}
        self.last_error: Optional[str] = None
        self.last_reason: Optional[str] = None
        self.last_run: Optional[datetime] = None
        self.applied_config: Optional[LadderConfig] = None
        self._lock = asyncio.Lock()
        self.history: Deque[Dict[str, object]] = deque(maxlen=60)

    async def _record(self, action: str, note: str, **metrics):
        e: Dict[str, object] = {"ts": datetime.now(timezone.utc).isoformat(), "action": action, "note": note}
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, float):
                if k in {"interval", "price", "size_notional", "max_notional"}:
                    e[k] = round(v, 2)
                elif k in {"size_asset"}:
                    e[k] = round(v, 6)
                else:
                    e[k] = round(v, 2)
            else:
                e[k] = v
        async with self._lock:
            self.history.append(e)
        try:
            log.info("Autopilot %s: %s | %s", action, note, {k:metrics[k] for k in sorted(metrics.keys())})
        except Exception:
            log.info("Autopilot %s: %s", action, note)

    async def start(self, cfg: AutopilotConfig):
        async with self._lock:
            if self.task and not self.task.done():
                raise RuntimeError("Autopilot already running")
            await bot_manager.stop()
            self.cfg = cfg
            self.last_signal = None
            self.last_decision = {}
            self.last_error = None
            self.last_reason = None
            self.last_run = None
            self.applied_config = None
            self.history.clear()
            self.task = asyncio.create_task(self._run())
        await self._record("Started", note=f"Poll every {cfg.poll_seconds:.0f}s on {cfg.symbol}", symbol=cfg.symbol)
        log.info("Autopilot started")

    async def stop(self):
        async with self._lock:
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            self.task = None
            self.last_signal = "stopped"
            self.applied_config = None
        await bot_manager.stop()
        await self._record("Stopped", note="Autopilot stopped and ladder halted")
        log.info("Autopilot stopped")

    async def _run(self):
        assert self.cfg is not None
        cfg = self.cfg
        lookback = max(cfg.slow_window, cfg.rsi_window, cfg.volatility_lookback) + 5
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    hist = await fetch_crypto_bars(client, cfg.symbol, limit=lookback)
                    closes = [b["c"] for b in hist]
                    if len(closes) < lookback:
                        raise RuntimeError("Insufficient history for strategy")
                    fast = compute_ema(closes[-cfg.fast_window - 1 :], cfg.fast_window)
                    slow = compute_ema(closes[-cfg.slow_window - 1 :], cfg.slow_window)
                    rsi = compute_rsi(closes, cfg.rsi_window)
                    trend_strength = (fast - slow) / slow if slow else 0.0
                    vol = compute_pct_volatility(closes[-(cfg.volatility_lookback + 1) :])
                    direction, reason = self._signal(fast, slow, rsi, cfg, vol)
                    await self._apply(direction=direction, price=closes[-1], volatility=vol,
                                      trend_strength=trend_strength, rsi=rsi, fast=fast, slow=slow, reason=reason)
                    async with self._lock:
                        self.last_signal = direction or "HOLD"
                        self.last_reason = reason
                        self.last_run = datetime.now(timezone.utc)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.exception("Autopilot error")
                    async with self._lock:
                        self.last_error = str(exc)
                    raise
                await asyncio.sleep(cfg.poll_seconds)

    def _signal(self, fast: float, slow: float, rsi: float, cfg: AutopilotConfig, volatility: float):
        # Adaptive gap: require EMA divergence > max(0.03%, 0.4 × 1m volatility)
        if slow == 0:
            return None, "Slow EMA is zero; skip"
        gap = max(0.0003, 0.4 * max(volatility, 0.0))  # expressed as fraction
        up_ok = fast > slow * (1.0 + gap) and rsi < cfg.overbought
        dn_ok = fast < slow * (1.0 - gap) and rsi > cfg.oversold
        if up_ok:
            return "BUY", f"Trend up (gap>{gap*100:.2f}%), RSI {rsi:.1f}"
        if dn_ok:
            return "SELL", f"Trend down (gap>{gap*100:.2f}%), RSI {rsi:.1f}"
        # Fallback: choose weak direction by EMA slope when gap threshold not met
        if fast > slow:
            return "BUY", f"Weak uptrend (gap<{gap*100:.2f}%) | RSI {rsi:.1f}"
        if fast < slow:
            return "SELL", f"Weak downtrend (gap<{gap*100:.2f}%) | RSI {rsi:.1f}"
        return None, f"Neutral (gap<{gap*100:.2f}%) | RSI {rsi:.1f}"

    async def _apply(self, *, direction: Optional[str], price: float, volatility: float,
                     trend_strength: float, rsi: float, fast: float, slow: float, reason: str):
        cfg = self.cfg
        if cfg is None:
            return
        vol_pct = volatility * 100.0
        trend_pct = trend_strength * 100.0
        snap = {"volatility_pct": vol_pct, "trend_pct": trend_pct, "rsi": rsi, "price": price,
                "fast": fast, "slow": slow, "direction": direction, "reason": reason}
        async with self._lock:
            self.last_decision = snap

        if not direction:
            # Maintain-on-HOLD. If no ladder is running yet, deploy a neutral ladder so paper UI shows activity.
            running = bot_manager.task is not None and not bot_manager.task.done()
            if not running:
                # Spot-only: never deploy SELL neutral unless inventory exists
                has_inv = (bot_manager.position_qty or 0.0) > 1e-8
                neutral_dir = "BUY" if (not has_inv or rsi <= 50.0) else "SELL"
                interval = round(max(0.01, cfg.base_interval * cfg.risk_multiplier), 2)
                steps = max(3, min(20, cfg.base_steps))
                size_asset = round((cfg.rung_notional * cfg.risk_multiplier) / max(price, 1e-12), 8)
                max_exposure_asset = round((cfg.max_notional * cfg.risk_multiplier) / max(price, 1e-12), 8)
                new_cfg = LadderConfig(
                    symbol=cfg.symbol, direction=neutral_dir, steps=steps,
                    interval=interval, size=size_asset, max_exposure=max_exposure_asset
                )
                await self._ensure_ladder(new_cfg, {"reason": "Neutral deploy", "volatility_pct": vol_pct, "trend_pct": trend_pct,
                                                    "size_notional": cfg.rung_notional * cfg.risk_multiplier,
                                                    "max_notional": cfg.max_notional * cfg.risk_multiplier}, price=price)
                return
            async with bot_manager._lock:
                bot_manager.last_action = "Autopilot standby"
                bot_manager.manual_price = None
            async with self._lock:
                idle = self.applied_config is None
            if not idle:
                await self._record("Standby", note=reason, reason=reason, trend_pct=trend_pct,
                                   volatility_pct=vol_pct, rsi=rsi, price=price)
            return

        # Spot-only: if SELL signal but no inventory, hold
        if direction == "SELL" and (bot_manager.position_qty or 0.0) <= 1e-8:
            async with bot_manager._lock:
                bot_manager.last_action = "Standby (spot only: no inventory to sell)"
            await self._record("Standby", note="Spot only: no inventory to sell", reason=reason,
                               trend_pct=trend_pct, volatility_pct=vol_pct, rsi=rsi, price=price)
            return

        # adapt ladder
        interval_mult = 1.0; steps_adj = 0; size_mult = 1.0
        if vol_pct >= 1.2:
            interval_mult = 1.8; steps_adj = -2; size_mult = 0.7
        elif vol_pct >= 0.7:
            interval_mult = 1.3; steps_adj = -1; size_mult = 0.85
        elif vol_pct <= 0.25:
            interval_mult = 0.75; steps_adj = 1; size_mult = 1.1
        if (direction == "BUY" and trend_strength > 0) or (direction == "SELL" and trend_strength < 0):
            size_mult *= 1.1

        interval = max(0.01, cfg.base_interval * interval_mult * cfg.risk_multiplier)
        steps = max(3, min(20, cfg.base_steps + steps_adj))
        size_notional = cfg.rung_notional * size_mult * cfg.risk_multiplier
        max_notional = cfg.max_notional * cfg.risk_multiplier
        if size_notional * steps > max_notional:
            size_notional = max_notional / max(steps, 1)

        size_asset = round(size_notional / price, 8)
        max_exposure_asset = round(max_notional / price, 8)
        interval = round(interval, 2)
        if size_asset <= 0 or max_exposure_asset <= 0:
            raise RuntimeError("Computed non-positive sizes; adjust inputs")

        new_cfg = LadderConfig(
            symbol=cfg.symbol, direction=direction, steps=steps,
            interval=interval, size=size_asset, max_exposure=max_exposure_asset
        )
        meta = {"reason": reason, "volatility_pct": vol_pct, "trend_pct": trend_pct,
                "size_notional": size_notional, "max_notional": max_notional}
        await self._ensure_ladder(new_cfg, meta, price=price)

    async def _ensure_ladder(self, cfg: LadderConfig, meta: Dict[str, float], *, price: float):
        running = bot_manager.task is not None and not bot_manager.task.done()
        current = bot_manager.cfg
        # Guard: do not deploy SELL ladder when flat (spot only)
        if cfg.direction == "SELL" and (bot_manager.position_qty or 0.0) <= 1e-8:
            async with bot_manager._lock:
                bot_manager.last_action = "Standby (spot only: no inventory to sell)"
            await self._record("Standby", note="Spot only: no inventory to sell",
                               reason=meta.get("reason",""), trend_pct=meta.get("trend_pct"),
                               volatility_pct=meta.get("volatility_pct"), price=price)
            return
        if running and current and ladder_configs_close(current, cfg):
            async with bot_manager._lock:
                bot_manager.last_action = f"Autopilot maintaining {cfg.direction} ladder – vol {meta['volatility_pct']:.2f}%"
            async with self._lock:
                self.applied_config = cfg
            # coalesce spam: only record first maintain after a change
            last_action = self.history[-1]["action"] if self.history else None
            if last_action and str(last_action).startswith("Maintain"):
                return
            await self._record("Maintain "+cfg.direction,
                               note=f"{cfg.steps} steps @ ${cfg.interval:.2f} (size {cfg.size:.6f})",
                               reason=meta["reason"], trend_pct=meta["trend_pct"], volatility_pct=meta["volatility_pct"],
                               interval=cfg.interval, steps=cfg.steps, size_asset=cfg.size,
                               size_notional=meta["size_notional"], max_notional=meta["max_notional"], price=price)
            return

        if running:
            await bot_manager.stop()
        await bot_manager.start(cfg)
        async with bot_manager._lock:
            bot_manager.last_action = f"Autopilot set {cfg.direction} ladder: {meta['reason']} | interval ${cfg.interval:.2f}"
        async with self._lock:
            self.applied_config = cfg
        await self._record("Deploy "+cfg.direction,
                           note=f"{cfg.steps} steps @ ${cfg.interval:.2f} (size {cfg.size:.6f})",
                           reason=meta["reason"], trend_pct=meta["trend_pct"], volatility_pct=meta["volatility_pct"],
                           interval=cfg.interval, steps=cfg.steps, size_asset=cfg.size,
                           size_notional=meta["size_notional"], max_notional=meta["max_notional"], price=price)

    def snapshot(self) -> Dict[str, Optional[object]]:
        return {
            "running": self.task is not None and not self.task.done(),
            "config": self.cfg.dict() if self.cfg else None,
            "last_signal": self.last_signal,
            "last_reason": self.last_reason,
            "last_decision": self.last_decision,
            "last_error": self.last_error,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "applied_ladder": self.applied_config.dict() if self.applied_config else None,
            "history": list(self.history),
        }

autopilot_manager = AutopilotManager()

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
async def index():
    return FileResponse("static/index.html")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/api/bars")
async def get_bars(params: BarParams = Depends()):
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    normalized = normalize_symbol(params.symbols)
    query = params.dict(exclude_none=True)
    query["symbols"] = normalized
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=query, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        return JSONResponse(content=r.json())

# Manual ladder endpoints have been removed in favor of Autopilot-only control

@app.get("/api/account")
async def account():
    # Simple cache to avoid hitting rate limits (HTTP 429) when the UI polls frequently
    now = datetime.now(timezone.utc)
    ttl = timedelta(seconds=20)
    cache = getattr(account, "_cache", None)
    cache_ts = getattr(account, "_cache_ts", None)
    if cache is not None and cache_ts and (now - cache_ts) < ttl:
        return cache

    endpoint = f"{ALPACA_PAPER_URL}/v2/account"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, headers=headers)
        if r.status_code == 429 and cache is not None:
            # return last good cache if rate limited
            return cache
        r.raise_for_status()
        data = r.json()
        account._cache = data  # type: ignore[attr-defined]
        account._cache_ts = now  # type: ignore[attr-defined]
        return data

@app.get("/api/logs")
async def logs(tail: int = 200):
    with open("ladder_bot.log", "r", encoding="utf-8", errors="strict") as f:
        lines = f.readlines()[-tail:]
    return {"logs": [l.rstrip("\n") for l in lines]}

@app.post("/api/start-autopilot")
async def start_autopilot(cfg: AutopilotConfig):
    try:
        await autopilot_manager.start(cfg)
        return {"status": "started", "config": cfg.dict()}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/api/stop-autopilot")
async def stop_autopilot():
    await autopilot_manager.stop()
    return {"status": "stopped"}

@app.get("/api/autopilot-status")
async def get_autopilot_status():
    return autopilot_manager.snapshot()

@app.get("/api/status")
async def get_status():
    # last price
    last_price: Optional[float] = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        last_price = await get_latest_price(client, bot_manager.cfg.symbol if bot_manager.cfg else "BTC/USD")

    position_qty = bot_manager.position_qty
    avg_price = bot_manager.avg_price
    if last_price is None:
        raise HTTPException(status_code=502, detail="Failed to fetch last price")
    lp = last_price
    capital_used = abs(position_qty) * lp
    unrealized = (lp - avg_price) * position_qty if position_qty else 0.0
    unrealized_pct = ((lp - avg_price) / avg_price * 100.0) if avg_price else 0.0

    day_realized = 0.0
    today = date.today()
    for o in bot_manager.filled_orders:
        try:
            ts = o.get("filled_at") or o.get("updated_at") or o.get("created_at")
            if not ts:
                continue
            tsd = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).date()
            if tsd == today:
                day_realized += float(o.get("realized_delta", 0.0) or 0.0)
        except Exception:
            continue

    cfg = bot_manager.cfg
    steps = cfg.steps if cfg else 0
    size = cfg.size if cfg else 0.0
    ladder_notional_max = steps * size * lp if lp else 0.0
    capacity_remaining = max(0.0, ladder_notional_max - capital_used)

    return {
        "running": (bot_manager.task is not None and not bot_manager.task.done()),
        "config": bot_manager.cfg.dict() if bot_manager.cfg else None,
        "position_qty": position_qty,
        "avg_price": avg_price,
        "realized_pnl": bot_manager.pnl,
        "open_orders": bot_manager.open_orders,
        "filled_orders": bot_manager.filled_orders[-10:],
        "last_error": bot_manager.last_error,
        "last_action": bot_manager.last_action,
        "last_price": last_price,
        "capital_used": capital_used,
        "unrealized_pnl_usd": unrealized,
        "unrealized_pnl_pct": unrealized_pct,
        "day_realized_pnl_usd": day_realized,
        "ladder_notional_max": ladder_notional_max,
        "capacity_remaining": capacity_remaining,
        "open_order_count": len(bot_manager.open_orders),
        # New explicit capital fields
        "position_notional_usd": abs(position_qty) * lp,
        "open_orders_notional_usd": sum(
            (float(o.get("qty", 0)) if isinstance(o.get("qty"), (int, float, str)) else 0.0)
            * float(o.get("limit_price") or o.get("filled_avg_price") or 0.0)
            for o in bot_manager.open_orders
        ),
        "deployed_ladder_notional_usd": (steps * size * lp) if lp else 0.0,
        "net_deployed_usd": (abs(position_qty) * lp) + sum(
            (float(o.get("qty", 0)) if isinstance(o.get("qty"), (int, float, str)) else 0.0)
            * float(o.get("limit_price") or o.get("filled_avg_price") or 0.0)
            for o in bot_manager.open_orders
        ),
        "autopilot": autopilot_manager.snapshot(),
    }


@app.get("/api/health")
async def get_health():
    # env
    env_ok = bool(ALPACA_API_KEY and ALPACA_API_SECRET)
    trading_ok = False
    data_ok = False
    last_price_age_s: Optional[float] = None
    autopilot_running = autopilot_manager.task is not None and not autopilot_manager.task.done()
    autopilot_last_run_age_s: Optional[float] = None
    chart_plugin = os.path.exists(os.path.join("static", "vendor", "chartjs-chart-financial.min.js"))

    now = datetime.now(timezone.utc)
    acct = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # trading
            r = await client.get(f"{ALPACA_PAPER_URL}/v2/account", headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET
            })
            trading_ok = (r.status_code == 200)
            if trading_ok:
                acct = r.json()
            # data
            bar = await get_latest_bar(client, bot_manager.cfg.symbol if bot_manager.cfg else "BTC/USD")
            data_ok = bool(bar and bar.get("c"))
            try:
                ts = bar.get("t")
                bt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(timezone.utc)
                last_price_age_s = max(0.0, (now - bt).total_seconds())
            except Exception:
                last_price_age_s = None
    except Exception:
        pass

    if autopilot_manager.last_run:
        autopilot_last_run_age_s = max(0.0, (now - autopilot_manager.last_run).total_seconds())

    # Build human-readable report
    cfg = autopilot_manager.cfg
    poll = cfg.poll_seconds if cfg else None
    issues = []
    severity = "ok"
    if not env_ok:
        issues.append("Missing Alpaca API credentials")
        severity = "error"
    if not trading_ok:
        issues.append("Trading API unreachable")
        severity = "error"
    if not data_ok:
        issues.append("Data API unreachable")
        severity = "error"
    if last_price_age_s is None or (last_price_age_s is not None and last_price_age_s > 90):
        issues.append("Price feed appears stale")
        if severity != "error":
            severity = "warn"
    if autopilot_running and poll and (autopilot_last_run_age_s is not None) and autopilot_last_run_age_s > (poll * 3):
        issues.append("Autopilot poll delayed")
        if severity != "error":
            severity = "warn"
    if not chart_plugin:
        issues.append("Candlestick plugin missing (charts only)")
        if severity != "error":
            severity = "warn"

    summary = "All systems go" if not issues else f"Needs attention: {issues[0]}"

    return {
        "env_ok": env_ok,
        "trading_ok": trading_ok,
        "data_ok": data_ok,
        "last_price_age_s": last_price_age_s,
        "autopilot_running": autopilot_running,
        "autopilot_last_run_age_s": autopilot_last_run_age_s,
        "chart_plugin": chart_plugin,
        "poll_seconds": poll,
        "severity": severity,
        "summary": summary,
        "issues": issues,
        "equity": (acct or {}).get("equity"),
        "buying_power": (acct or {}).get("buying_power"),
        "non_marginable_buying_power": (acct or {}).get("non_marginable_buying_power"),
        "crypto_status": (acct or {}).get("crypto_status"),
        "account_blocked": (acct or {}).get("account_blocked"),
    }


@app.post("/api/paper-seed")
async def paper_seed(notional_usd: float = 10.0, cancel_open: bool = True, side: str = "buy"):
    """Place a tiny market order in paper to seed a live position and trigger TP/SL maintenance.
    Non-destructive: defaults to $10 notional.
    """
    sym = bot_manager.cfg.symbol if bot_manager.cfg else (autopilot_manager.cfg.symbol if autopilot_manager.cfg else "BTC/USD")
    if cancel_open:
        try:
            await cancel_all_open_orders(sym)
        except Exception:
            pass
    async with httpx.AsyncClient(timeout=10.0) as client:
        last = await get_latest_price(client, sym)
        qty = round(max(0.00000001, notional_usd / max(last, 1e-12)), 8)
        o = await submit_market_order(client, symbol=sym, side=side, qty=qty)
        # Refresh state and ensure TP/SL reflect the new position; cfg only supplies symbol
        dummy_cfg = bot_manager.cfg or LadderConfig(symbol=sym, direction="BUY", steps=5, interval=50.0, size=qty, max_exposure=qty * 10)
        await update_fills_and_position(client, dummy_cfg)
    return {"ok": True, "symbol": sym, "qty": qty}


@app.post("/api/cancel-open")
async def cancel_open_all(symbol: Optional[str] = None):
    sym = symbol or (bot_manager.cfg.symbol if bot_manager.cfg else (autopilot_manager.cfg.symbol if autopilot_manager.cfg else "BTC/USD"))
    await cancel_all_open_orders(sym)
    return {"ok": True, "symbol": sym}

@app.post("/api/self-test")
async def run_self_test():
    started = datetime.now(timezone.utc)
    lines: List[str] = []
    ok = True

    # Env
    env_ok = bool(ALPACA_API_KEY and ALPACA_API_SECRET)
    lines.append(f"Environment: credentials {'OK' if env_ok else 'MISSING'}")
    if not env_ok:
        ok = False

    acct = None
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(f"{ALPACA_PAPER_URL}/v2/account", headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
            })
            if r.status_code == 200:
                acct = r.json()
                lines.append("Trading API: OK")
            else:
                ok = False
                lines.append(f"Trading API: FAIL ({r.status_code})")

            symbol = bot_manager.cfg.symbol if bot_manager.cfg else (autopilot_manager.cfg.symbol if autopilot_manager.cfg else "BTC/USD")
            hist = await fetch_crypto_bars(client, symbol, limit=120)
            closes = [b["c"] for b in hist]
            last_price = closes[-1]
            # recency
            last_bar = hist[-1]
            try:
                bt = datetime.fromisoformat(str(last_bar.get("t")).replace("Z", "+00:00")).astimezone(timezone.utc)
                age = max(0.0, (datetime.now(timezone.utc) - bt).total_seconds())
                lines.append(f"Data API: OK (last ${last_price:,.2f}, age {age:.0f}s)")
            except Exception:
                lines.append(f"Data API: OK (last ${last_price:,.2f})")

            # Strategy preview (no orders)
            cfg = autopilot_manager.cfg or AutopilotConfig(symbol=symbol, rung_notional=1000.0, max_notional=7000.0)
            fast = compute_ema(closes[-cfg.fast_window - 1 :], cfg.fast_window)
            slow = compute_ema(closes[-cfg.slow_window - 1 :], cfg.slow_window)
            rsi = compute_rsi(closes, cfg.rsi_window)
            vol = compute_pct_volatility(closes[-(cfg.volatility_lookback + 1) :])
            direction, reason = autopilot_manager._signal(fast, slow, rsi, cfg, vol)

            vol_pct = vol * 100.0
            # mirror sizing logic for preview
            interval_mult = 1.0; steps_adj = 0; size_mult = 1.0
            if vol_pct >= 1.2:
                interval_mult = 1.8; steps_adj = -2; size_mult = 0.7
            elif vol_pct >= 0.7:
                interval_mult = 1.3; steps_adj = -1; size_mult = 0.85
            elif vol_pct <= 0.25:
                interval_mult = 0.75; steps_adj = 1; size_mult = 1.1
            if (direction == "BUY" and fast > slow) or (direction == "SELL" and fast < slow):
                size_mult *= 1.1

            interval = round(max(0.01, cfg.base_interval * interval_mult * cfg.risk_multiplier), 2)
            steps = max(3, min(20, cfg.base_steps + steps_adj))
            size_notional = cfg.rung_notional * size_mult * cfg.risk_multiplier
            max_notional = cfg.max_notional * cfg.risk_multiplier
            if size_notional * steps > max_notional:
                size_notional = max_notional / max(steps, 1)
            size_asset = round(size_notional / last_price, 8) if last_price else 0.0

            lines.append(f"Signal preview: {direction or 'HOLD'} | {reason}")
            if direction:
                lines.append(f"Would deploy: {steps} steps @ ${interval:.2f}, per-rung {size_asset:.6f}")
            else:
                lines.append("Neutral: would maintain existing ladder if running")
    except Exception as exc:
        ok = False
        lines.append(f"Connectivity/preview error: {exc}")

    snap = autopilot_manager.snapshot()
    lines.append(f"Autopilot: {'RUNNING' if snap.get('running') else 'IDLE'}")
    duration_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)
    lines.append(f"Completed in {duration_ms} ms")

    return {"ok": ok, "report": "\n".join(lines), "autopilot": snap}


# demo endpoints removed (Autopilot-only)


# -------------- Panic close: cancel all and flatten positions --------------
async def _list_all_positions(client: httpx.AsyncClient) -> List[dict]:
    endpoint = f"{ALPACA_PAPER_URL}/v2/positions"
    headers = {"APCA-API-KEY-ID": ALPACA_API_KEY, "APCA-API-SECRET-KEY": ALPACA_API_SECRET}
    r = await client.get(endpoint, headers=headers)
    if r.status_code == 200:
        try:
            return r.json()
        except Exception:
            return []
    return []


@app.post("/api/panic-close")
async def panic_close(symbol: Optional[str] = None):
    target_symbols: List[str] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        if not symbol or str(symbol).upper() == "ALL":
            pos = await _list_all_positions(client)
            target_symbols = [p.get("symbol") or p.get("asset_id") for p in pos if p.get("symbol")]
            if not target_symbols:
                # fallback to current symbol if none listed
                target_symbols = [bot_manager.cfg.symbol if bot_manager.cfg else (autopilot_manager.cfg.symbol if autopilot_manager.cfg else "BTC/USD")]
        else:
            target_symbols = [symbol]

        results: List[Dict[str, object]] = []
        for sym in target_symbols:
            # Cancel open orders
            await cancel_all_open_orders(sym)
            # Flatten position
            qty = await get_position(client, sym)
            if abs(qty) > 0:
                side = "sell" if qty > 0 else "buy"
                await submit_market_order(client, symbol=sym, side=side, qty=abs(qty))
            results.append({"symbol": sym, "flattened": abs(qty) > 0})
    return {"ok": True, "results": results}
