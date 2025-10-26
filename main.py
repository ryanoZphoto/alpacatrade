import os
import math
import asyncio
import logging
import statistics
from fastapi import FastAPI, HTTPException, Depends, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
import httpx
from datetime import datetime, timezone, date
from dotenv import load_dotenv

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_PAPER_URL = os.getenv(
    "ALPACA_PAPER_URL", "https://paper-api.alpaca.markets"
)
ALPACA_DATA_URL = os.getenv(
    "ALPACA_DATA_URL", "https://data.alpaca.markets"
)
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
# FastAPI app
# -------------------------------------------------
app = FastAPI(title="Alpaca Ladder Bot UI")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class BarParams(BaseModel):
    symbols: str = Field(
        ..., description="comma‑separated symbols, e.g. BTCUSD"
    )
    timeframe: str = Field(..., description="Alpaca timeframe")
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
        if v.upper() not in {"BUY", "SELL"}:
            raise ValueError("direction must be BUY or SELL")
        return v.upper()

class AutopilotConfig(BaseModel):
    symbol: str = Field(..., description="Trading symbol, e.g. BTC/USD")
    fast_window: int = Field(12, ge=3, le=60)
    slow_window: int = Field(26, ge=5, le=240)
    rsi_window: int = Field(14, ge=5, le=240)
    overbought: float = Field(70.0, gt=50.0, lt=100.0)
    oversold: float = Field(30.0, gt=0.0, lt=50.0)
    base_interval: float = Field(150.0, gt=0.0)
    base_steps: int = Field(7, ge=3, le=20)
    rung_notional: float = Field(..., gt=0.0, description="USD per rung before adjustments")
    max_notional: float = Field(..., gt=0.0, description="Maximum USD the ladder may deploy")
    volatility_lookback: int = Field(60, ge=10, le=500)
    risk_multiplier: float = Field(1.0, gt=0.1, le=5.0)
    poll_seconds: float = Field(30.0, ge=10.0, le=120.0)

    @validator("slow_window")
    def slow_greater_than_fast(cls, v, values):
        fast = values.get("fast_window")
        if fast and v <= fast:
            raise ValueError("slow_window must be greater than fast_window")
        return v

    @validator("overbought")
    def overbought_above_oversold(cls, v, values):
        oversold = values.get("oversold")
        if oversold and v <= oversold:
            raise ValueError("overbought must exceed oversold")
        return v

class NudgeRequest(BaseModel):
    direction: str

# -------------------------------------------------
# Bot manager (singleton)
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
        # if set, ladder centers here
        self.manual_price: Optional[float] = None
        self.last_action: Optional[str] = None

    async def start(self, cfg: LadderConfig):
        async with self._lock:
            if self.task and not self.task.done():
                raise RuntimeError("Bot already running")
            self.cfg = cfg
            self.position_qty = 0.0
            self.avg_price = 0.0
            self.pnl = 0.0
            self.open_orders.clear()
            self.filled_orders.clear()
            self.last_error = None
            self.manual_price = None
            self.last_action = "Started ladder"
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
                self.last_action = "Stopped ladder and cancelled rungs"

    async def _run(self):
        cfg = self.cfg
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                while True:
                    # 1️⃣ get latest bar / price
                    bar = await get_latest_bar(client, cfg.symbol)
                    price = float(bar["c"]) 
                    high = float(bar["h"]) 
                    low = float(bar["l"]) 
                    log.info(f"price {cfg.symbol}: {price}")

                    # Circuit breaker (cancel all and stop, manual restart only)
                    pct_range = ((high - low) / max(price, 1e-9)) * 100.0
                    if pct_range >= CB_THRESHOLD_PCT:
                        await cancel_all_open_orders(cfg.symbol)
                        self.last_error = (
                            f"Circuit breaker: 1m range {pct_range:.2f}% ≥ {CB_THRESHOLD_PCT:.2f}%"
                        )
                        self.last_action = "Circuit breaker tripped; stopped and cancelled rungs"
                        break

                    # 2️⃣ sync ladder orders
                    center_price = self.manual_price if self.manual_price is not None else price
                    await sync_ladder(client, cfg, center_price)

                    # 3️⃣ process fills / update position
                    await update_fills_and_position(client, cfg)

                    await asyncio.sleep(5)   # ← tick interval
            except Exception as exc:
                self.last_error = str(exc)
                log.exception("Bot crashed")
            finally:
                await client.aclose()

bot_manager = BotManager()


def normalize_symbol(symbol: str) -> str:
    normalized = symbol.replace("-", "/").upper()
    if "/" not in normalized and len(normalized) > 3:
        normalized = f"{normalized[:-3]}/{normalized[-3:]}"
    return normalized


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
        for key in candidates:
            if key in container:
                return container[key]
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
    params = {
        "symbols": normalize_symbol(symbol),
        "timeframe": timeframe,
        "limit": limit,
        "sort": "desc",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    series = _pick_series(symbol, data.get("bars", []))
    if not series:
        raise RuntimeError(f"No bar data for {symbol}")
    bars: List[Dict[str, float]] = []
    for raw in reversed(series):  # chronological order (oldest → newest)
        bars.append(
            {
                "t": raw.get("t") or raw.get("timestamp"),
                "o": float(raw.get("o", raw.get("open"))),
                "h": float(raw.get("h", raw.get("high", raw.get("c", 0)))),
                "l": float(raw.get("l", raw.get("low", raw.get("c", 0)))),
                "c": float(raw.get("c", raw.get("close"))),
            }
        )
    return bars[-limit:]


def compute_ema(values: List[float], window: int) -> float:
    if not values:
        return 0.0
    k = 2 / (window + 1)
    ema_value = values[0]
    for price in values[1:]:
        ema_value = price * k + ema_value * (1 - k)
    return ema_value


def compute_rsi(values: List[float], window: int) -> float:
    if len(values) <= window:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains[-window:]) / window
    avg_loss = sum(losses[-window:]) / window
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_pct_volatility(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    returns = []
    for prev, curr in zip(values[:-1], values[1:]):
        if prev == 0:
            continue
        returns.append((curr - prev) / prev)
    if not returns:
        return 0.0
    return statistics.pstdev(returns)


def ladder_configs_close(a: LadderConfig, b: LadderConfig) -> bool:
    return (
        a.symbol == b.symbol
        and a.direction == b.direction
        and math.isclose(a.interval, b.interval, rel_tol=0.05, abs_tol=0.5)
        and math.isclose(a.size, b.size, rel_tol=0.05, abs_tol=1e-6)
        and a.steps == b.steps
        and math.isclose(a.max_exposure, b.max_exposure, rel_tol=0.05, abs_tol=1e-6)
    )


async def get_latest_price(client: httpx.AsyncClient, symbol: str) -> float:
    bars = await fetch_crypto_bars(client, symbol, limit=1)
    if not bars:
        raise RuntimeError(f"No bar data for {symbol}")
    return float(bars[-1]["c"])


async def get_latest_bar(client: httpx.AsyncClient, symbol: str) -> dict:
    bars = await fetch_crypto_bars(client, symbol, limit=1)
    if not bars:
        raise RuntimeError(f"No bar data for {symbol}")
    return bars[-1]


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
            self.task = asyncio.create_task(self._run())
        log.info("Autopilot started with %s", cfg.dict())

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
        log.info("Autopilot stopped")

    async def _run(self):
        assert self.cfg is not None
        cfg = self.cfg
        lookback = max(cfg.slow_window, cfg.rsi_window, cfg.volatility_lookback) + 5
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    history = await fetch_crypto_bars(client, cfg.symbol, limit=lookback)
                    closes = [bar["c"] for bar in history]
                    if len(closes) < lookback:
                        raise RuntimeError(
                            "Insufficient history returned from Alpaca for strategy computation"
                        )
                    fast = compute_ema(closes[-cfg.fast_window - 1 :], cfg.fast_window)
                    slow = compute_ema(closes[-cfg.slow_window - 1 :], cfg.slow_window)
                    rsi = compute_rsi(closes, cfg.rsi_window)
                    trend_strength = (fast - slow) / slow if slow else 0.0
                    volatility = compute_pct_volatility(
                        closes[-(cfg.volatility_lookback + 1) :]
                    )
                    direction, reason = self._determine_direction(fast, slow, rsi, cfg)
                    await self._apply_decision(
                        direction=direction,
                        price=closes[-1],
                        volatility=volatility,
                        trend_strength=trend_strength,
                        rsi=rsi,
                        fast=fast,
                        slow=slow,
                        reason=reason,
                    )
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
                await asyncio.sleep(cfg.poll_seconds)

    def _determine_direction(
        self,
        fast: float,
        slow: float,
        rsi: float,
        cfg: AutopilotConfig,
    ) -> (Optional[str], str):
        if slow == 0:
            return None, "Slow EMA is zero; skipping signal"
        if fast > slow * 1.001 and rsi < cfg.overbought:
            return "BUY", f"Trend up (fast {fast:.2f} > slow {slow:.2f}), RSI {rsi:.1f}"
        if fast < slow * 0.999 and rsi > cfg.oversold:
            return "SELL", f"Trend down (fast {fast:.2f} < slow {slow:.2f}), RSI {rsi:.1f}"
        return None, f"Neutral: fast {fast:.2f} vs slow {slow:.2f}, RSI {rsi:.1f}"

    async def _apply_decision(
        self,
        *,
        direction: Optional[str],
        price: float,
        volatility: float,
        trend_strength: float,
        rsi: float,
        fast: float,
        slow: float,
        reason: str,
    ):
        cfg = self.cfg
        if cfg is None:
            return
        volatility_pct = volatility * 100.0
        trend_pct = trend_strength * 100.0
        snapshot = {
            "volatility_pct": volatility_pct,
            "trend_pct": trend_pct,
            "rsi": rsi,
            "price": price,
            "fast": fast,
            "slow": slow,
            "direction": direction,
            "reason": reason,
        }
        async with self._lock:
            self.last_decision = snapshot

        if not direction:
            await bot_manager.stop()
            async with bot_manager._lock:
                bot_manager.last_action = "Autopilot on standby – waiting for directional edge"
                bot_manager.manual_price = None
            async with self._lock:
                self.applied_config = None
            return

        interval_mult = 1.0
        steps_adj = 0
        size_mult = 1.0
        if volatility_pct >= 1.2:
            interval_mult = 1.8
            steps_adj = -2
            size_mult = 0.7
        elif volatility_pct >= 0.7:
            interval_mult = 1.3
            steps_adj = -1
            size_mult = 0.85
        elif volatility_pct <= 0.25:
            interval_mult = 0.75
            steps_adj = 1
            size_mult = 1.1

        if direction == "BUY" and trend_strength > 0:
            size_mult *= 1.1
        elif direction == "SELL" and trend_strength < 0:
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
            raise RuntimeError("Computed ladder sizes are non-positive; adjust autopilot inputs")

        new_cfg = LadderConfig(
            symbol=cfg.symbol,
            direction=direction,
            steps=steps,
            interval=interval,
            size=size_asset,
            max_exposure=max_exposure_asset,
        )

        meta = {
            "reason": reason,
            "volatility_pct": volatility_pct,
            "trend_pct": trend_pct,
            "size_notional": size_notional,
            "max_notional": max_notional,
        }
        await self._ensure_ladder(new_cfg, meta)

    async def _ensure_ladder(self, cfg: LadderConfig, meta: Dict[str, float]):
        running = bot_manager.task is not None and not bot_manager.task.done()
        current = bot_manager.cfg
        if running and current and ladder_configs_close(current, cfg):
            async with bot_manager._lock:
                bot_manager.last_action = (
                    f"Autopilot maintaining {cfg.direction} ladder – vol {meta['volatility_pct']:.2f}%"
                )
            async with self._lock:
                self.applied_config = cfg
            return

        if running:
            await bot_manager.stop()
        await bot_manager.start(cfg)
        async with bot_manager._lock:
            bot_manager.last_action = (
                f"Autopilot set {cfg.direction} ladder: {meta['reason']} | interval ${cfg.interval}"
            )
        async with self._lock:
            self.applied_config = cfg

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
        }

autopilot_manager = AutopilotManager()

def compute_step_prices(cfg: LadderConfig, price: float) -> List[float]:
    sign = -1 if cfg.direction == "BUY" else 1
    return [
        round(price + sign * i * cfg.interval, 2)
        for i in range(cfg.steps)
    ]

async def list_open_orders(
    client: httpx.AsyncClient, symbol: str
) -> List[dict]:
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    params = {"status": "open", "symbol": symbol, "asset_class": "crypto"}
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()

async def cancel_order(client: httpx.AsyncClient, order_id: str):
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders/{order_id}"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.delete(endpoint, headers=headers)
    # Alpaca returns 204 No Content on successful cancel
    if resp.status_code == 204 or not resp.content:
        return {"status": "cancelled"}
    resp.raise_for_status()
    return resp.json()

async def cancel_all_open_orders(symbol: str):
    async with httpx.AsyncClient() as client:
        open_orders = await list_open_orders(client, symbol)
        for o in open_orders:
            await cancel_order(client, o["id"])
            log.info(f"Cancelled {o['id']}")

async def place_limit_order(client: httpx.AsyncClient,
                            cfg: LadderConfig,
                            *, price: float):
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    payload = {
        "symbol": cfg.symbol,
        "qty": cfg.size,
        "side": cfg.direction.lower(),
        "type": "limit",
        "time_in_force": "gtc",
        "limit_price": price,
        "order_class": "simple",
        "asset_class": "crypto",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }
    resp = await client.post(endpoint, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

async def place_market_order(client: httpx.AsyncClient,
                             symbol: str,
                             *, side: str, qty: float):
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side.lower(),
        "type": "market",
        # For crypto market orders Alpaca requires IOC/FOK
        "time_in_force": "ioc",
        "order_class": "simple",
        "asset_class": "crypto",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Content-Type": "application/json",
    }
    resp = await client.post(endpoint, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()

async def sync_ladder(client: httpx.AsyncClient,
                      cfg: LadderConfig,
                      price: float):
    # Load open orders that match our size & side
    open_orders = await list_open_orders(client, cfg.symbol)
    matching = [
        o for o in open_orders
        if float(o["qty"]) == cfg.size and
           o["side"].lower() == cfg.direction.lower()
    ]

    # Map price → order
    price_to_order = {float(o["limit_price"]): o for o in matching}

    # Desired step prices
    target_prices = compute_step_prices(cfg, price)

    # Cancel stray orders (not in target set)
    for p, o in price_to_order.items():
        if p not in target_prices:
            await cancel_order(client, o["id"])
            log.info(f"Cancelled stray {o['id']} @ {p}")

    # Place any missing steps
    for p in target_prices:
        if p not in price_to_order:
            await place_limit_order(client, cfg, price=p)
            log.info(f"Placed {cfg.direction} limit @ {p}")

    # Refresh the open‑order snapshot for the UI
    bot_manager.open_orders = await list_open_orders(client, cfg.symbol)

async def update_fills_and_position(client: httpx.AsyncClient,
                                    cfg: LadderConfig):
    endpoint = f"{ALPACA_PAPER_URL}/v2/orders"
    params = {
        "status": "filled",
        "symbol": cfg.symbol,
        "asset_class": "crypto",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    filled = resp.json()

    # Filter out already‑processed fills
    processed = {o["id"] for o in bot_manager.filled_orders}
    new_fills = [o for o in filled if o["id"] not in processed]

    for o in new_fills:
        qty = float(o["filled_qty"])
        fill_price = float(o["filled_avg_price"])
        side = o["side"].lower()
        prev_pnl = bot_manager.pnl
        # Default realized delta for this fill (used for Day P&L)
        realized_delta = 0.0

        # ----- POSITION LOGIC -----
        if side == "buy":
            new_qty = bot_manager.position_qty + qty
            if new_qty != 0:
                bot_manager.avg_price = (
                    bot_manager.position_qty * bot_manager.avg_price +
                    qty * fill_price
                ) / new_qty
            bot_manager.position_qty = new_qty
        else:  # sell
            new_qty = bot_manager.position_qty - qty
            # Realised P&L for reducing a long
            if bot_manager.position_qty > 0:
                bot_manager.pnl += (fill_price - bot_manager.avg_price) * qty
            # Realised P&L for reducing a short
            elif bot_manager.position_qty < 0:
                bot_manager.pnl += (bot_manager.avg_price - fill_price) * qty
            bot_manager.position_qty = new_qty
            if abs(bot_manager.position_qty) < 1e-8:
                bot_manager.avg_price = 0.0

        # Compute realized delta from this fill (change in total realized pnl)
        realized_delta = bot_manager.pnl - prev_pnl
        try:
            o["realized_delta"] = realized_delta
        except Exception:
            pass

        bot_manager.filled_orders.append(o)
        log.info(f"Processed fill {o['id']}: {side} {qty}@{fill_price}")

    # Update open‑order snapshot after processing fills
    bot_manager.open_orders = await list_open_orders(client, cfg.symbol)

# -------------------------------------------------
# FastAPI routes
# -------------------------------------------------
@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/favicon.ico")
async def favicon():
    # Browsers request a favicon automatically; return an empty 204 instead of a 404
    return Response(status_code=204)

@app.get("/api/bars")
async def get_bars(params: BarParams = Depends()):
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    # Normalize symbol to Alpaca format: BTCUSD -> BTC/USD
    normalized = normalize_symbol(params.symbols)
    query = params.dict(exclude_none=True)
    query["symbols"] = normalized
    async with httpx.AsyncClient() as client:
        resp = await client.get(endpoint, params=query, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code,
                                detail=resp.text)
        return JSONResponse(content=resp.json())

@app.post("/api/start-ladder")
async def start_ladder(cfg: LadderConfig):
    try:
        await bot_manager.start(cfg)
        return {"status": "started", "cfg": cfg.dict()}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/stop-ladder")
async def stop_ladder():
    await bot_manager.stop()
    return {"status": "stopped"}


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
    # compute last price by peeking at latest bar using a short-lived client
    last_price: Optional[float] = None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            last_price = await get_latest_price(client, bot_manager.cfg.symbol if bot_manager.cfg else "BTC/USD")
    except Exception:
        last_price = None

    # derived metrics
    position_qty = bot_manager.position_qty
    avg_price = bot_manager.avg_price
    lp = last_price or 0.0
    capital_used = abs(position_qty) * lp
    unrealized = (lp - avg_price) * position_qty if position_qty else 0.0
    unrealized_pct = ((lp - avg_price) / avg_price * 100.0) if avg_price else 0.0

    # day realized PnL (UTC calendar day): sum per-fill realized deltas
    day_realized = 0.0
    today = date.today()
    for o in bot_manager.filled_orders:
        try:
            ts = o.get("filled_at") or o.get("updated_at") or o.get("created_at")
            if not ts:
                continue
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if dt.date() == today:
                day_realized += float(o.get("realized_delta", 0.0))
        except Exception:
            continue

    cfg = bot_manager.cfg
    steps = cfg.steps if cfg else 0
    size = cfg.size if cfg else 0.0
    ladder_notional_max = steps * size * lp if lp else 0.0
    capacity_remaining = max(0.0, ladder_notional_max - capital_used)

    return {
        "running": (bot_manager.task is not None
                    and not bot_manager.task.done()),
        "config": bot_manager.cfg.dict() if bot_manager.cfg else None,
        "position_qty": position_qty,
        "avg_price": avg_price,
        "realized_pnl": bot_manager.pnl,
        "open_orders": bot_manager.open_orders,
        "filled_orders": bot_manager.filled_orders[-10:],  # last 10
        "last_error": bot_manager.last_error,
        "last_action": bot_manager.last_action,
        # new derived fields for PnL/funds strip
        "last_price": last_price,
        "capital_used": capital_used,
        "unrealized_pnl_usd": unrealized,
        "unrealized_pnl_pct": unrealized_pct,
        "day_realized_pnl_usd": day_realized,
        "ladder_notional_max": ladder_notional_max,
        "capacity_remaining": capacity_remaining,
        "open_order_count": len(bot_manager.open_orders),
        "autopilot": autopilot_manager.snapshot(),
    }

@app.post("/api/nudge")
async def nudge_ladder(req: NudgeRequest):
    if not bot_manager.cfg:
        raise HTTPException(status_code=400, detail="Bot not running")
    step = bot_manager.cfg.interval
    async with bot_manager._lock:
        base = bot_manager.manual_price
        if base is None:
            # if no manual center, use avg of current rungs if available
            try:
                prices = [float(o.get("limit_price", 0)) for o in bot_manager.open_orders]
                base = sum(prices) / len(prices) if prices else None
            except Exception:
                base = None
        if base is None:
            # fallback to last avg_price or 0
            base = bot_manager.avg_price or 0.0
        if req.direction.lower() == "up":
            bot_manager.manual_price = base + step
            bot_manager.last_action = f"Nudged up by {step}"
        elif req.direction.lower() == "down":
            bot_manager.manual_price = base - step
            bot_manager.last_action = f"Nudged down by {step}"
        else:
            raise HTTPException(status_code=400, detail="direction must be 'up' or 'down'")
    return {"status": "ok", "manual_price": bot_manager.manual_price}

@app.post("/api/recenter")
async def recenter_ladder():
    async with bot_manager._lock:
        bot_manager.manual_price = None
        bot_manager.last_action = "Recentered to market price"
    return {"status": "ok"}

@app.post("/api/close-all")
async def close_all_positions():
    if not bot_manager.cfg:
        raise HTTPException(status_code=400, detail="Bot not running")

    # 1) Stop the ladder first so it doesn't re-seed rungs while we flatten
    #    This also cancels all open rungs
    await bot_manager.stop()

    symbol = bot_manager.cfg.symbol
    qty = abs(bot_manager.position_qty)

    # 2) Flatten any remaining position at market (IOC to avoid 422 on crypto)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            if qty > 0:
                side = "sell" if bot_manager.position_qty > 0 else "buy"
                await place_market_order(client, symbol, side=side, qty=qty)
            async with bot_manager._lock:
                bot_manager.last_action = "Closed all positions and cancelled orders"
        except Exception as exc:
            async with bot_manager._lock:
                bot_manager.last_error = str(exc)
            raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok"}

@app.post("/api/cancel-open")
async def cancel_open_orders_only():
    if not bot_manager.cfg:
        raise HTTPException(status_code=400, detail="Bot not running")
    try:
        await cancel_all_open_orders(bot_manager.cfg.symbol)
        async with bot_manager._lock:
            bot_manager.last_action = "Cancelled all open orders"
        return {"status": "ok"}
    except Exception as exc:
        async with bot_manager._lock:
            bot_manager.last_error = str(exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/api/logs")
async def get_logs(tail: int = 100):
    import pathlib, collections
    log_path = pathlib.Path("ladder_bot.log")
    if not log_path.is_file():
        return {"logs": []}
    lines = collections.deque(log_path.open(encoding="utf-8"), maxlen=tail)
    return {"logs": list(lines)}

@app.get("/api/account")
async def get_account():
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    endpoint = f"{ALPACA_PAPER_URL}/v2/account"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(endpoint, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        # Return a compact subset relevant to UI
        subset = {k: data.get(k) for k in (
            "cash", "equity", "buying_power", "multiplier", "portfolio_value",
        )}
        return subset
