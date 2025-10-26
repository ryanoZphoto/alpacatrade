import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional
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

# -------------------------------------------------
# Alpaca helper functions
# -------------------------------------------------
async def get_latest_price(client: httpx.AsyncClient, symbol: str) -> float:
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    params = {
        "symbols": symbol,
        "timeframe": "1Min",
        "limit": 1,
        "sort": "desc",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    container = data.get("bars", [])
    if isinstance(container, dict):
        # Prefer exact symbol key; otherwise take the first series
        series = (
            container.get(symbol)
            or container.get(symbol.replace("-", "/"))
            or next(iter(container.values()), [])
        )
    else:
        series = container
    if not series:
        raise RuntimeError(f"No bar data for {symbol}")
    last = series[0]
    price_value = last.get("c") if isinstance(last, dict) else None
    if price_value is None:
        price_value = last.get("close") if isinstance(last, dict) else None
    if price_value is None:
        raise RuntimeError("Bar format unexpected; missing close price")
    return float(price_value)

async def get_latest_bar(client: httpx.AsyncClient, symbol: str) -> dict:
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    params = {
        "symbols": symbol,
        "timeframe": "1Min",
        "limit": 1,
        "sort": "desc",
    }
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    resp = await client.get(endpoint, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    container = data.get("bars", [])
    if isinstance(container, dict):
        series = (
            container.get(symbol)
            or container.get(symbol.replace("-", "/"))
            or container.get(symbol.replace("/", ""))
            or next(iter(container.values()), [])
        )
    else:
        series = container
    if not series:
        raise RuntimeError(f"No bar data for {symbol}")
    bar = series[0]
    # Ensure keys exist; map alternative keys if present
    return {
        "c": float(bar.get("c", bar.get("close"))),
        "h": float(bar.get("h", bar.get("high", bar.get("c", 0)))),
        "l": float(bar.get("l", bar.get("low", bar.get("c", 0)))),
    }

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

@app.get("/api/bars")
async def get_bars(params: BarParams = Depends()):
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    # Normalize symbol to Alpaca format: BTCUSD -> BTC/USD
    normalized = params.symbols.replace("-", "/")
    if "/" not in normalized and len(normalized) > 3:
        normalized = f"{normalized[:-3]}/{normalized[-3:]}"
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
