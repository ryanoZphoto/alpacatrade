#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    print(f"   -> {path}")


def main() -> int:
    root = Path(__file__).resolve().parent
    static_dir = root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Creating project folder at {root}")

    # ---------- requirements.txt ----------
    requirements_txt = """fastapi>=0.100
uvicorn[standard]>=0.23
httpx>=0.27
python-dotenv>=1.0
pydantic>=2.5
"""
    write_text_file(root / "requirements.txt", requirements_txt)

    # ---------- .env.example ----------
    env_example = """# -------------------------------------------------
# Alpaca credentials – paste your paper keys here
# -------------------------------------------------
ALPACA_API_KEY=YOUR_PAPER_API_KEY
ALPACA_API_SECRET=YOUR_PAPER_API_SECRET

# Endpoints (paper by default)
ALPACA_PAPER_URL=https://paper-api.alpaca.markets
ALPACA_DATA_URL=https://data.alpaca.markets
"""
    write_text_file(root / ".env.example", env_example)

    # ---------- main.py ----------
    main_py = """import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import httpx
from datetime import datetime
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
        cfg = self.cfg
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                while True:
                    # 1️⃣ get latest price (1‑minute close)
                    price = await get_latest_price(client, cfg.symbol)
                    log.info(f"price {cfg.symbol}: {price}")

                    # 2️⃣ sync ladder orders
                    await sync_ladder(client, cfg, price)

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
    bars = data.get("bars", [])
    if not bars:
        raise RuntimeError(f"No bar data for {symbol}")
    return float(bars[0]["c"])

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
async def get_bars(params: BarParams):
    endpoint = f"{ALPACA_DATA_URL}/v1beta3/crypto/us/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(endpoint,
                                params=params.dict(exclude_none=True),
                                headers=headers)
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
    return {
        "running": (bot_manager.task is not None
                    and not bot_manager.task.done()),
        "config": bot_manager.cfg.dict() if bot_manager.cfg else None,
        "position_qty": bot_manager.position_qty,
        "avg_price": bot_manager.avg_price,
        "realized_pnl": bot_manager.pnl,
        "open_orders": bot_manager.open_orders,
        "filled_orders": bot_manager.filled_orders[-10:],  # last 10
        "last_error": bot_manager.last_error,
    }

@app.get("/api/logs")
async def get_logs(tail: int = 100):
    import pathlib, collections
    log_path = pathlib.Path("ladder_bot.log")
    if not log_path.is_file():
        return {"logs": []}
    lines = collections.deque(log_path.open(encoding="utf-8"), maxlen=tail)
    return {"logs": list(lines)}
"""
    write_text_file(root / "main.py", main_py)

    # ---------- static/index.html ----------
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Alpaca BTC Ladder Bot (Paper)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.0">
    </script>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>Alpaca BTC‑USD Ladder Bot (Paper)</h1>

    <!-- Chart Section -->
    <section id="chart-section">
        <h2>Historical Candles</h2>
        <div>
            Symbol:
            <input id="chart-symbol" type="text" value="BTCUSD" size="8">
            Timeframe:
            <select id="chart-timeframe">
                <option value="5Min">5Min</option>
                <option value="15Min">15Min</option>
                <option value="1Hour">1Hour</option>
                <option value="1Day" selected>1Day</option>
            </select>
            <button id="load-chart">Load Chart</button>
        </div>
        <canvas id="candles-canvas" width="900" height="400"></canvas>
    </section>

    <!-- Ladder Config -->
    <section id="ladder-section">
        <h2>Ladder Configuration</h2>
        <form id="ladder-form">
            <label>Symbol:
                <input name="symbol" value="BTCUSD" required>
            </label><br>
            <label>Direction:
                <select name="direction">
                    <option value="BUY">Buy (Long)</option>
                    <option value="SELL">Sell (Short)</option>
                </select>
            </label><br>
            <label>Steps:
                <input name="steps" type="number" value="5" min="1" max="20">
            </label><br>
            <label>Interval (USD):
                <input name="interval" type="number" step="0.01" value="200">
            </label><br>
            <label>Size per step (BTC):
                <input name="size" type="number" step="0.0001" value="0.01">
            </label><br>
            <label>Max exposure (BTC):
                <input name="max_exposure" type="number" step="0.0001"
                       value="0.1">
            </label><br><br>
            <button type="submit">Start Ladder</button>
            <button type="button" id="stop-btn">Stop Ladder</button>
        </form>
    </section>

    <!-- Bot Status -->
    <section id="status-section">
        <h2>Bot Status (refreshes every 5 s)</h2>
        <pre id="status-box">{}</pre>
    </section>

    <!-- Log tail -->
    <section id="log-section">
        <h2>Recent Logs (refreshes every 7 s)</h2>
        <pre id="log-box"></pre>
    </section>

    <script src="/static/script.js"></script>
</body>
</html>
"""
    write_text_file(static_dir / "index.html", index_html)

    # ---------- static/style.css ----------
    style_css = """body {
    font-family: Arial, Helvetica, sans-serif;
    margin: 20px;
    max-width: 1100px;
}
section {
    margin-bottom: 30px;
    border: 1px solid #ddd;
    padding: 15px;
    border-radius: 4px;
}
label {
    display: block;
    margin: 5px 0;
}
input, select, button {
    margin-left: 10px;
}
pre {
    background: #f4f4f4;
    border: 1px solid #ccc;
    padding: 8px;
    max-height: 250px;
    overflow-y: auto;
    font-size: 0.9em;
}
"""
    write_text_file(static_dir / "style.css", style_css)

    # ---------- static/script.js ----------
    script_js = """async function apiGet(path, params = {}) {
    const url = new URL(`/api/${path}`, location.origin);
    Object.entries(params).forEach(([k, v]) => url.searchParams.append(k, v));
    const resp = await fetch(url);
    if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`API ${resp.status}: ${txt}`);
    }
    return resp.json();
}
async function apiPost(path, body) {
    const resp = await fetch(`/api/${path}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
    });
    if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`API ${resp.status}: ${txt}`);
    }
    return resp.json();
}

/* ---------- Chart ---------- */
document.getElementById('load-chart').addEventListener('click', async () => {
    const symbol = document.getElementById('chart-symbol').value.trim();
    const tf = document.getElementById('chart-timeframe').value;
    try {
        const data = await apiGet(
            'bars',
            { symbols: symbol, timeframe: tf, limit: 500 }
        );
        const candles = data.bars.map(b => ({
            x: new Date(b.t),
            o: b.o, h: b.h, l: b.l, c: b.c
        }));
        if (window.candleChart) window.candleChart.destroy();
        const ctx = document.getElementById('candles-canvas').getContext('2d');
        window.candleChart = new Chart(ctx, {
            type: 'candlestick',
            data: {datasets: [{label: symbol, data: candles}]},
            options: {
                responsive: true,
                scales: {
                    x: {time: {unit: tf.includes('Day') ? 'day' : 'hour'}}
                }
            }
        });
    } catch (e) { alert('Chart error: '+e.message); }
});

/* ---------- Ladder form ---------- */
document.getElementById('ladder-form').addEventListener('submit', async e => {
    e.preventDefault();
    const f = e.target;
    const cfg = {
        symbol: f.symbol.value.trim(),
        direction: f.direction.value,
        steps: Number(f.steps.value),
        interval: Number(f.interval.value),
        size: Number(f.size.value),
        max_exposure: Number(f.max_exposure.value)
    };
    try { await apiPost('start-ladder', cfg); }
    catch (err) { alert('Start error: '+err.message); }
});
document.getElementById('stop-btn').addEventListener('click', async () => {
    try { await apiPost('stop-ladder', {}); }
    catch (err) { alert('Stop error: '+err.message); }
});

/* ---------- Auto‑refresh status & logs ---------- */
async function refreshStatus() {
    try {
        const st = await apiGet('status');
        document.getElementById('status-box').textContent =
            JSON.stringify(st, null, 2);
    } catch (e) {
        document.getElementById('status-box').textContent = 'Error: '+e;
    }
}
async function refreshLogs() {
    try {
        const L = await apiGet('logs', {tail: 200});
        document.getElementById('log-box').textContent = L.logs.join('\n');
    } catch (e) {
        document.getElementById('log-box').textContent = 'Log error: '+e;
    }
}
setInterval(refreshStatus, 5000);
setInterval(refreshLogs, 7000);
refreshStatus();
refreshLogs();
"""
    write_text_file(static_dir / "script.js", script_js)

    # ---------- README.md ----------
    readme_md = """# Alpaca BTC‑Ladder Prototype

This project scaffolds a small FastAPI app that:
- pulls the latest BTC‑USD price from Alpaca’s crypto‑bars endpoint
- builds a grid (ladder) of limit orders
- keeps the ladder alive (cancel stray, place missing)
- tracks fills, net position, and realized P&L
- serves a simple web UI (chart + config + live status/logs)

## How to run

1. Create/activate a virtual environment
2. Install deps: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill credentials
4. Start server: `uvicorn main:app --reload --port 8000`
"""
    write_text_file(root / "README.md", readme_md)

    print("All files written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
