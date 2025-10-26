# Alpaca BTC‑Ladder Prototype

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
