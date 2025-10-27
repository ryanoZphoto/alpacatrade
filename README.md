# Alpaca BTC-Ladder Prototype

Small FastAPI app that:
- pulls latest BTC/USD bars from Alpaca
- maintains a price “ladder” of limit orders
- tracks fills, position, and realized P&L
- serves a simple web UI (chart, config, live status)
- includes an **Autopilot** controller that tunes the ladder using EMA + RSI with volatility-aware spacing

---

## Prerequisites

- Python 3.10+  
- Alpaca **paper** API key/secret  
- `pip`, `uvicorn`

---

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/ryanoZphoto/alpacatrade.git
Set-Location alpacatrade

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env   # paste your paper API key/secret, save
