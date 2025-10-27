Alpaca BTC‑Ladder Prototype
This project scaffolds a small FastAPI app that:

pulls the latest BTC‑USD price from Alpaca’s crypto‑bars endpoint

builds a grid (ladder) of limit orders

keeps the ladder alive (cancel stray, place missing)

tracks fills, net position, and realized P&L

serves a simple web UI (chart + config + live status/logs)

includes an automated “autopilot” controller that tunes the ladder using EMA + RSI signals

How to run the current version
Follow these steps on Windows PowerShell (commands also work on macOS/Linux with path tweaks):

Clone or pull the repo

powershell
git clone https://github.com/ryanoZphoto/alpacatrade.git
Set-Location alpacatrade
If already cloned, run git pull from the project folder to update static assets.

Create and activate a virtual environment

powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
Install Python dependencies

powershell
pip install -r requirements.txt
Configure credentials
Copy and edit the environment file:

powershell
Copy-Item .env.example .env
notepad .env
Paste your Alpaca paper account key/secret. Leave the paper URLs unless you want live trading.

Smoke-test the code

powershell
python -m compileall main.py
Should finish without errors.

Run the FastAPI server

powershell
uvicorn main:app --reload --port 8000
Open the UI and force-refresh
Visit http://127.0.0.1:8000/. Due to browser caching, press Ctrl+F5 (or Cmd+Shift+R on macOS) after pulling changes.

Navigate the tabs in order

Overview: Recommended workflow, account state, recent fills.

Manual Ladder: Bot manual config and deployment.

Autopilot: EMA/RSI controller.

Market Data: Historic candle viewer.

Shut down
Hit Ctrl+C in the terminal when finished.

Quick test (stand-alone)
To verify the project still compiles:

powershell
python -m compileall main.py
UI Map
Overview – Dashboard with fills, metrics, ladder status, and step-by-step workflow.

Manual Ladder – Presets, rung spacing, circuit breakers, preview what deploys.

Autopilot – EMA/RSI controller console, grouped fieldsets, live capital/P&L chart, status timeline.

Market Data – Historic BTC/USD candle viewer.

Autopilot strategy
The UI exposes a Strategy Autopilot section, driving the ladder with:

Dual EMA crossover (fast vs. slow) and RSI levels: decide long/short/flat.

Volatility-sensitive spacing: 1m bar volatility affects rung count and spacing.

Dynamic sizing: USD notionals capped/scaled by user risk.

Key inputs:

Field	Description
Fast/Slow EMA	Trend detection, slow > fast.
RSI window/settings	Avoid entering in overbought/oversold.
Base/step interval	Starting rung spacing/count before volatility adjustment.
Per-rung/Max deploy	USD per rung and total ceiling, converted to BTC.
Volatility lookback	# of 1min bars for volatility estimate.
Risk multiplier	Scales rung/size up/down.
Poll seconds	How often bot reevaluates market data.
Endpoints:

POST /api/start-autopilot – Start/retune controller.

POST /api/stop-autopilot – Stop bot, flatten positions.

GET /api/autopilot-status – UI data snapshot.

Autopilot will flatten manual ladders, recompute and restart as needed if conditions change.

Replace your README.md with the above for proper formatting and clarity. No patching step is needed; simply overwrite the file. If you need a shell command to overwrite it:

powershell
notepad C:\Users\ryano\Desktop\alpacatrade\README.md
Paste the contents and save. Now your project’s README is clear and user-friendly.
