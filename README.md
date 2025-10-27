# Alpaca BTC‑Ladder Prototype

This project scaffolds a small FastAPI app that:
- pulls the latest BTC‑USD price from Alpaca’s crypto‑bars endpoint
- builds a grid (ladder) of limit orders
- keeps the ladder alive (cancel stray, place missing)
- tracks fills, net position, and realized P&L
- serves a simple web UI (chart + config + live status/logs)
- includes an automated “autopilot” controller that tunes the ladder using EMA + RSI signals

## How to run the current version

Follow the exact steps below on Windows PowerShell (the same commands work on macOS/Linux with the usual path tweaks):

1. **Clone or pull the repo**
   ```powershell
   git clone https://github.com/ryanoZphoto/alpacatrade.git
   Set-Location alpacatrade
   ```
   If you already cloned it, run `git pull` and stay inside the project folder so the new static assets are available.

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Python dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure credentials**
   Copy the sample environment file and drop your Alpaca paper keys inside:
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```
   Paste the paper account key/secret from your Alpaca dashboard, save, and close Notepad. Leave the paper URLs as-is unless you intend to hit the live environment.

5. **Smoke-test the code**
   ```powershell
   python -m compileall main.py
   ```
   This is the repo’s quick “does it start?” check. It should finish without errors.

6. **Run the FastAPI server**
   ```powershell
   uvicorn main:app --reload --port 8000
   ```

7. **Open the UI (and force-refresh once)**
   Visit <http://127.0.0.1:8000/>. Because browsers cache `/static/script.js` and `/static/style.css`, press **Ctrl+F5** (or **Cmd+Shift+R** on macOS) the first time after pulling new changes to ensure you see the latest tabbed layout.

8. **Navigate the tabs in order**
   *Overview* lists the recommended workflow, account state, and recent fills.
   Move to *Manual Ladder* if you want to configure the bot yourself, or jump straight to the dedicated *Autopilot* tab to let the EMA/RSI controller manage the ladder. The *Market Data* tab gives you historical context.

9. **Shut down**
   Hit `Ctrl+C` in the terminal to stop Uvicorn when you are done.

### Quick test (stand-alone)

If you only need to verify the project still compiles before committing, run:

```powershell
python -m compileall main.py
```

## UI map

- **Overview** – landing dashboard with the latest fills, account metrics, ladder status, and a “Run order” card that reminds you of the recommended workflow (prep ladder → choose manual or autopilot → monitor).
- **Manual Ladder** – presets, rung spacing controls, circuit breaker switches, and preview cards for what will be deployed when you press **Start Ladder**.
- **Autopilot** – a dedicated console for the EMA/RSI controller with grouped fieldsets, status cards, a decision timeline, telemetry readouts, and a live capital vs. P/L chart.
- **Market Data** – ad-hoc historical candle viewer for quick context around the current BTC/USD trend.

## Autopilot strategy

The UI exposes a *Strategy Autopilot* section. It drives the ladder with:

- Dual EMA crossover (fast vs. slow) combined with RSI levels to decide long/short/flat.
- Volatility-sensitive spacing: 1-minute bar volatility widens or tightens rung spacing and adjusts step count.
- Dynamic sizing: USD notionals are capped by your maximum deployment and scaled by a risk multiplier.

Key inputs:

| Field | Description |
| --- | --- |
| Fast/Slow EMA | Windows for trend direction. Slow must exceed fast. |
| RSI window / thresholds | Helps avoid buying into overbought or selling into oversold. |
| Base interval / steps | Baseline rung spacing and count before volatility adjustments. |
| Per-rung / Max deployment (USD) | Gross capital per rung and ceiling for the ladder. Converted to BTC using the latest price. |
| Volatility lookback | Number of 1-minute bars used for volatility estimates. |
| Risk multiplier | Scales interval/size responses up or down. |
| Poll seconds | How often the strategy re-evaluates market data. |

The backend exposes matching endpoints:

- `POST /api/start-autopilot` – start or retune the controller.
- `POST /api/stop-autopilot` – cancel the controller and flatten ladder management.
- `GET /api/autopilot-status` – JSON snapshot used by the UI.

When the autopilot is running it will automatically stop any manual ladder instance, compute a new ladder config, and restart the bot as conditions change.

## Package the current snapshot (for PR handoff)

If you need to hand someone the exact files from this branch without pushing to GitHub, run the bundler script. It zips every tracked file so the recipient can unzip and run the app without using Git.

```powershell
python export_pr_bundle.py
```

The archive is written to `dist/` and includes the branch name, short commit hash, and a UTC timestamp in the filename.
