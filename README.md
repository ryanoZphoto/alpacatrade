# Alpaca BTC‑Ladder Prototype

This project scaffolds a small FastAPI app that:
- pulls the latest BTC‑USD price from Alpaca’s crypto‑bars endpoint
- builds a grid (ladder) of limit orders
- keeps the ladder alive (cancel stray, place missing)
- tracks fills, net position, and realized P&L
- serves a simple web UI (chart + config + live status/logs)
- includes an automated “autopilot” controller that tunes the ladder using EMA + RSI signals

## How to run

1. Create/activate a virtual environment
2. Install deps: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill credentials
4. Start server: `uvicorn main:app --reload --port 8000`

## UI map

- **Overview** – landing dashboard with the latest fills, account metrics, ladder status, and a “Run order” card that reminds you of the recommended workflow (prep ladder → choose manual or autopilot → monitor).
- **Manual Ladder** – presets, rung spacing controls, circuit breaker switches, and preview cards for what will be deployed when you press **Start Ladder**.
- **Autopilot** – a dedicated console for the EMA/RSI controller with grouped fieldsets, button reference, telemetry readouts, and a live capital vs. P/L chart.
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
