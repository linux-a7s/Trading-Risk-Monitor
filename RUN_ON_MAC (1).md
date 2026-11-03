# Trading Risk Monitor: run on your Mac

## One-time setup (Terminal)

```bash
cd ~/Desktop/trading_risk_dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Start the dashboard

```bash
source .venv/bin/activate
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. If it doesn't, open that link yourself.

## Use it

1. In the left sidebar, upload `Tradebook-Kite.xlsx`.
2. Wait about 10 to 30 seconds while it trains the five models.
3. Explore the four tabs:
   - **Overview**: KPIs, trades per month, risky vs normal, top symbols.
   - **Model comparison**: all five algorithms, the top two, and the final model.
   - **Trade explorer**: filter trades, then pick one to see which warning signs fired.
   - **How it works**: the pipeline and the limits.
4. Use the sidebar to filter by risky/normal, show anomalies only, or change the alert threshold.

## Stop it

Press `Control + C` in the Terminal.

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: streamlit` | Run `source .venv/bin/activate` first. |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502`. |
| `Sheet 'Trade Book' not found` | Upload the original Kite tradebook export, not a re-saved copy with renamed sheets. |
| Slow first load | Normal. The models train once per file and are then cached. |
