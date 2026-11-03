# Moonshot

Practice trading bot for MEXC futures (fake money first) plus a simple dashboard.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your MEXC keys in `.env` if you have them. Keep `LIVE_TRADING_ENABLED=0`.

## Everyday use

**Start (survives Cursor + auto-restarts):**
```bash
./install_keepalive.sh
```

Open **http://127.0.0.1:8501/live**

Stop: `./stop_moonshot.sh` · fully remove: `./uninstall_keepalive.sh`

Never enables real-money trading. (`LIVE_TRADING_ENABLED` must stay `0`.)
Logs: `logs/autopilot.log`, `logs/dashboard.log`

Open **http://127.0.0.1:8501**

Nav: **Live** · **Trades** · **Backtest** · **Status** · **Health** · **About**

You do not need to change settings. Real money stays off.

Autopilot hunt researches on **Binance spot** calendar windows (~180d confirm, deeper finalist under the bar cap; same wall-clock for every TF). Live/paper still use MEXC. Override with `HUNT_PRICE_SOURCE=mexc` in `.env` if needed.

## Safety

Educational software. Leveraged futures can lose the entire balance. Nothing here is financial advice.
