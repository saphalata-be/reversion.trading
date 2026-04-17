"""
SMA Slope Analysis — Interface Web
-----------------------------------
Lance un serveur Flask local pour consulter les résultats de l'analyse
et modifier les paramètres via un formulaire.

Usage:
    python web_app.py
    python web_app.py --port 5001
"""

import argparse
import glob
import json
import os

from flask import Flask, render_template, request

from sma_slope_analysis import (
    analyse_symbol,
    load_symbol_config,
    get_symbol_fee,
    DEFAULT_DIR,
    DEFAULT_TIMEFRAME,
    DEFAULT_PERIOD,
    CONFIG_FILE,
)

app = Flask(__name__)

TIMEFRAMES = ["1m", "5m", "15m", "30m", "1H", "4H", "1D"]
SORT_CHOICES = ["symbol", "avg", "positive", "negative", "abs_avg", "pct_up", "crossings"]
SORT_LABELS = {
    "symbol":    "Symbole",
    "avg":       "Moy. générale",
    "positive":  "Moy. positive",
    "negative":  "Moy. négative",
    "abs_avg":   "Moy. absolue",
    "pct_up":    "% haussier",
    "crossings": "Croisements",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")


def load_latest_strategy_results():
    """Charge le dernier fichier JSON de résultats de stratégie."""
    if not os.path.isdir(OUTPUT_DIR):
        return None
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "strategy_*.json")), reverse=True)
    if not files:
        return None
    with open(files[0], "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.route("/")
def index():
    tf      = request.args.get("timeframe", DEFAULT_TIMEFRAME)
    period  = request.args.get("period",    DEFAULT_PERIOD, type=int)
    sort    = request.args.get("sort",      "symbol")

    if sort not in SORT_CHOICES:
        sort = "symbol"
    if period < 2:
        period = 2

    rows  = []
    error = None

    if not os.path.isdir(DEFAULT_DIR):
        error = f"Répertoire introuvable : {DEFAULT_DIR}"
    else:
        pattern = f"_{tf}.csv"
        files = sorted(f for f in os.listdir(DEFAULT_DIR) if f.endswith(pattern))

        if not files:
            error = f"Aucun fichier trouvé pour le timeframe '{tf}' dans {DEFAULT_DIR}"
        else:
            symbol_config = load_symbol_config(CONFIG_FILE)
            for filename in files:
                symbol   = filename.replace(pattern, "")
                filepath = os.path.join(DEFAULT_DIR, filename)
                fee      = get_symbol_fee(symbol_config, symbol)
                stats    = analyse_symbol(filepath, period, fee)
                if stats:
                    rows.append({"symbol": symbol, **stats})

            if rows:
                ascending = sort == "symbol" or sort not in ("negative",)
                rows.sort(
                    key=lambda r: r[sort],
                    reverse=not ascending,
                )
                if sort == "negative":
                    rows.sort(key=lambda r: r["negative"])

    summary = {}
    if rows:
        for key in ("avg", "positive", "negative", "abs_avg", "pct_up", "crossings"):
            summary[key] = sum(r[key] for r in rows) / len(rows)

    return render_template(
        "index.html",
        timeframes=TIMEFRAMES,
        sort_labels=SORT_LABELS,
        selected_tf=tf,
        selected_period=period,
        selected_sort=sort,
        rows=rows,
        summary=summary,
        data_dir=DEFAULT_DIR,
        error=error,
    )


_TF_ORDER = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1H": 4, "4H": 5, "1D": 6}


@app.route("/strategy")
def strategy():
    selected_symbol = request.args.get("symbol", "")
    data = load_latest_strategy_results()

    symbols = []
    rows_by_symbol: dict = {}
    generated_at = None

    _long_short_keys = [
        "bt40_long", "bt30_long", "bt20_long", "bt10_long",
        "bt40_short", "bt30_short", "bt20_short", "bt10_short",
    ]
    if data:
        generated_at = data.get("generated_at")
        for row in data["rows"]:
            # Rétro-compatibilité : fichiers antérieurs sans colonnes long/short
            for k in _long_short_keys:
                row.setdefault(k, 0.0)
            sym = row["symbol"]
            if sym not in rows_by_symbol:
                rows_by_symbol[sym] = []
            rows_by_symbol[sym].append(row)
        symbols = sorted(rows_by_symbol.keys())

    if not selected_symbol and symbols:
        selected_symbol = symbols[0]

    selected_rows = sorted(
        rows_by_symbol.get(selected_symbol, []),
        key=lambda r: (_TF_ORDER.get(r["timeframe"], 99), r["period"]),
    )

    return render_template(
        "strategy.html",
        symbols=symbols,
        selected_symbol=selected_symbol,
        rows=selected_rows,
        generated_at=generated_at,
        has_data=bool(data),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMA Slope Analysis — Interface Web")
    parser.add_argument("--port", type=int, default=5000, help="Port du serveur (défaut: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Serveur démarré : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
