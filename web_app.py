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
import os

from flask import Flask, render_template_string, request

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
SORT_CHOICES = ["symbol", "avg", "positive", "negative", "abs_avg", "pct_up",
                "crossings", "p40", "p30", "p20", "p10",
                "bt40", "bt30", "bt20", "bt10"]
SORT_LABELS = {
    "symbol":   "Symbole",
    "avg":      "Moy. générale",
    "positive": "Moy. positive",
    "negative": "Moy. négative",
    "abs_avg":  "Moy. absolue",
    "pct_up":   "% haussier",
    "crossings": "Croisements",
    "p40": "Top 40%",
    "p30": "Top 30%",
    "p20": "Top 20%",
    "p10": "Top 10%",
    "bt40": "BT top40%",
    "bt30": "BT top30%",
    "bt20": "BT top20%",
    "bt10": "BT top10%",
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SMA Slope Analysis</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0f1117;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem 1rem;
    }

    h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: #38bdf8;
      margin-bottom: 1.5rem;
      letter-spacing: 0.02em;
    }

    /* ── Form ──────────────────────────────────────────────── */
    form.params {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem 1.5rem;
      align-items: flex-end;
      background: #1e2130;
      border: 1px solid #2d3148;
      border-radius: 10px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 2rem;
    }

    .field { display: flex; flex-direction: column; gap: 0.35rem; }

    label {
      font-size: 0.78rem;
      font-weight: 600;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    select, input[type="number"] {
      background: #0f1117;
      border: 1px solid #374151;
      border-radius: 6px;
      color: #e2e8f0;
      padding: 0.45rem 0.75rem;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.15s;
    }
    select:focus, input[type="number"]:focus { border-color: #38bdf8; }
    input[type="number"] { width: 90px; }

    button[type="submit"] {
      background: #0ea5e9;
      color: #fff;
      border: none;
      border-radius: 6px;
      padding: 0.5rem 1.5rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
      height: 38px;
    }
    button[type="submit"]:hover { background: #38bdf8; }

    /* ── Meta info ─────────────────────────────────────────── */
    .meta {
      font-size: 0.85rem;
      color: #64748b;
      margin-bottom: 1rem;
    }
    .meta span { color: #94a3b8; font-weight: 600; }

    /* ── Error ─────────────────────────────────────────────── */
    .error {
      color: #f87171;
      background: #1e1a1a;
      border: 1px solid #7f1d1d;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 1.5rem;
    }

    /* ── Table ─────────────────────────────────────────────── */
    .table-wrap {
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid #2d3148;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }

    thead tr {
      background: #1e2130;
    }

    th {
      padding: 0.7rem 1rem;
      text-align: right;
      font-size: 0.75rem;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }
    th:first-child { text-align: left; }
    th a { color: inherit; text-decoration: none; }
    th.sorted-asc::after  { content: " ▲"; color: #38bdf8; }
    th.sorted-desc::after { content: " ▼"; color: #38bdf8; }

    tbody tr { border-top: 1px solid #1e2130; transition: background 0.1s; }
    tbody tr:hover { background: #1a1f30; }

    td {
      padding: 0.55rem 1rem;
      text-align: right;
      white-space: nowrap;
      color: #cbd5e1;
    }
    td:first-child { text-align: left; font-weight: 600; color: #e2e8f0; }

    .pos { color: #4ade80; }
    .neg { color: #f87171; }

    /* Summary row */
    tfoot tr { border-top: 2px solid #2d3148; background: #1e2130; }
    tfoot td { padding: 0.65rem 1rem; font-weight: 700; color: #94a3b8; }
  </style>
</head>
<body>
  <h1>SMA Slope Analysis</h1>

  <form class="params" method="get" action="/">
    <div class="field">
      <label for="timeframe">Timeframe</label>
      <select id="timeframe" name="timeframe">
        {% for tf in timeframes %}
          <option value="{{ tf }}" {% if tf == selected_tf %}selected{% endif %}>{{ tf }}</option>
        {% endfor %}
      </select>
    </div>

    <div class="field">
      <label for="period">Période SMA</label>
      <input id="period" name="period" type="number" min="2" max="500"
             value="{{ selected_period }}" />
    </div>

    <div class="field">
      <label for="sort">Tri</label>
      <select id="sort" name="sort">
        {% for col, lbl in sort_labels.items() %}
          <option value="{{ col }}" {% if col == selected_sort %}selected{% endif %}>{{ lbl }}</option>
        {% endfor %}
      </select>
    </div>

    <button type="submit">Analyser</button>
  </form>

  {% if error %}
    <div class="error">{{ error }}</div>
  {% endif %}

  {% if rows %}
    <p class="meta">
      Répertoire : <span>{{ data_dir }}</span> &nbsp;|&nbsp;
      Symboles analysés : <span>{{ rows|length }}</span>
    </p>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            {% set cols = [
              ("symbol",    "Symbole"),
              ("bars",      "Bars"),
              ("avg",       "Moy. générale"),
              ("positive",  "Moy. positive"),
              ("negative",  "Moy. négative"),
              ("abs_avg",   "Moy. absolue"),
              ("pct_up",    "% haussier"),
              ("crossings", "Croisements"),
              ("p40",       "Top 40%"),
              ("p30",       "Top 30%"),
              ("p20",       "Top 20%"),
              ("p10",       "Top 10%"),
              ("bt40",      "BT top40%"),
              ("bt30",      "BT top30%"),
              ("bt20",      "BT top20%"),
              ("bt10",      "BT top10%"),
            ] %}
            {% for col, label in cols %}
              <th class="{{ 'sorted-asc' if (selected_sort == col and col == 'symbol') else ('sorted-desc' if selected_sort == col else '') }}">
                <a href="/?timeframe={{ selected_tf }}&period={{ selected_period }}&sort={{ col }}">{{ label }}</a>
              </th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
          {% for row in rows %}
          <tr>
            <td>{{ row.symbol }}</td>
            <td>{{ row.bars }}</td>
            <td class="{{ 'pos' if row.avg > 0 else 'neg' }}">{{ '%+.6f' % row.avg }}%</td>
            <td class="pos">{{ '%+.6f' % row.positive }}%</td>
            <td class="neg">{{ '%+.6f' % row.negative }}%</td>
            <td>{{ '%.6f' % row.abs_avg }}%</td>
            <td>{{ '%.1f' % row.pct_up }}%</td>
            <td>{{ row.crossings }}</td>
            <td>{{ '%.4f' % row.p40 }}%</td>
            <td>{{ '%.4f' % row.p30 }}%</td>
            <td>{{ '%.4f' % row.p20 }}%</td>
            <td>{{ '%.4f' % row.p10 }}%</td>
            <td class="{{ 'pos' if row.bt40 > 0 else 'neg' }}">{{ '%+.2f' % row.bt40 }}%</td>
            <td class="{{ 'pos' if row.bt30 > 0 else 'neg' }}">{{ '%+.2f' % row.bt30 }}%</td>
            <td class="{{ 'pos' if row.bt20 > 0 else 'neg' }}">{{ '%+.2f' % row.bt20 }}%</td>
            <td class="{{ 'pos' if row.bt10 > 0 else 'neg' }}">{{ '%+.2f' % row.bt10 }}%</td>
          </tr>
          {% endfor %}
        </tbody>
        <tfoot>
          <tr>
            <td>MOYENNE GLOBALE</td>
            <td></td>
            <td class="{{ 'pos' if summary.avg > 0 else 'neg' }}">{{ '%+.6f' % summary.avg }}%</td>
            <td class="pos">{{ '%+.6f' % summary.positive }}%</td>
            <td class="neg">{{ '%+.6f' % summary.negative }}%</td>
            <td>{{ '%.6f' % summary.abs_avg }}%</td>
            <td>{{ '%.1f' % summary.pct_up }}%</td>
            <td>{{ '%.1f' % summary.crossings }}</td>
            <td>{{ '%.4f' % summary.p40 }}%</td>
            <td>{{ '%.4f' % summary.p30 }}%</td>
            <td>{{ '%.4f' % summary.p20 }}%</td>
            <td>{{ '%.4f' % summary.p10 }}%</td>
            <td class="{{ 'pos' if summary.bt40 > 0 else 'neg' }}">{{ '%+.2f' % summary.bt40 }}%</td>
            <td class="{{ 'pos' if summary.bt30 > 0 else 'neg' }}">{{ '%+.2f' % summary.bt30 }}%</td>
            <td class="{{ 'pos' if summary.bt20 > 0 else 'neg' }}">{{ '%+.2f' % summary.bt20 }}%</td>
            <td class="{{ 'pos' if summary.bt10 > 0 else 'neg' }}">{{ '%+.2f' % summary.bt10 }}%</td>
          </tr>
        </tfoot>
      </table>
    </div>
  {% endif %}
</body>
</html>
"""


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
        for key in ("avg", "positive", "negative", "abs_avg", "pct_up",
                    "crossings", "p40", "p30", "p20", "p10",
                    "bt40", "bt30", "bt20", "bt10"):
            summary[key] = sum(r[key] for r in rows) / len(rows)

    return render_template_string(
        HTML_TEMPLATE,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMA Slope Analysis — Interface Web")
    parser.add_argument("--port", type=int, default=5000, help="Port du serveur (défaut: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Serveur démarré : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
