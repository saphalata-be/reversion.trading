#!/usr/bin/env python3
"""
Mean Reversion Web Dashboard
Affiche les données JSON générées par mean_reversion_dashboard.py.

Usage:
    pip install flask
    python app.py
    → http://localhost:5000
"""

import json
from pathlib import Path
from flask import Flask, render_template_string, abort

OUTPUT_DIR = Path(__file__).parent / "output"

app = Flask(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_index():
    with open(OUTPUT_DIR / "index.json", encoding="utf-8") as f:
        return json.load(f)


def _load_symbol(symbol: str):
    p = OUTPUT_DIR / "symbols" / f"{symbol}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _score_cls(score):
    if score is None:   return "s-na"
    if score >= 8:      return "s-excellent"
    if score >= 6:      return "s-good"
    if score >= 4.5:    return "s-moderate"
    if score >= 3:      return "s-weak"
    return "s-trend"


def _score_label(score):
    if score is None:   return "N/A"
    if score >= 7.5:    return "Fort MR"
    if score >= 6.0:    return "Bon MR"
    if score >= 4.5:    return "Modéré MR"
    if score >= 3.0:    return "Faible MR"
    return "Tendance"


def _stars(score):
    if score is None:
        return ""
    n = round(score / 2)
    return "★" * n + "☆" * (5 - n)


def _categorize(symbol: str) -> str:
    if "IDX" in symbol:         return "Indices"
    if symbol.startswith("XA"): return "Métaux"
    if symbol.endswith("USUSD"): return "Actions"
    return "Forex"


def _fmt_val(key: str, val):
    if val is None: return "N/A"
    if key == "hurst":           return f"{val:.3f}"
    if key == "half_life_days":  return f"{val:.1f} j"
    if key == "adf_pvalue":      return f"{val:.4f}"
    if key == "variance_ratio":  return f"{val:.3f}"
    if key == "ou_theta":        return f"{val:.2f}"
    return str(val)


def _fmt_date(iso: str) -> str:
    return iso.replace("T", " ").split(".")[0] + " UTC"


LABEL_COLORS = {
    "s-excellent": "#00e676",
    "s-good":      "#66bb6a",
    "s-moderate":  "#ffd740",
    "s-weak":      "#ffa726",
    "s-trend":     "#ef5350",
    "s-na":        "#8b949e",
}

PERIOD_KEYS  = ["all_history", "10y", "5y", "1y", "6m", "3m"]
METRIC_KEYS  = ["hurst", "half_life_days", "adf_pvalue", "variance_ratio", "ou_theta"]


# ─── CSS partagé ──────────────────────────────────────────────────────────────

BASE_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.5;
}
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }

/* ── Header ── */
.header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 14px 24px;
}
.header h1 { font-size: 1.15rem; font-weight: 600; color: #58a6ff; }
.header .sub { font-size: 0.78rem; color: #8b949e; margin-top: 2px; }

.container { max-width: 1450px; margin: 0 auto; padding: 20px 24px; }

/* ── Score badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    white-space: nowrap;
}
.s-excellent { background: rgba(0,230,118,.12); color: #00e676; border: 1px solid #00e676; }
.s-good      { background: rgba(102,187,106,.12); color: #66bb6a; border: 1px solid #66bb6a; }
.s-moderate  { background: rgba(255,215,64,.12); color: #ffd740; border: 1px solid #ffd740; }
.s-weak      { background: rgba(255,167,38,.12); color: #ffa726; border: 1px solid #ffa726; }
.s-trend     { background: rgba(239,83,80,.12); color: #ef5350; border: 1px solid #ef5350; }
.s-na        { background: rgba(139,148,158,.12); color: #8b949e; border: 1px solid #8b949e; }

/* ── Tables ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
    margin-bottom: 16px;
}
.card-header {
    padding: 10px 16px;
    border-bottom: 1px solid #30363d;
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-header h3 { font-size: 0.9rem; color: #c9d1d9; font-weight: 600; }

.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.tbl th, .tbl td {
    padding: 7px 12px;
    border-bottom: 1px solid #21262d;
    text-align: right;
}
.tbl th {
    background: #0d1117;
    color: #8b949e;
    font-weight: 600;
    text-align: center;
    position: sticky;
    top: 0;
    z-index: 1;
    white-space: nowrap;
}
.tbl th.left, .tbl td.left { text-align: left; }
.tbl tbody tr:hover td { background: rgba(88,166,255,.05); }
.tbl tfoot td {
    border-top: 2px solid #30363d;
    border-bottom: none;
    background: #161b22;
}

/* ── Controls / filters ── */
.controls {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
    flex-wrap: wrap;
    align-items: center;
}
.btn-group { display: flex; gap: 3px; }
.btn {
    padding: 5px 13px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #21262d;
    color: #c9d1d9;
    cursor: pointer;
    font-size: 0.8rem;
    transition: border-color .15s, color .15s;
}
.btn:hover { border-color: #58a6ff; color: #58a6ff; }
.btn.active { background: #1f6feb; border-color: #1f6feb; color: #fff; }

.search-box {
    flex: 1; max-width: 200px;
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid #30363d;
    background: #21262d;
    color: #c9d1d9;
    font-size: 0.8rem;
}
.search-box:focus { outline: none; border-color: #58a6ff; }

/* ── Sortable headers ── */
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: #58a6ff; }
th.sort-asc::after  { content: " ↑"; color: #58a6ff; }
th.sort-desc::after { content: " ↓"; color: #58a6ff; }

/* ── Tabs ── */
.tabs {
    display: flex;
    gap: 0;
    border-bottom: 2px solid #30363d;
    margin-bottom: 16px;
}
.tab-btn {
    padding: 8px 18px;
    cursor: pointer;
    border: none;
    background: none;
    color: #8b949e;
    font-size: 0.88rem;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    display: flex;
    align-items: center;
    gap: 7px;
}
.tab-btn:hover { color: #c9d1d9; }
.tab-btn.active { color: #58a6ff; border-bottom-color: #58a6ff; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Metric cell ── */
.cell-val { font-family: monospace; }
.cell-sub { font-size: 0.72rem; color: #8b949e; }

/* ── Legend ── */
.legend {
    margin-top: 14px;
    padding: 10px 14px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.77rem;
    color: #8b949e;
    line-height: 2;
}
"""


# ─── Templates ────────────────────────────────────────────────────────────────

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mean Reversion Dashboard</title>
<style>{{ css }}</style>
</head>
<body>

<div class="header">
  <h1>Mean Reversion Dashboard</h1>
  <div class="sub">Généré le {{ generated_at }} &nbsp;·&nbsp; {{ total }} symboles</div>
</div>

<div class="container">
  <div class="controls">
    <div class="btn-group" id="cat-btns">
      <button class="btn active" onclick="filterCat(this,'all')">Tous ({{ total }})</button>
      {% for cat, count in cat_counts %}
      <button class="btn" onclick="filterCat(this,'{{ cat }}')">{{ cat }} ({{ count }})</button>
      {% endfor %}
    </div>
    <input class="search-box" type="search" placeholder="Rechercher…" oninput="filterSearch(this.value)">
  </div>

  <div class="card">
    <table class="tbl" id="main-table">
      <thead>
        <tr>
          <th class="left sortable" onclick="sortTable(0)">Symbole</th>
          <th class="sortable" onclick="sortTable(1)">Catégorie</th>
          <th class="sortable" onclick="sortTable(2)">Score Global</th>
          <th class="sortable" onclick="sortTable(3)">1D</th>
          <th class="sortable" onclick="sortTable(4)">4H</th>
          <th class="sortable" onclick="sortTable(5)">1H</th>
          <th>Évaluation</th>
        </tr>
      </thead>
      <tbody id="tbody">
        {% for s in symbols %}
        <tr data-cat="{{ s.cat }}">
          <td class="left">
            <a href="/symbol/{{ s.symbol }}" style="font-family:monospace;font-weight:600">{{ s.symbol }}</a>
          </td>
          <td style="text-align:center;color:#8b949e;font-size:0.78rem">{{ s.cat }}</td>
          <td>
            {% if s.overall_score is not none %}
              <span class="badge {{ s.cls }}">{{ "%.2f"|format(s.overall_score) }}</span>
            {% else %}<span style="color:#8b949e">—</span>{% endif %}
          </td>
          <td>
            {% if s.tf_1d is not none %}
              <span class="badge {{ s.cls_1d }}">{{ "%.2f"|format(s.tf_1d) }}</span>
            {% else %}<span style="color:#8b949e">—</span>{% endif %}
          </td>
          <td>
            {% if s.tf_4h is not none %}
              <span class="badge {{ s.cls_4h }}">{{ "%.2f"|format(s.tf_4h) }}</span>
            {% else %}<span style="color:#8b949e">—</span>{% endif %}
          </td>
          <td>
            {% if s.tf_1h is not none %}
              <span class="badge {{ s.cls_1h }}">{{ "%.2f"|format(s.tf_1h) }}</span>
            {% else %}<span style="color:#8b949e">—</span>{% endif %}
          </td>
          <td style="white-space:nowrap">
            {% if s.overall_score is not none %}
              <span style="color:{{ s.label_color }}">{{ s.stars }} {{ s.label }}</span>
            {% else %}<span style="color:#8b949e">—</span>{% endif %}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <div class="legend">
    <strong>Scores :</strong>
    <span style="color:#00e676">■</span> Fort MR (≥ 7.5) &nbsp;
    <span style="color:#66bb6a">■</span> Bon MR (≥ 6.0) &nbsp;
    <span style="color:#ffd740">■</span> Modéré MR (≥ 4.5) &nbsp;
    <span style="color:#ffa726">■</span> Faible MR (≥ 3.0) &nbsp;
    <span style="color:#ef5350">■</span> Tendance (< 3.0)
    <br>
    <strong>Métriques :</strong>
    Hurst &lt; 0.5 = prix mean-reverting &nbsp;·&nbsp;
    Half-Life court = retour rapide &nbsp;·&nbsp;
    ADF p &lt; 0.05 = série stationnaire &nbsp;·&nbsp;
    VR &lt; 1 = mean-reverting &nbsp;·&nbsp;
    OU theta élevé = retour rapide à la moyenne
  </div>
</div>

<script>
let sortCol = -1, sortDir = 1;
let currentCat = 'all', currentSearch = '';

function filterCat(btn, cat) {
  document.querySelectorAll('#cat-btns .btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  currentCat = cat;
  applyFilters();
}

function filterSearch(val) {
  currentSearch = val.trim().toLowerCase();
  applyFilters();
}

function applyFilters() {
  document.querySelectorAll('#tbody tr').forEach(row => {
    const catOk = currentCat === 'all' || row.dataset.cat === currentCat;
    const sym = row.querySelector('a').textContent.toLowerCase();
    const searchOk = !currentSearch || sym.includes(currentSearch);
    row.style.display = (catOk && searchOk) ? '' : 'none';
  });
}

function sortTable(col) {
  const ths = document.querySelectorAll('#main-table th');
  ths.forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
  if (sortCol === col) sortDir *= -1;
  else { sortCol = col; sortDir = (col <= 1) ? 1 : -1; }
  ths[col].classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');

  const tbody = document.getElementById('tbody');
  const rows = [...tbody.querySelectorAll('tr')];
  rows.sort((a, b) => {
    const ca = getCellVal(a, col), cb = getCellVal(b, col);
    if (typeof ca === 'string') return ca.localeCompare(cb) * sortDir;
    if (isNaN(ca)) return 1;
    if (isNaN(cb)) return -1;
    return (ca - cb) * sortDir;
  });
  rows.forEach(r => tbody.appendChild(r));
}

function getCellVal(row, col) {
  const cells = row.querySelectorAll('td');
  if (col === 0) return cells[0].querySelector('a').textContent.trim();
  if (col === 1) return cells[1].textContent.trim();
  const badge = cells[col] && cells[col].querySelector('.badge');
  return badge ? parseFloat(badge.textContent) : NaN;
}

sortTable(2); // tri initial par Score Global décroissant
</script>
</body>
</html>"""


SYMBOL_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ symbol }} — Mean Reversion</title>
<style>{{ css }}</style>
</head>
<body>

<div class="header">
  <h1>
    <a href="/" style="color:#8b949e;font-size:0.8rem;font-weight:400">← Dashboard</a>
    &ensp; {{ symbol }}
    {% if overall_score is not none %}
      &ensp; <span class="badge {{ score_cls }}">{{ "%.2f"|format(overall_score) }} / 10</span>
      &ensp; <span style="color:{{ label_color }};font-size:0.88rem">{{ stars }} {{ label }}</span>
    {% endif %}
  </h1>
  <div class="sub">Généré le {{ generated_at }} &nbsp;·&nbsp; Catégorie : {{ cat }}</div>
</div>

<div class="container">
  {% if timeframes %}
  <div class="tabs">
    {% for tf in timeframes %}
    <button class="tab-btn {% if loop.first %}active{% endif %}"
            onclick="showTab('{{ tf.key }}', this)">
      {{ tf.label }}
      {% if tf.score is not none %}
        <span class="badge {{ tf.score_cls }}" style="font-size:0.72rem;padding:1px 7px">
          {{ "%.1f"|format(tf.score) }}
        </span>
      {% endif %}
    </button>
    {% endfor %}
  </div>

  {% for tf in timeframes %}
  <div class="tab-panel {% if loop.first %}active{% endif %}" id="tab-{{ tf.key }}">
    <div class="card">
      <div class="card-header">
        <h3>{{ tf.label_long }}</h3>
        {% if tf.score is not none %}
          <span class="badge {{ tf.score_cls }}">Score global : {{ "%.2f"|format(tf.score) }}</span>
        {% endif %}
      </div>
      <table class="tbl">
        <thead>
          <tr>
            <th class="left" style="min-width:155px">Métrique</th>
            {% for p in tf.periods %}
            <th style="min-width:100px">
              {{ p.label }}
              {% if p.bars is not none %}
              <br><span style="font-weight:400;font-size:0.69rem">{{ p.bars }} barres</span>
              {% endif %}
            </th>
            {% endfor %}
            <th style="min-width:90px">Moy.</th>
          </tr>
        </thead>
        <tbody>
          {% for m in tf.metrics %}
          <tr>
            <td class="left" style="font-weight:500">{{ m.label }}</td>
            {% for cell in m.cells %}
            <td>
              {% if cell.val != 'N/A' %}
                <span class="cell-val" style="color:{{ cell.color }}">{{ cell.val }}</span>
                <br><span class="cell-sub">{{ cell.score }}/10</span>
              {% else %}
                <span style="color:#8b949e">N/A</span>
              {% endif %}
            </td>
            {% endfor %}
            <td>
              {% if m.avg is not none %}
                <span class="badge {{ m.avg_cls }}">{{ "%.1f"|format(m.avg) }}</span>
              {% else %}
                <span style="color:#8b949e">—</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
        <tfoot>
          <tr>
            <td class="left" style="font-weight:700;color:#c9d1d9">SCORE GLOBAL</td>
            {% for p in tf.periods %}
            <td>
              {% if p.score is not none %}
                <span class="badge {{ p.score_cls }}">{{ "%.1f"|format(p.score) }}</span>
              {% else %}
                <span style="color:#8b949e">—</span>
              {% endif %}
            </td>
            {% endfor %}
            <td>
              {% if tf.score is not none %}
                <span class="badge {{ tf.score_cls }}">{{ "%.1f"|format(tf.score) }}</span>
              {% else %}
                <span style="color:#8b949e">—</span>
              {% endif %}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  </div>
  {% endfor %}

  {% else %}
  <div style="color:#8b949e;padding:60px;text-align:center">
    Aucune donnée disponible pour ce symbole.
  </div>
  {% endif %}

  <div class="legend">
    <strong>Scores :</strong>
    <span style="color:#00e676">■</span> Fort MR (≥ 7.5) &nbsp;
    <span style="color:#66bb6a">■</span> Bon MR (≥ 6.0) &nbsp;
    <span style="color:#ffd740">■</span> Modéré MR (≥ 4.5) &nbsp;
    <span style="color:#ffa726">■</span> Faible MR (≥ 3.0) &nbsp;
    <span style="color:#ef5350">■</span> Tendance (< 3.0)
    <br>
    <strong>Hurst :</strong> &lt; 0.5 = prix mean-reverting (variance scaling log-prix) &nbsp;·&nbsp;
    <strong>Half-Life :</strong> zone idéale 3–60 jours &nbsp;·&nbsp;
    <strong>ADF p :</strong> &lt; 0.05 = série stationnaire &nbsp;·&nbsp;
    <strong>VR :</strong> &lt; 1 = mean-reverting (Lo-MacKinlay q=10) &nbsp;·&nbsp;
    <strong>OU theta :</strong> élevé = retour rapide à la moyenne (annualisé)
  </div>
</div>

<script>
function showTab(key, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + key).classList.add('active');
  btn.classList.add('active');
}
</script>
</body>
</html>"""


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    idx = _load_index()
    cat_counter: dict[str, int] = {}
    symbols = []

    for s in idx["symbols"]:
        sc = s["overall_score"]
        cls = _score_cls(sc)
        tf = s.get("timeframe_scores", {})
        cat = _categorize(s["symbol"])
        cat_counter[cat] = cat_counter.get(cat, 0) + 1

        symbols.append({
            "symbol":        s["symbol"],
            "overall_score": sc,
            "cls":           cls,
            "stars":         _stars(sc),
            "label":         _score_label(sc),
            "label_color":   LABEL_COLORS[cls],
            "cat":           cat,
            "tf_1d":         tf.get("1D"),
            "cls_1d":        _score_cls(tf.get("1D")),
            "tf_4h":         tf.get("4H"),
            "cls_4h":        _score_cls(tf.get("4H")),
            "tf_1h":         tf.get("1H"),
            "cls_1h":        _score_cls(tf.get("1H")),
        })

    cat_counts = sorted(cat_counter.items(), key=lambda x: x[0])
    return render_template_string(
        INDEX_TEMPLATE,
        css=BASE_CSS,
        symbols=symbols,
        total=len(symbols),
        cat_counts=cat_counts,
        generated_at=_fmt_date(idx["generated_at"]),
    )


def _score_color(score) -> str:
    if score is None:   return "#8b949e"
    if score >= 8:      return "#00e676"
    if score >= 6:      return "#66bb6a"
    if score >= 4.5:    return "#ffd740"
    if score >= 3:      return "#ffa726"
    return "#ef5350"


@app.route("/symbol/<symbol>")
def symbol_detail(symbol):
    data = _load_symbol(symbol)
    if data is None:
        abort(404)

    sc = data.get("overall_score")
    cls = _score_cls(sc)
    generated_at = _fmt_date(data["generated_at"])

    timeframes = []
    for tf_key, tf_data in data.get("timeframes", {}).items():
        periods_data = tf_data.get("periods", {})
        tf_score = tf_data.get("score")

        # Résumé des périodes (pour entête de colonne + footer)
        periods = []
        for pk in PERIOD_KEYS:
            pd = periods_data.get(pk)
            if pd:
                periods.append({
                    "key":       pk,
                    "label":     pd["label"],
                    "bars":      pd.get("bars"),
                    "score":     pd.get("score"),
                    "score_cls": _score_cls(pd.get("score")),
                })
            else:
                periods.append({
                    "key": pk, "label": pk, "bars": None,
                    "score": None, "score_cls": "s-na",
                })

        # Lignes métriques
        metrics = []
        for mk in METRIC_KEYS:
            cells = []
            scores = []
            metric_label = mk

            for pk in PERIOD_KEYS:
                pd = periods_data.get(pk)
                if pd and pd.get("metrics", {}).get(mk):
                    m = pd["metrics"][mk]
                    metric_label = m["label"]
                    sc_cell = m["score"]
                    scores.append(sc_cell)
                    cells.append({
                        "val":   _fmt_val(mk, m["value"]),
                        "score": sc_cell,
                        "color": _score_color(sc_cell),
                    })
                else:
                    cells.append({"val": "N/A", "score": None, "color": "#8b949e"})

            avg = (sum(scores) / len(scores)) if scores else None
            metrics.append({
                "key":     mk,
                "label":   metric_label,
                "cells":   cells,
                "avg":     avg,
                "avg_cls": _score_cls(avg),
            })

        timeframes.append({
            "key":       tf_key,
            "label":     tf_key,
            "label_long": tf_data["label"],
            "score":     tf_score,
            "score_cls": _score_cls(tf_score),
            "periods":   periods,
            "metrics":   metrics,
        })

    return render_template_string(
        SYMBOL_TEMPLATE,
        css=BASE_CSS,
        symbol=symbol,
        overall_score=sc,
        score_cls=cls,
        stars=_stars(sc),
        label=_score_label(sc),
        label_color=LABEL_COLORS[cls],
        cat=_categorize(symbol),
        generated_at=generated_at,
        timeframes=timeframes,
    )


# ─── Lancement ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading
    import webbrowser

    def _open():
        webbrowser.open("http://localhost:5000")

    threading.Timer(1.0, _open).start()
    print("Mean Reversion Dashboard -> http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
