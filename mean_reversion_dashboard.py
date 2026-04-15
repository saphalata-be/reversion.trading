#!/usr/bin/env python3
"""
Mean Reversion Dashboard
========================
Analyse statistique multi-timeframe pour le trading mean reversion.
Un tableau par symbole et par timeframe (1D / H4 / H1).
Colonnes : Historique complet | 10 ans | 5 ans | 1 an | 6 mois | 3 mois | Score

Métriques calculées :
  - Hurst Exponent     (variance scaling sur log-prix, approche Ernie Chan)
  - Half-Life          (régression OU, converti en jours)
  - ADF p-value        (test racine unitaire sur log-prix)
  - Variance Ratio     (Lo-MacKinlay, q=10, sur log-returns)
  - OU theta (speed)   (vitesse de retour à la moyenne, annualisée)

Usage :
  python mean_reversion_dashboard.py                 # tous les symboles
  python mean_reversion_dashboard.py EURUSD GBPUSD   # symboles spécifiques
  python mean_reversion_dashboard.py --list          # lister les symboles disponibles
  python mean_reversion_dashboard.py EURUSD --export eurusd.html
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[WARN] 'rich' non installé — pip install rich")

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────

HISTORY_DIR = r"E:\Forex\History\TickStory"

# Nombre de barres par jour de trading (forex ~24h/jour)
BARS_PER_DAY: dict[str, float] = {
    "1D": 1.0,
    "4H": 6.0,
    "1H": 24.0,
}

# Nombre de barres par an (252 jours de trading)
BARS_PER_YEAR: dict[str, float] = {
    "1D": 252.0,
    "4H": 252.0 * 6,
    "1H": 252.0 * 24,
}

# Timeframes à analyser : (label_affichage, code_fichier)
TIMEFRAMES: list[tuple[str, str]] = [
    ("Journalier (1D)", "1D"),
    ("4 Heures  (H4)",  "4H"),
    ("1 Heure   (H1)",  "1H"),
]

# Périodes à analyser : (label_affichage, jours_lookback | None = tout l'historique)
PERIODS: list[tuple[str, int | None]] = [
    ("Historique", None),
    ("10 ans",     365 * 10),
    ("5 ans",      365 * 5),
    ("1 an",       365),
    ("6 mois",     183),
    ("3 mois",     91),
]

# Clés JSON stables pour chaque période et métrique
PERIOD_KEYS:  list[str] = ["all_history", "10y", "5y", "1y", "6m", "3m"]
METRIC_NAMES: list[str] = [
    "Hurst Exponent",
    "Half-Life (jours)",
    "ADF p-value",
    "Variance Ratio",
    "OU theta (speed)",
]
METRIC_KEYS: list[str] = [
    "hurst",
    "half_life_days",
    "adf_pvalue",
    "variance_ratio",
    "ou_theta",
]

# q pour le Variance Ratio
VR_Q = 10

# Fenêtre glissante (1 an) et pas hebdomadaire pour l'historique de scores
HISTORY_WINDOW_BARS: dict[str, int] = {"1D": 252,  "4H": 1512,  "1H": 6048}
HISTORY_STEP_BARS:   dict[str, int] = {"1D": 5,    "4H": 30,    "1H": 120}

# ─── Calcul des métriques ──────────────────────────────────────────────────────

def hurst_rs(prices: np.ndarray) -> float:
    """
    Exposant de Hurst via variance scaling sur les log-prix (approche Ernie Chan).
    Mesure comment std(log(P[t+τ]) - log(P[t])) évolue avec τ :
      std(τ) ∝ τ^H  →  H = pente de log(std) vs log(τ)

    H < 0.5 : le PRIX lui-même est mean-reverting (std croît moins vite que √τ)
    H = 0.5 : marche aléatoire (std ∝ √τ)
    H > 0.5 : prix tendanciel (std croît plus vite que √τ)
    """
    lp = np.log(prices)
    n = len(lp)
    if n < 40:
        return np.nan

    max_lag = min(n // 4, 200)
    if max_lag < 4:
        return np.nan

    lags = np.unique(np.geomspace(2, max_lag, 20).astype(int))
    tau = np.array([np.std(lp[lag:] - lp[:-lag]) for lag in lags])

    # Filtre les éventuels std nuls
    valid = tau > 0
    if valid.sum() < 3:
        return np.nan

    try:
        return float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0])
    except Exception:
        return np.nan


def half_life(prices: np.ndarray, tf: str) -> float:
    """
    Half-life en JOURS via régression Ornstein-Uhlenbeck sur les log-prix.
    ΔP_t = α + β·P_{t-1} + ε  →  HL = -ln(2)/β / bars_per_day
    """
    lp = np.log(prices)
    dy = np.diff(lp)
    y = lp[:-1]

    X = np.column_stack([np.ones(len(y)), y])
    try:
        b = np.linalg.lstsq(X, dy, rcond=None)[0]
    except Exception:
        return np.nan

    if b[1] >= 0:
        return np.nan  # pas de mean reversion

    hl_bars = -np.log(2) / b[1]
    hl_days = hl_bars / BARS_PER_DAY[tf]

    return float(hl_days) if 0 < hl_days < 50_000 else np.nan


def adf_pvalue(prices: np.ndarray, maxlag: int | None = None) -> float:
    """
    ADF p-value sur les log-prix.
    p < 0.05 : série stationnaire (mean-reverting)
    """
    lp = np.log(prices)
    if len(lp) < 20:
        return np.nan
    try:
        kw = {"autolag": "AIC"}
        if maxlag is not None:
            kw["maxlag"] = maxlag
        return float(adfuller(lp, **kw)[1])
    except Exception:
        return np.nan


def variance_ratio(prices: np.ndarray, q: int = VR_Q) -> float:
    """
    Variance Ratio (Lo-MacKinlay) sur les log-returns.
    VR < 1 : mean-reverting | VR = 1 : RW | VR > 1 : tendance
    """
    if len(prices) < q * 5:
        return np.nan

    r = np.diff(np.log(prices))
    n = len(r)
    v1 = np.var(r, ddof=1)
    if v1 == 0:
        return np.nan

    # Somme glissante vectorisée via cumsum (O(n), beaucoup plus rapide que boucle Python)
    cs = np.zeros(n + 1)
    np.cumsum(r, out=cs[1:])
    rq = cs[q:] - cs[:-q]
    vq = np.var(rq, ddof=1)

    return float(vq / (q * v1))


def ou_theta(prices: np.ndarray, tf: str) -> float:
    """
    Vitesse de retour OU θ annualisée.
    θ = vitesse de retour à la moyenne (plus θ est élevé, plus c'est rapide).
    """
    lp = np.log(prices)
    dy = np.diff(lp)
    y = lp[:-1]

    X = np.column_stack([np.ones(len(y)), y])
    try:
        b = np.linalg.lstsq(X, dy, rcond=None)[0]
    except Exception:
        return np.nan

    if b[1] >= 0:
        return np.nan

    return float(-b[1] * BARS_PER_YEAR[tf])


# ─── Scoring (0-10) ────────────────────────────────────────────────────────────

def score_metric(val: float, m_idx: int) -> float:
    """Retourne un score de 0 (tendance) à 10 (forte mean reversion)."""
    if not np.isfinite(val):
        return 0.0

    if m_idx == 0:  # Hurst (plus bas = mieux)
        if val <= 0.35: return 10.0
        if val <= 0.40: return 9.0
        if val <= 0.45: return 7.0
        if val <= 0.48: return 5.0
        if val <= 0.50: return 3.0
        if val <= 0.55: return 1.0
        return 0.0

    if m_idx == 1:  # Half-life en jours (zone idéale : 3-60j)
        if   3  <= val <= 60:  return 10.0
        if   1  <= val <  3:   return 6.0
        if  60  <  val <= 120: return 6.0
        if 120  <  val <= 250: return 3.0
        if   0  <  val <  1:   return 2.0
        return 0.0

    if m_idx == 2:  # ADF p-value (plus bas = mieux)
        if val < 0.01: return 10.0
        if val < 0.05: return 8.0
        if val < 0.10: return 5.0
        if val < 0.20: return 2.0
        return 0.0

    if m_idx == 3:  # Variance Ratio (moins de 1 = mieux)
        if val < 0.70: return 10.0
        if val < 0.80: return 8.0
        if val < 0.90: return 6.0
        if val < 0.95: return 4.0
        if val < 1.00: return 2.0
        return 0.0

    if m_idx == 4:  # OU theta annualisé (plus élevé = mieux)
        if val > 15: return 10.0
        if val > 8:  return 9.0
        if val > 4:  return 7.0
        if val > 2:  return 5.0
        if val > 1:  return 3.0
        if val > 0:  return 1.0
        return 0.0

    return 0.0


# ─── Formatage Rich ────────────────────────────────────────────────────────────

def _color(sc: float) -> str:
    if sc >= 8: return "bright_green"
    if sc >= 6: return "green"
    if sc >= 4: return "yellow"
    if sc >= 2: return "orange3"
    return "red"


def _stars(sc: float) -> str:
    n = int(round(sc / 2))
    return "★" * n + "☆" * (5 - n)


def _fmt_cell(val: float, m_idx: int, sc: float) -> str:
    if not np.isfinite(val):
        return "[dim]N/A[/dim]"
    c = _color(sc)
    if m_idx == 0:  formatted = f"{val:.3f}"
    elif m_idx == 1: formatted = f"{val:.1f} j"
    elif m_idx == 2: formatted = f"{val:.4f}"
    elif m_idx == 3: formatted = f"{val:.3f}"
    else:            formatted = f"{val:.2f}"
    return f"[{c}]{formatted}[/{c}]"


def _fmt_eval(avg_sc: float) -> str:
    c = _color(avg_sc)
    label = (
        "Fort MR"        if avg_sc >= 7.5 else
        "Bon MR"         if avg_sc >= 6.0 else
        "Modéré MR"      if avg_sc >= 4.5 else
        "Faible MR"      if avg_sc >= 3.0 else
        "Tendance"
    )
    return f"[{c}]{avg_sc:.1f}/10  {_stars(avg_sc)}\n[dim]{label}[/dim][/{c}]"


# ─── Cache & chargement ────────────────────────────────────────────────────────

_cache: dict[tuple[str, str], pd.DataFrame | None] = {}


def _load_df(symbol: str, tf: str) -> pd.DataFrame | None:
    key = (symbol, tf)
    if key not in _cache:
        path = os.path.join(HISTORY_DIR, f"{symbol}_{tf}.csv")
        if not os.path.exists(path):
            _cache[key] = None
        else:
            try:
                df = pd.read_csv(
                    path,
                    usecols=["Datetime", "Close"],
                    parse_dates=["Datetime"],
                )
                df.sort_values("Datetime", inplace=True)
                df.reset_index(drop=True, inplace=True)
                _cache[key] = df
            except Exception as exc:
                print(f"[WARN] Impossible de lire {path} : {exc}")
                _cache[key] = None
    return _cache[key]


def get_series(
    symbol: str, tf: str, days: int | None
) -> tuple[np.ndarray | None, dict | None]:
    """
    Charge les prix de clôture filtrés sur les `days` derniers jours.
    Si days est None, retourne tout l'historique disponible.

    Retourne (prices, meta) où meta = {"bars": N, "from": iso, "to": iso},
    ou (None, None) si les données sont insuffisantes.
    """
    df = _load_df(symbol, tf)
    if df is None:
        return None, None

    if days is None:
        sub = df
    else:
        ref = df["Datetime"].iloc[-1]
        cutoff = ref - pd.Timedelta(days=days)
        sub = df.loc[df["Datetime"] >= cutoff]

    arr = sub["Close"].dropna().values
    if len(arr) < 40:
        return None, None

    meta = {
        "bars": int(len(arr)),
        "from": sub["Datetime"].iloc[0].isoformat(),
        "to":   sub["Datetime"].iloc[-1].isoformat(),
    }
    return arr, meta


# ─── Calcul par symbole × timeframe ───────────────────────────────────────────

def compute_timeframe(
    symbol: str, tf: str
) -> tuple[
    list[list[tuple[float, float]] | None],
    list[dict | None],
]:
    """
    Pour un symbole et un timeframe donnés, calcule les 5 métriques sur
    chacune des périodes définies dans PERIODS.

    Retourne (data, metas) où :
      data[i][m] = (valeur, score) pour la période i, métrique m
      metas[i]   = {"bars": N, "from": iso, "to": iso} | None
    """
    data:  list[list[tuple[float, float]] | None] = []
    metas: list[dict | None] = []

    for _, days in PERIODS:
        series, meta = get_series(symbol, tf, days)
        if series is None:
            data.append(None)
            metas.append(None)
            continue

        vals = [
            hurst_rs(series),
            half_life(series, tf),
            adf_pvalue(series),
            variance_ratio(series),
            ou_theta(series, tf),
        ]
        col = [(v, score_metric(v, m)) for m, v in enumerate(vals)]
        data.append(col)
        metas.append(meta)

    return data, metas


# ─── Historique des scores par période ────────────────────────────────────────

# Jours de lookback par clé de période (None = fenêtre croissante depuis l'origine)
PERIOD_HISTORY_DAYS: dict[str, int | None] = {
    "all_history": None,
    "10y": 365 * 10,
    "5y":  365 * 5,
    "1y":  365,
    "6m":  183,
    "3m":  91,
}

# Nombre max de points de sortie par période (step adaptatif pour éviter des calculs trop longs)
_MAX_PTS: dict[str, int] = {
    "all_history": 100,
    "10y":         150,
    "5y":          200,
    "1y":          500,
    "6m":          500,
    "3m":          500,
}

# Tailles max pour les métriques lentes dans l'historique
_ADF_MAX_BARS   = 3000   # sous-échantillonnage pour ADF (stationnarité préservée)
_HURST_MAX_BARS = 5000   # sous-échantillonnage pour Hurst (H est invariant d'échelle)


def _subsample(arr: np.ndarray, max_bars: int) -> np.ndarray:
    """Sous-échantillonne arr à max_bars points équidistants."""
    if len(arr) <= max_bars:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, max_bars)).astype(int)
    return arr[idx]


def compute_period_score_history(symbol: str, tf: str) -> dict:
    """
    Calcule l'évolution du score global (overall) semaine par semaine pour
    chacune des périodes définies dans PERIOD_HISTORY_DAYS.

    - "all_history" : fenêtre croissante depuis le début de l'historique.
    - Autres périodes : fenêtre glissante de taille fixe.

    Seul le score global (moyenne des 5 métriques) est retenu.
    Les métriques lentes (ADF, Hurst) sont calculées sur une version
    sous-échantillonnée pour maintenir des temps de calcul raisonnables.

    Retourne un dict :
      {
        "all_history": {"dates": [...], "overall": [...]},
        "10y":         {"dates": [...], "overall": [...]},
        ...
      }
    ou {} si les données sont insuffisantes.
    """
    df = _load_df(symbol, tf)
    if df is None:
        return {}

    prices_all = df["Close"].dropna().values
    dates_all  = df["Datetime"].values
    n = len(prices_all)

    base_step = HISTORY_STEP_BARS.get(tf, 5)
    min_bars  = 40

    if n < min_bars:
        return {}

    result: dict = {}

    for p_key, days in PERIOD_HISTORY_DAYS.items():
        window_bars    = None if days is None else int(days * BARS_PER_DAY[tf])
        max_pts        = _MAX_PTS.get(p_key, 500)
        effective_step = max(base_step, n // max_pts)

        dates_out:   list[str]   = []
        overall_out: list[float] = []

        for end_idx in range(min_bars, n + 1, effective_step):
            start_idx = 0 if window_bars is None else max(0, end_idx - window_bars)
            prices = prices_all[start_idx:end_idx]
            if len(prices) < min_bars:
                continue

            date_str = str(pd.Timestamp(dates_all[end_idx - 1]).date())

            # Métriques lentes : sous-échantillonnage pour la performance
            p_adf   = _subsample(prices, _ADF_MAX_BARS)
            p_hurst = _subsample(prices, _HURST_MAX_BARS)

            vals = [
                hurst_rs(p_hurst),
                half_life(prices, tf),
                adf_pvalue(p_adf, maxlag=10),   # maxlag limité pour la performance
                variance_ratio(prices),
                ou_theta(prices, tf),
            ]
            m_scores = [score_metric(v, i) for i, v in enumerate(vals)]

            dates_out.append(date_str)
            overall_out.append(round(float(np.mean(m_scores)), 2))

        if dates_out:
            result[p_key] = {
                "dates":   dates_out,
                "overall": overall_out,
            }

    return result


# ─── Rendu Rich ────────────────────────────────────────────────────────────────

_LEGEND = (
    "[dim]  Hurst<0.5=prix MR (variance scaling log-prix)"
    "  HL court=MR  ADF p<0.05=stationnaire"
    f"  VR<1=MR  OU theta eleve=MR  (VR q={VR_Q})[/dim]"
)


def _build_tf_table(
    symbol: str,
    tf_label: str,
    data: list[list[tuple[float, float]] | None],
) -> "Table":
    """Construit un tableau Rich pour un symbole + timeframe donné."""

    table = Table(
        title=(
            f"[bold cyan]{symbol}[/bold cyan]"
            f"  [bold white]{tf_label}[/bold white]"
        ),
        box=box.ROUNDED,
        header_style="bold bright_blue",
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )

    # ── En-têtes ──────────────────────────────────────────────────────────────
    table.add_column("Métrique", style="bold white", min_width=19, no_wrap=True)
    for lbl, _ in PERIODS:
        table.add_column(lbl, justify="right", min_width=11, no_wrap=True)
    table.add_column("Score", justify="center", min_width=16, style="bold")

    # ── Lignes métriques ──────────────────────────────────────────────────────
    for m_idx, name in enumerate(METRIC_NAMES):
        row: list[str] = [name]
        metric_scores: list[float] = []

        for col in data:
            if col is None:
                row.append("[dim]—[/dim]")
            else:
                v, sc = col[m_idx]
                metric_scores.append(sc)
                row.append(_fmt_cell(v, m_idx, sc))

        if metric_scores:
            avg = float(np.mean(metric_scores))
            row.append(_fmt_eval(avg))
        else:
            row.append("[dim]N/A[/dim]")

        table.add_row(*row)

    # ── Séparateur + ligne Score Global ───────────────────────────────────────
    table.add_section()

    global_row: list[str] = ["[bold]SCORE GLOBAL[/bold]"]
    col_avgs: list[float] = []

    for col in data:
        if col is None:
            global_row.append("[dim]—[/dim]")
        else:
            avg_col = float(np.mean([sc for _, sc in col]))
            col_avgs.append(avg_col)
            c = _color(avg_col)
            global_row.append(f"[{c}]{avg_col:.1f}/10[/{c}]")

    overall = float(np.mean(col_avgs)) if col_avgs else 0.0
    global_row.append(_fmt_eval(overall))
    table.add_row(*global_row)

    return table


def render_dashboard(symbol: str, console: "Console") -> None:
    """Affiche un tableau par timeframe pour le symbole donné."""
    any_printed = False

    for tf_label, tf in TIMEFRAMES:
        data, _metas = compute_timeframe(symbol, tf)

        # Ignorer silencieusement les TF sans aucune donnée
        if all(col is None for col in data):
            continue

        table = _build_tf_table(symbol, tf_label, data)
        console.print(table)
        any_printed = True

    if any_printed:
        console.print(_LEGEND)
    console.print()


# ─── Export HTML ──────────────────────────────────────────────────────────────

def export_html(symbol: str, path: str) -> None:
    """Sauvegarde le tableau au format HTML via Rich."""
    from rich.console import Console as RichConsole
    html_console = RichConsole(record=True, width=160)
    render_dashboard(symbol, html_console)
    html_console.save_html(path)
    print(f"  → Export HTML : {path}")


# ─── Export JSON ──────────────────────────────────────────────────────────────

import json
from datetime import datetime, timezone


def _to_json_val(v: float) -> float | None:
    """Convertit NaN/Inf en null JSON, arrondit les floats valides."""
    return None if not np.isfinite(v) else round(float(v), 6)


def build_symbol_dict(symbol: str) -> dict:
    """
    Calcule toutes les métriques pour un symbole et retourne un dict
    prêt à sérialiser en JSON.

    Structure :
      symbol, generated_at, overall_score,
      timeframes → {tf_key: {label, score, periods → {period_key: {
          label, bars, from, to, score, metrics → {metric_key: {value, score}}
      }}}}
    """
    generated_at = datetime.now(timezone.utc).isoformat()

    tf_scores: list[float] = []
    timeframes_out: dict = {}

    for tf_label, tf in TIMEFRAMES:
        data, metas = compute_timeframe(symbol, tf)

        if all(col is None for col in data):
            continue  # TF sans données (ex. XAUUSD sans 1D)

        periods_out: dict = {}
        period_scores: list[float] = []

        for p_idx, ((p_label, _days), p_key) in enumerate(zip(PERIODS, PERIOD_KEYS)):
            col  = data[p_idx]
            meta = metas[p_idx]

            if col is None:
                periods_out[p_key] = None
                continue

            metrics_out: dict = {}
            m_scores: list[float] = []

            for m_idx, (m_key, m_name) in enumerate(zip(METRIC_KEYS, METRIC_NAMES)):
                v, sc = col[m_idx]
                metrics_out[m_key] = {
                    "label": m_name,
                    "value": _to_json_val(v),
                    "score": round(sc, 2),
                }
                m_scores.append(sc)

            p_score = round(float(np.mean(m_scores)), 2) if m_scores else None
            period_scores.append(p_score or 0.0)

            periods_out[p_key] = {
                "label": p_label,
                "bars":  meta["bars"],
                "from":  meta["from"],
                "to":    meta["to"],
                "score": p_score,
                "metrics": metrics_out,
            }

        tf_score = round(float(np.mean(period_scores)), 2) if period_scores else None
        if tf_score is not None:
            tf_scores.append(tf_score)

        timeframes_out[tf] = {
            "label":   tf_label,
            "score":   tf_score,
            "periods": periods_out,
        }

    overall = round(float(np.mean(tf_scores)), 2) if tf_scores else None

    # ── Historique des scores par période (fenêtre glissante par période) ─────
    period_history_out: dict = {}
    for _, tf in TIMEFRAMES:
        if tf in timeframes_out:   # seulement les TF qui ont des données
            ph = compute_period_score_history(symbol, tf)
            if ph:
                period_history_out[tf] = ph

    return {
        "symbol":         symbol,
        "generated_at":   generated_at,
        "overall_score":  overall,
        "timeframes":     timeframes_out,
        "period_history": period_history_out,
    }


def save_json_output(symbols: list[str], output_dir: str) -> None:
    """
    Génère les fichiers JSON dans output_dir :
      index.json            — métadonnées globales + résumé par symbole
      symbols/{SYMBOL}.json — détail complet par symbole

    Le répertoire symbols/ est créé si nécessaire.
    """
    symbols_dir = os.path.join(output_dir, "symbols")
    os.makedirs(symbols_dir, exist_ok=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    index_symbols: list[dict] = []

    for sym in symbols:
        sym_dict = build_symbol_dict(sym)

        # Fichier par symbole
        sym_path = os.path.join(symbols_dir, f"{sym}.json")
        with open(sym_path, "w", encoding="utf-8") as f:
            json.dump(sym_dict, f, ensure_ascii=False, indent=2)

        # Résumé pour l'index (scores par TF, sans le détail des métriques)
        tf_scores = {
            tf: sym_dict["timeframes"][tf]["score"]
            for tf in sym_dict["timeframes"]
        }
        index_symbols.append({
            "symbol":            sym,
            "overall_score":     sym_dict["overall_score"],
            "timeframe_scores":  tf_scores,
            "file":              f"symbols/{sym}.json",
        })

        print(f"  ✓  {sym:<18}  score={sym_dict['overall_score']}  → {sym_path}")

    # Fichier index
    index = {
        "generated_at": generated_at,
        "config": {
            "vr_q":     VR_Q,
            "min_bars": 40,
        },
        "schema": {
            "timeframes": [
                {"key": tf, "label": lbl} for lbl, tf in TIMEFRAMES
            ],
            "periods": [
                {"key": k, "label": lbl} for (lbl, _), k in zip(PERIODS, PERIOD_KEYS)
            ],
            "metrics": [
                {"key": k, "label": lbl} for k, lbl in zip(METRIC_KEYS, METRIC_NAMES)
            ],
        },
        "symbols": index_symbols,
    }

    index_path = os.path.join(output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\n  index.json → {index_path}  ({len(symbols)} symboles)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def list_symbols() -> list[str]:
    """Liste tous les symboles ayant au moins un fichier CSV de données."""
    known_tf = {"_1D.csv", "_4H.csv", "_1H.csv", "_30m.csv", "_15m.csv", "_5m.csv", "_1m.csv"}
    symbols: set[str] = set()
    for f in os.listdir(HISTORY_DIR):
        for tf in known_tf:
            if f.endswith(tf):
                symbols.add(f[: -len(tf)])
                break
    return sorted(symbols)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mean Reversion Dashboard — analyse statistique multi-timeframe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "symbols",
        nargs="*",
        metavar="SYMBOL",
        help="Symbole(s) à analyser (ex: EURUSD GBPUSD). Défaut : tous.",
    )
    ap.add_argument(
        "--list", "-l",
        action="store_true",
        help="Lister les symboles disponibles et quitter.",
    )
    ap.add_argument(
        "--export", "-e",
        metavar="FICHIER.html",
        help="Exporter le tableau en HTML (valide pour un seul symbole à la fois).",
    )
    ap.add_argument(
        "--json-dir", "-j",
        metavar="RÉPERTOIRE",
        help=(
            "Répertoire de sortie JSON. Génère index.json + symbols/{SYM}.json. "
            "Peut être combiné avec --quiet pour désactiver l'affichage console."
        ),
    )
    ap.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Supprime l'affichage console (utile avec --json-dir en production).",
    )
    args = ap.parse_args()

    if not os.path.isdir(HISTORY_DIR):
        print(f"[ERREUR] Répertoire introuvable : {HISTORY_DIR}")
        sys.exit(1)

    all_symbols = list_symbols()

    if args.list:
        print(f"Symboles disponibles ({len(all_symbols)}) :")
        col_w = max(len(s) for s in all_symbols) + 2
        per_row = max(1, 80 // col_w)
        for i, sym in enumerate(all_symbols):
            end = "\n" if (i + 1) % per_row == 0 else ""
            print(f"  {sym:<{col_w}}", end=end)
        print()
        return

    symbols = args.symbols if args.symbols else all_symbols

    # Validation
    unknown = [s for s in symbols if s not in all_symbols]
    if unknown:
        print(f"[WARN] Symboles introuvables : {', '.join(unknown)}")
        symbols = [s for s in symbols if s not in unknown]
        if not symbols:
            sys.exit(1)

    if not HAS_RICH:
        print("Installez rich : pip install rich statsmodels")
        sys.exit(1)

    # ── Export JSON ───────────────────────────────────────────────────────────
    if args.json_dir:
        print(f"Export JSON → {args.json_dir}  ({len(symbols)} symboles)\n")
        save_json_output(symbols, args.json_dir)

    # ── Affichage console ─────────────────────────────────────────────────────
    if args.quiet:
        return

    console = Console(force_terminal=True, legacy_windows=False)

    for idx, sym in enumerate(symbols):
        console.rule(f"[bold blue]{sym}[/bold blue]  [dim]({idx+1}/{len(symbols)})[/dim]")

        if args.export:
            export_path = args.export if len(symbols) == 1 else f"{sym}_{args.export}"
            export_html(sym, export_path)

        render_dashboard(sym, console)


if __name__ == "__main__":
    main()
