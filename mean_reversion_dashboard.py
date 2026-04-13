#!/usr/bin/env python3
"""
Mean Reversion Dashboard
========================
Analyse statistique multi-timeframe pour le trading mean reversion.

Métriques calculées :
  - Hurst Exponent     (R/S analysis sur log-prix)
  - Half-Life          (régression OU, converti en jours)
  - ADF p-value        (test racine unitaire sur log-prix)
  - Variance Ratio     (Lo-MacKinlay, q=10, sur log-returns)
  - OU θ (speed)       (vitesse de retour à la moyenne, annualisée)

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

# Définition des colonnes : (label_affichage, timeframe, jours_lookback)
COLUMNS: list[tuple[str, str, int]] = [
    ("Journ. 3 ans", "1D", 365 * 3),
    ("Journ. 2 ans", "1D", 365 * 2),
    ("Journ. 1 an",  "1D", 365),
    ("H4  1 an",     "4H", 365),
    ("H4  6 mois",   "4H", 183),
    ("H1  1 an",     "1H", 365),
    ("H1  6 mois",   "1H", 183),
]

METRIC_NAMES: list[str] = [
    "Hurst Exponent",
    "Half-Life (jours)",
    "ADF p-value",
    "Variance Ratio",
    "OU theta (speed)",
]

# q pour le Variance Ratio
VR_Q = 10

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


def adf_pvalue(prices: np.ndarray) -> float:
    """
    ADF p-value sur les log-prix.
    p < 0.05 : série stationnaire (mean-reverting)
    """
    lp = np.log(prices)
    if len(lp) < 20:
        return np.nan
    try:
        return float(adfuller(lp, autolag="AIC")[1])
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

    rq = np.array([r[i : i + q].sum() for i in range(n - q + 1)])
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


def get_series(symbol: str, tf: str, days: int) -> np.ndarray | None:
    """Charge et filtre les prix de clôture sur les `days` derniers jours."""
    df = _load_df(symbol, tf)
    if df is None:
        return None
    ref = df["Datetime"].iloc[-1]
    cutoff = ref - pd.Timedelta(days=days)
    arr = df.loc[df["Datetime"] >= cutoff, "Close"].dropna().values
    return arr if len(arr) >= 40 else None


# ─── Calcul complet par symbole ────────────────────────────────────────────────

def compute_symbol(symbol: str) -> list[list[tuple[float, float]] | None]:
    """
    Retourne data[col_idx][metric_idx] = (valeur, score)
    ou None si les données sont absentes pour cette colonne.
    """
    data: list[list[tuple[float, float]] | None] = []

    for _, tf, days in COLUMNS:
        series = get_series(symbol, tf, days)
        if series is None:
            data.append(None)
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

    return data


# ─── Rendu Rich ────────────────────────────────────────────────────────────────

def render_dashboard(symbol: str, console: "Console") -> None:
    data = compute_symbol(symbol)

    table = Table(
        title=f"[bold cyan]Mean Reversion Dashboard — [white]{symbol}[/white][/bold cyan]",
        box=box.ROUNDED,
        header_style="bold bright_blue",
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )

    # En-têtes
    table.add_column("Métrique", style="bold white", min_width=19, no_wrap=True)
    for lbl, _, _ in COLUMNS:
        table.add_column(lbl, justify="right", min_width=12, no_wrap=True)
    table.add_column("Éval. Générale", justify="center", min_width=16, style="bold")

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

    console.print(table)
    console.print(
        "[dim]  Hurst<0.5=prix MR (variance scaling log-prix)"
        "  HL court=MR  ADF p<0.05=stationnaire"
        f"  VR<1=MR  OU theta eleve=MR  (VR q={VR_Q})[/dim]\n"
    )


# ─── Export HTML ──────────────────────────────────────────────────────────────

def export_html(symbol: str, path: str) -> None:
    """Sauvegarde le tableau au format HTML via Rich."""
    from rich.console import Console as RichConsole
    html_console = RichConsole(record=True, width=160)
    render_dashboard(symbol, html_console)
    html_console.save_html(path)
    print(f"  → Export HTML : {path}")


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

    console = Console(force_terminal=True, legacy_windows=False)

    for idx, sym in enumerate(symbols):
        console.rule(f"[bold blue]{sym}[/bold blue]  [dim]({idx+1}/{len(symbols)})[/dim]")

        if args.export:
            export_path = args.export if len(symbols) == 1 else f"{sym}_{args.export}"
            export_html(sym, export_path)

        render_dashboard(sym, console)


if __name__ == "__main__":
    main()
