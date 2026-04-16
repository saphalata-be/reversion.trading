"""
SMA Slope Analysis
------------------
Lit les fichiers CSV du répertoire TickStory et calcule la pente moyenne
(en %) de la SMA pour chaque symbole.

Usage:
    python sma_slope_analysis.py [--dir DIR] [--timeframe TF] [--period N] [--sort COLUMN]

Exemples:
    python sma_slope_analysis.py --timeframe 1D --period 20
    python sma_slope_analysis.py --timeframe 4H --period 50 --sort avg
    python sma_slope_analysis.py --timeframe 1H --period 14 --sort positive
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


DEFAULT_DIR = r"E:\Forex\History\TickStory"
DEFAULT_TIMEFRAME = "1D"
DEFAULT_PERIOD = 20


def compute_sma_slopes(close: pd.Series, period: int) -> pd.Series:
    """Retourne la série des pentes de la SMA en pourcentage."""
    sma = close.rolling(window=period).mean()
    # Pente en % : variation relative entre deux valeurs consécutives
    slope_pct = sma.pct_change() * 100
    return slope_pct.dropna()


def analyse_symbol(filepath: str, period: int) -> dict | None:
    try:
        df = pd.read_csv(filepath, parse_dates=["Datetime"])
        if len(df) < period + 2:
            return None

        slopes = compute_sma_slopes(df["Close"], period)
        if slopes.empty:
            return None

        pos = slopes[slopes > 0]
        neg = slopes[slopes < 0]

        return {
            "bars": len(df),
            "avg":      round(slopes.mean(), 6),
            "positive": round(pos.mean(),    6) if not pos.empty else 0.0,
            "negative": round(neg.mean(),    6) if not neg.empty else 0.0,
            "pct_up":  round(len(pos) / len(slopes) * 100, 1),
            "abs_avg": round(slopes.abs().mean(), 6),
        }
    except Exception as exc:
        print(f"  [ERREUR] {os.path.basename(filepath)}: {exc}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Analyse de la pente de la SMA par symbole")
    parser.add_argument("--dir",       default=DEFAULT_DIR,       help="Répertoire des fichiers CSV")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME, help="Timeframe (1D, 4H, 1H, 30m, 15m, 5m, 1m)")
    parser.add_argument("--period",    default=DEFAULT_PERIOD, type=int, help="Longueur de la SMA")
    parser.add_argument("--sort",      default="symbol",
                        choices=["symbol", "avg", "positive", "negative", "abs_avg", "pct_up"],
                        help="Colonne de tri")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Répertoire introuvable : {args.dir}", file=sys.stderr)
        sys.exit(1)

    pattern = f"_{args.timeframe}.csv"
    files = sorted(f for f in os.listdir(args.dir) if f.endswith(pattern))

    if not files:
        print(f"Aucun fichier trouvé pour le timeframe '{args.timeframe}' dans {args.dir}")
        sys.exit(1)

    print(f"\nAnalyse SMA({args.period}) — Timeframe : {args.timeframe}")
    print(f"Répertoire : {args.dir}")
    print(f"Symboles trouvés : {len(files)}\n")

    results = []
    for filename in files:
        symbol = filename.replace(pattern, "")
        filepath = os.path.join(args.dir, filename)
        stats = analyse_symbol(filepath, args.period)
        if stats:
            results.append({"symbol": symbol, **stats})

    if not results:
        print("Aucun résultat calculable.")
        sys.exit(0)

    df_out = pd.DataFrame(results)

    # Tri
    ascending = args.sort == "symbol"
    if args.sort != "symbol":
        # Pour negative, trier du plus négatif au moins négatif
        ascending = args.sort not in ("negative",)
        df_out = df_out.sort_values(args.sort, ascending=ascending)
    else:
        df_out = df_out.sort_values("symbol")

    df_out = df_out.reset_index(drop=True)

    # ── Affichage ────────────────────────────────────────────────────────────
    col_w = {"symbol": 18, "bars": 7, "avg": 12, "positive": 12, "negative": 12, "abs_avg": 12, "pct_up": 9}
    header = (
        f"{'Symbole':<{col_w['symbol']}}"
        f"{'Bars':>{col_w['bars']}}"
        f"{'Moy. générale':>{col_w['avg']}}"
        f"{'Moy. positive':>{col_w['positive']}}"
        f"{'Moy. négative':>{col_w['negative']}}"
        f"{'Moy. absolue':>{col_w['abs_avg']}}"
        f"{'% haussier':>{col_w['pct_up']}}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    for _, row in df_out.iterrows():
        avg_str = f"{row['avg']:+.6f}%"
        pos_str = f"{row['positive']:+.6f}%"
        neg_str = f"{row['negative']:+.6f}%"
        abs_str = f"{row['abs_avg']:.6f}%"
        up_str  = f"{row['pct_up']:.1f}%"
        print(
            f"{row['symbol']:<{col_w['symbol']}}"
            f"{int(row['bars']):>{col_w['bars']}}"
            f"{avg_str:>{col_w['avg']}}"
            f"{pos_str:>{col_w['positive']}}"
            f"{neg_str:>{col_w['negative']}}"
            f"{abs_str:>{col_w['abs_avg']}}"
            f"{up_str:>{col_w['pct_up']}}"
        )

    print(sep)

    # Résumé global
    avg_all = df_out["avg"].mean()
    avg_pos = df_out["positive"].mean()
    avg_neg = df_out["negative"].mean()
    avg_abs = df_out["abs_avg"].mean()
    avg_up  = df_out["pct_up"].mean()
    print(
        f"{'MOYENNE GLOBALE':<{col_w['symbol']}}"
        f"{'':>{col_w['bars']}}"
        f"{avg_all:>+{col_w['avg'] - 1}.6f}%"
        f"{avg_pos:>+{col_w['positive'] - 1}.6f}%"
        f"{avg_neg:>+{col_w['negative'] - 1}.6f}%"
        f"{avg_abs:>{col_w['abs_avg'] - 1}.6f}%"
        f"{avg_up:>{col_w['pct_up'] - 1}.1f}%"
    )
    print()


if __name__ == "__main__":
    main()
