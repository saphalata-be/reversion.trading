"""
SMA Grid Search
---------------
Lance l'analyse SMA sur toutes les combinaisons de timeframe × période,
regroupe les résultats par symbole et sauvegarde un rapport CSV + texte.

Usage:
    python sma_grid_search.py [--dir DIR] [--out OUT]

Timeframes : 1H, 4H, 1D
Périodes   : 25, 50, 75, 100, 125, 150, 200
"""

import argparse
import os
import sys
from datetime import datetime
from itertools import product

import pandas as pd

# Import des fonctions internes du script d'analyse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sma_slope_analysis import (
    DEFAULT_DIR,
    CONFIG_FILE,
    load_symbol_config,
    save_symbol_config,
    get_symbol_fee,
    analyse_symbol,
    DEFAULT_FEE_PCT,
)

TIMEFRAMES = ["1H", "4H", "1D"]
PERIODS    = [25, 50, 75, 100, 125, 150, 200]

METRICS = [
    "bars", "avg", "positive", "negative", "abs_avg", "pct_up",
    "fee_pct", "net_abs", "crossings",
    "p40", "p30", "p20", "p10",
    "bt40", "bt30", "bt20", "bt10",
]

# ─── Colonnes affichées dans le rapport texte (plus compact) ─────────────────
TEXT_COLS = ["timeframe", "period", "bars", "bt40", "bt30", "bt20", "bt10",
             "crossings", "p40", "p30", "p20", "p10", "abs_avg", "pct_up"]

COL_W = {
    "timeframe": 10, "period": 8, "bars": 7,
    "bt40": 9, "bt30": 9, "bt20": 9, "bt10": 9,
    "crossings": 11, "p40": 9, "p30": 9, "p20": 9, "p10": 9,
    "abs_avg": 11, "pct_up": 9,
}

HEADER_LABELS = {
    "timeframe": "TF", "period": "Période", "bars": "Bars",
    "bt40": "BT 40%", "bt30": "BT 30%", "bt20": "BT 20%", "bt10": "BT 10%",
    "crossings": "Croisements", "p40": "P60", "p30": "P70", "p20": "P80", "p10": "P90",
    "abs_avg": "Moy. abs.", "pct_up": "% haussier",
}


def fmt_val(col: str, val) -> str:
    if col in ("bt40", "bt30", "bt20", "bt10", "avg", "positive", "negative", "net_abs"):
        return f"{val:+.2f}%"
    if col in ("abs_avg",):
        return f"{val:.4f}%"
    if col in ("pct_up",):
        return f"{val:.1f}%"
    if col in ("p40", "p30", "p20", "p10"):
        return f"{val:.4f}%"
    if col == "bars":
        return str(int(val))
    if col == "crossings":
        return str(int(val))
    if col == "period":
        return str(int(val))
    return str(val)


def run_grid(data_dir: str) -> pd.DataFrame:
    """Parcourt toutes les combinaisons TF × période et retourne un DataFrame."""
    symbol_config = load_symbol_config(CONFIG_FILE)
    new_symbols_before = set(symbol_config.keys())

    rows = []
    total = len(TIMEFRAMES) * len(PERIODS)
    done = 0

    for tf, period in product(TIMEFRAMES, PERIODS):
        done += 1
        pattern = f"_{tf}.csv"
        files = sorted(f for f in os.listdir(data_dir) if f.endswith(pattern))

        if not files:
            print(f"  [{done}/{total}] TF={tf} période={period:3d} — aucun fichier trouvé, ignoré.")
            continue

        print(f"  [{done}/{total}] TF={tf:3s}  période={period:3d}  ({len(files)} symboles)…")

        for filename in files:
            symbol = filename.replace(pattern, "")
            filepath = os.path.join(data_dir, filename)
            fee = get_symbol_fee(symbol_config, symbol)
            stats = analyse_symbol(filepath, period, fee)
            if stats is None:
                continue
            stats["fee_pct"] = fee
            stats["net_abs"] = round(stats["abs_avg"] - fee, 6)
            rows.append({
                "symbol":    symbol,
                "timeframe": tf,
                "period":    period,
                **stats,
            })

    save_symbol_config(symbol_config, CONFIG_FILE)
    new_symbols = set(symbol_config.keys()) - new_symbols_before
    if new_symbols:
        print(f"\n  → {len(new_symbols)} nouveau(x) symbole(s) ajouté(s) dans {CONFIG_FILE}.")

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nCSV sauvegardé : {path}")


def write_text_report(df: pd.DataFrame, path: str) -> None:
    """Rapport texte : une section par symbole, toutes les combinaisons TF × période."""
    symbols = sorted(df["symbol"].unique())

    # Largeur de la table par symbole
    header_line = "".join(
        f"{HEADER_LABELS.get(c, c):>{COL_W[c]}}" for c in TEXT_COLS
    )
    sep = "─" * len(header_line)

    lines = [
        "SMA Grid Search — Rapport",
        f"Généré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Timeframes : {', '.join(TIMEFRAMES)}",
        f"Périodes   : {', '.join(map(str, PERIODS))}",
        f"Symboles   : {len(symbols)}",
        "",
    ]

    for symbol in symbols:
        sub = df[df["symbol"] == symbol].copy()
        # Tri par BT top10% décroissant pour identifier rapidement la meilleure config
        sub = sub.sort_values("bt10", ascending=False)

        lines.append(f"{'═' * len(header_line)}")
        lines.append(f"  {symbol}")
        lines.append(header_line)
        lines.append(sep)

        for _, row in sub.iterrows():
            line = "".join(
                f"{fmt_val(c, row[c]):>{COL_W[c]}}" for c in TEXT_COLS
            )
            lines.append(line)

        # Meilleure config par BT top10%
        best = sub.iloc[0]
        lines.append(sep)
        lines.append(
            f"  → Meilleure config (BT top10%) : TF={best['timeframe']}  "
            f"période={int(best['period'])}  "
            f"BT10={best['bt10']:+.2f}%"
        )
        lines.append("")

    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Rapport texte sauvegardé : {path}")


def main():
    parser = argparse.ArgumentParser(description="Grid search SMA sur TF × période")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="Répertoire des fichiers CSV")
    parser.add_argument("--out", default=None,
                        help="Préfixe du fichier de sortie (sans extension)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Répertoire introuvable : {args.dir}", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.out if args.out else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"sma_grid_{timestamp}"
    )

    print(f"\nSMA Grid Search")
    print(f"Répertoire : {args.dir}")
    print(f"Combinaisons : {len(TIMEFRAMES)} TF × {len(PERIODS)} périodes = "
          f"{len(TIMEFRAMES) * len(PERIODS)} runs\n")

    df = run_grid(args.dir)

    if df.empty:
        print("Aucun résultat calculable.")
        sys.exit(0)

    write_csv(df, base + ".csv")
    write_text_report(df, base + ".txt")

    # Résumé global : top configs par BT10 moyen sur tous les symboles
    print("\n── Top 10 configurations (BT top10% moyen toutes paires) ──────────────")
    agg = (
        df.groupby(["timeframe", "period"])[["bt10", "bt20", "bt30", "bt40"]]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("bt10", ascending=False)
        .head(10)
    )
    print(agg.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
