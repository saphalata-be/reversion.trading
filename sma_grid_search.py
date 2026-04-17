"""
SMA Grid Search
---------------
Lance l'analyse SMA sur toutes les combinaisons de timeframe × période,
regroupe les résultats par symbole et sauvegarde un rapport CSV + texte.

Usage:
    python sma_grid_search.py [--dir DIR] [--out OUT]

Timeframes : 1H, 4H, 1D
Périodes   : 20, 50, 100, 150, 200
"""

import argparse
import json
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
from strategy_percentile import backtest_strategy

TIMEFRAMES = ["1H", "4H", "1D"]
PERIODS    = [20, 50, 100, 150, 200]

METRICS = [
    "bars", "avg", "positive", "negative", "abs_avg", "pct_up",
    "fee_pct", "net_abs", "crossings",
    "bt40", "bt30", "bt20", "bt10",
    "bt40_long", "bt30_long", "bt20_long", "bt10_long",
    "bt40_short", "bt30_short", "bt20_short", "bt10_short",
]

# ─── Colonnes affichées dans le rapport texte (plus compact) ─────────────────
TEXT_COLS = ["timeframe", "period", "bars", "bt40", "bt30", "bt20", "bt10",
             "crossings", "abs_avg", "pct_up"]

COL_W = {
    "timeframe": 10, "period": 8, "bars": 7,
    "bt40": 9, "bt30": 9, "bt20": 9, "bt10": 9,
    "crossings": 11, "abs_avg": 11, "pct_up": 9,
}

HEADER_LABELS = {
    "timeframe": "TF", "period": "Période", "bars": "Bars",
    "bt40": "BT 40%", "bt30": "BT 30%", "bt20": "BT 20%", "bt10": "BT 10%",
    "crossings": "Croisements", "abs_avg": "Moy. abs.", "pct_up": "% haussier",
}


def fmt_val(col: str, val) -> str:
    if col in ("bt40", "bt30", "bt20", "bt10", "avg", "positive", "negative", "net_abs"):
        return f"{val:+.2f}%"
    if col in ("abs_avg",):
        return f"{val:.4f}%"
    if col in ("pct_up",):
        return f"{val:.1f}%"
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
            try:
                df_sym = pd.read_csv(filepath, parse_dates=["Datetime"])
                bt = backtest_strategy(df_sym, period, fee)
                stats.update(bt)
            except Exception:
                stats.update({
                    "bt40": 0.0, "bt30": 0.0, "bt20": 0.0, "bt10": 0.0,
                    "bt40_long": 0.0, "bt30_long": 0.0, "bt20_long": 0.0, "bt10_long": 0.0,
                    "bt40_short": 0.0, "bt30_short": 0.0, "bt20_short": 0.0, "bt10_short": 0.0,
                })
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


def write_json_output(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde les résultats en JSON pour l'interface web (stratégie percentile)."""
    rows = []
    for _, row in df.iterrows():
        entry = {
            "symbol":    str(row["symbol"]),
            "timeframe": str(row["timeframe"]),
            "period":    int(row["period"]),
            "bars":      int(row["bars"]),
            "crossings": int(row.get("crossings", 0)),
            "abs_avg":   round(float(row["abs_avg"]), 6),
            "pct_up":    round(float(row["pct_up"]), 1),
        }
        for col in ("bt40", "bt30", "bt20", "bt10",
                    "bt40_long", "bt30_long", "bt20_long", "bt10_long",
                    "bt40_short", "bt30_short", "bt20_short", "bt10_short"):
            entry[col] = round(float(row.get(col, 0.0)), 2)
        rows.append(entry)
    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timeframes":   TIMEFRAMES,
        "periods":      PERIODS,
        "rows":         rows,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"JSON sauvegardé : {path}")


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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
    os.makedirs(output_dir, exist_ok=True)
    write_json_output(df, os.path.join(output_dir, f"strategy_{timestamp}.json"))

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
