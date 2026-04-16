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
import json
import os
import sys

import numpy as np
import pandas as pd


DEFAULT_DIR = r"E:\Forex\History\TickStory"
DEFAULT_TIMEFRAME = "1D"
DEFAULT_PERIOD = 20
DEFAULT_FEE_PCT = 0.01  # frais par défaut : 0.01 % par transaction
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "symbols_config.json")


def load_symbol_config(config_path: str) -> dict:
    """Charge la configuration des symboles depuis le fichier JSON."""
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_symbol_config(config: dict, config_path: str) -> None:
    """Sauvegarde la configuration des symboles dans le fichier JSON."""
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)


def get_symbol_fee(config: dict, symbol: str) -> float:
    """Retourne les frais du symbole ; l'ajoute avec la valeur par défaut si absent."""
    if symbol not in config:
        config[symbol] = {"fee_pct": DEFAULT_FEE_PCT}
    elif "fee_pct" not in config[symbol]:
        config[symbol]["fee_pct"] = DEFAULT_FEE_PCT
    return config[symbol]["fee_pct"]


def compute_sma_slopes(close: pd.Series, period: int) -> pd.Series:
    """Retourne la série des pentes de la SMA en pourcentage."""
    sma = close.rolling(window=period).mean()
    # Pente en % : variation relative entre deux valeurs consécutives
    slope_pct = sma.pct_change() * 100
    return slope_pct.dropna()


def compute_sma_crossings(close: pd.Series, period: int, min_bars: int = 3) -> int:
    """Compte le nombre de croisements de la SMA.

    Un croisement n'est comptabilisé que si le prix est resté au moins
    `min_bars` bougies consécutives du même côté avant de changer de camp.
    """
    sma = close.rolling(window=period).mean()
    valid_mask = sma.notna()
    close_v = close[valid_mask].values
    sma_v = sma[valid_mask].values

    # +1 au-dessus, -1 en-dessous, 0 sur la SMA (ignoré)
    position = np.sign(close_v - sma_v).astype(int)
    position = position[position != 0]

    if len(position) == 0:
        return 0

    crossings = 0
    run_length = 1
    for i in range(1, len(position)):
        if position[i] == position[i - 1]:
            run_length += 1
        else:
            if run_length >= min_bars:
                crossings += 1
            run_length = 1
    return crossings


def compute_crossing_excursions(df: pd.DataFrame, period: int, min_bars: int = 3) -> list:
    """Retourne la liste des excursions max close-based (%) de chaque run valide.

    Pour chaque run d'au moins `min_bars` bougies d'un même côté de la SMA,
    on enregistre l'excursion maximale :
        abs(close - sma) / sma * 100
    """
    close_arr = df["Close"].astype(float).values
    sma_arr = pd.Series(close_arr).rolling(window=period).mean().values

    valid_mask = ~np.isnan(sma_arr)
    if not valid_mask.any():
        return []

    first_valid = int(np.argmax(valid_mask))

    raw_sign = np.zeros(len(df), dtype=int)
    diff = close_arr[valid_mask] - sma_arr[valid_mask]
    raw_sign[valid_mask] = np.sign(diff).astype(int)

    excursions = []

    run_sign = 0
    run_start = None
    run_len = 0

    for idx in range(first_valid, len(df)):
        s = raw_sign[idx]

        if s == 0:
            continue

        if run_sign == 0:
            run_sign = s
            run_start = idx
            run_len = 1
            continue

        if s == run_sign:
            run_len += 1
            continue

        # Fin du run précédent
        if run_start is not None and run_len >= min_bars:
            sma_seg = sma_arr[run_start:idx]
            close_seg = close_arr[run_start:idx]
            v = ~np.isnan(sma_seg) & (sma_seg > 0)
            if v.any():
                exc_vals = np.abs(close_seg[v] - sma_seg[v]) / sma_seg[v] * 100
                excursions.append(float(exc_vals.max()))

        # Nouveau run
        run_sign = s
        run_start = idx
        run_len = 1

    # Dernier run
    if run_start is not None and run_len >= min_bars:
        sma_seg = sma_arr[run_start:len(df)]
        close_seg = close_arr[run_start:len(df)]
        v = ~np.isnan(sma_seg) & (sma_seg > 0)
        if v.any():
            exc_vals = np.abs(close_seg[v] - sma_seg[v]) / sma_seg[v] * 100
            excursions.append(float(exc_vals.max()))

    return excursions


def backtest_strategy(
    df: pd.DataFrame,
    period: int,
    fee_pct: float,
    min_bars: int = 3,
    warmup_years: int = 2,
    min_excursions_history: int = 10,
) -> dict:
    """Backteste une stratégie de mean reversion vers la SMA.

    Logique :
      - Warmup de `warmup_years` ans : accumulation des excursions historiques,
        mais aucun trade.
      - Un run correspond à une suite de bougies du même côté de la SMA.
      - Après `min_bars` bougies confirmées dans un run, si l'excursion close-based
        dépasse un seuil historique, entrée à l'open de la bougie suivante.
      - Sortie lorsqu'un crossing est détecté sur la clôture d'une bougie :
        exécution à l'open de la bougie suivante.
      - Une seule entrée par run et par niveau.
      - Les positions encore ouvertes à la fin sont clôturées au dernier close.
    """
    n = len(df)
    if n < period + 2:
        return {"bt40": 0.0, "bt30": 0.0, "bt20": 0.0, "bt10": 0.0}

    required_cols = {"Datetime", "Open", "Close"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour le backtest : {sorted(missing)}")

    close_arr = df["Close"].astype(float).values
    open_arr = df["Open"].astype(float).values
    dt_arr = pd.to_datetime(df["Datetime"]).values

    sma_arr = pd.Series(close_arr).rolling(window=period).mean().values

    valid_mask = ~np.isnan(sma_arr)
    if not valid_mask.any():
        return {"bt40": 0.0, "bt30": 0.0, "bt20": 0.0, "bt10": 0.0}

    first_valid = int(np.argmax(valid_mask))

    # Signe : +1 au-dessus, -1 en-dessous, 0 sur la SMA / avant SMA disponible
    raw_sign = np.zeros(n, dtype=int)
    diff = close_arr[valid_mask] - sma_arr[valid_mask]
    raw_sign[valid_mask] = np.sign(diff).astype(int)

    # Warmup en années calendaires
    warmup_dt = pd.Timestamp(dt_arr[first_valid]) + pd.DateOffset(years=warmup_years)
    warmup_end = n
    for idx in range(first_valid, n):
        if pd.Timestamp(dt_arr[idx]) >= warmup_dt:
            warmup_end = idx
            break

    # Historique des excursions max par run terminé
    excursions_history: list[float] = []

    # PnL cumulé par niveau : bt40, bt30, bt20, bt10
    pnl = [0.0, 0.0, 0.0, 0.0]
    pct_ranks = [60, 70, 80, 90]
    thresholds: list[float | None] = [None, None, None, None]

    # État des trades
    in_trade = [False, False, False, False]
    trade_entry = [0.0, 0.0, 0.0, 0.0]
    trade_dir = [0, 0, 0, 0]  # +1 si run au-dessus SMA => short ; -1 => long

    # Trouver le premier signe non nul pour démarrer les runs proprement
    run_sign = 0
    run_start = None
    run_len = 0
    triggered = [False, False, False, False]

    for idx in range(first_valid, n):
        s = raw_sign[idx]

        # Ignore les bougies sur la SMA exacte : elles ne changent pas le run
        if s == 0:
            continue

        # Initialisation du tout premier run non nul
        if run_sign == 0:
            run_sign = s
            run_start = idx
            run_len = 1
            triggered = [False, False, False, False]
            continue

        # Cas 1 : on reste dans le même run
        if s == run_sign:
            run_len += 1

            # Vérification des entrées à partir de min_bars bougies confirmées
            if idx >= warmup_end and run_len >= min_bars:
                sma_i = sma_arr[idx]
                if not np.isnan(sma_i) and sma_i > 0:
                    exc = abs(close_arr[idx] - sma_i) / sma_i * 100
                    entry_idx = idx + 1

                    if entry_idx < n and not np.isnan(open_arr[entry_idx]):
                        for k in range(4):
                            if (
                                thresholds[k] is not None
                                and exc > thresholds[k]
                                and not triggered[k]
                                and not in_trade[k]
                            ):
                                triggered[k] = True
                                in_trade[k] = True
                                trade_entry[k] = open_arr[entry_idx]
                                trade_dir[k] = run_sign

            continue

        # Cas 2 : crossing détecté à la clôture de idx
        # Le run précédent se termine sur [run_start, idx)
        if run_start is not None and run_len >= min_bars:
            sma_seg = sma_arr[run_start:idx]
            close_seg = close_arr[run_start:idx]
            v = ~np.isnan(sma_seg) & (sma_seg > 0)
            if v.any():
                exc_vals = np.abs(close_seg[v] - sma_seg[v]) / sma_seg[v] * 100
                excursions_history.append(float(exc_vals.max()))

                if len(excursions_history) >= min_excursions_history:
                    arr_e = np.array(excursions_history, dtype=float)
                    thresholds = [float(np.percentile(arr_e, r)) for r in pct_ranks]

        # Sortie des trades :
        # le crossing est connu sur la clôture de idx,
        # donc sortie réaliste à l'open de idx + 1
        exit_idx = idx + 1
        if exit_idx < n and not np.isnan(open_arr[exit_idx]):
            exit_price = open_arr[exit_idx]
            for k in range(4):
                if in_trade[k]:
                    entry = trade_entry[k]
                    td = trade_dir[k]

                    if td > 0:
                        # run au-dessus SMA => trade short
                        raw = (entry - exit_price) / entry * 100
                    else:
                        # run en-dessous SMA => trade long
                        raw = (exit_price - entry) / entry * 100

                    pnl[k] += raw - 2 * fee_pct
                    in_trade[k] = False
        else:
            for k in range(4):
                in_trade[k] = False

        # Démarrage du nouveau run
        run_sign = s
        run_start = idx
        run_len = 1
        triggered = [False, False, False, False]

        # Pas d'entrée immédiate ici :
        # il faut attendre que le nouveau run accumule au moins min_bars bougies

    # Clôture de fin de série au dernier close disponible
    final_close = close_arr[-1]
    if not np.isnan(final_close):
        for k in range(4):
            if in_trade[k]:
                entry = trade_entry[k]
                td = trade_dir[k]

                if td > 0:
                    raw = (entry - final_close) / entry * 100
                else:
                    raw = (final_close - entry) / entry * 100

                pnl[k] += raw - 2 * fee_pct
                in_trade[k] = False

    return {
        "bt40": round(pnl[0], 2),
        "bt30": round(pnl[1], 2),
        "bt20": round(pnl[2], 2),
        "bt10": round(pnl[3], 2),
    }


def analyse_symbol(filepath: str, period: int, fee_pct: float = DEFAULT_FEE_PCT) -> dict | None:
    try:
        df = pd.read_csv(filepath, parse_dates=["Datetime"])
        if len(df) < period + 2:
            return None

        slopes = compute_sma_slopes(df["Close"], period)
        if slopes.empty:
            return None

        pos = slopes[slopes > 0]
        neg = slopes[slopes < 0]

        crossings = compute_sma_crossings(df["Close"], period)

        excursions = compute_crossing_excursions(df, period)
        if excursions:
            arr = np.array(excursions)
            p40 = round(float(np.percentile(arr, 60)), 4)
            p30 = round(float(np.percentile(arr, 70)), 4)
            p20 = round(float(np.percentile(arr, 80)), 4)
            p10 = round(float(np.percentile(arr, 90)), 4)
        else:
            p40 = p30 = p20 = p10 = 0.0

        bt = backtest_strategy(df, period, fee_pct)

        return {
            "bars":      len(df),
            "avg":       round(slopes.mean(), 6),
            "positive":  round(pos.mean(),    6) if not pos.empty else 0.0,
            "negative":  round(neg.mean(),    6) if not neg.empty else 0.0,
            "pct_up":    round(len(pos) / len(slopes) * 100, 1),
            "abs_avg":   round(slopes.abs().mean(), 6),
            "crossings": crossings,
            "p40": p40, "p30": p30, "p20": p20, "p10": p10,
            "bt40": bt["bt40"], "bt30": bt["bt30"],
            "bt20": bt["bt20"], "bt10": bt["bt10"],
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
                        choices=["symbol", "avg", "positive", "negative", "abs_avg", "pct_up", "crossings",
                                 "p40", "p30", "p20", "p10",
                                 "bt40", "bt30", "bt20", "bt10"],
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

    # Chargement de la configuration des symboles (frais, etc.)
    symbol_config = load_symbol_config(CONFIG_FILE)
    new_symbols_before = set(symbol_config.keys())

    results = []
    for filename in files:
        symbol = filename.replace(pattern, "")
        filepath = os.path.join(args.dir, filename)
        fee = get_symbol_fee(symbol_config, symbol)
        stats = analyse_symbol(filepath, args.period, fee)
        if stats:
            stats["fee_pct"] = fee
            stats["net_abs"] = round(stats["abs_avg"] - fee, 6)
            results.append({"symbol": symbol, **stats})

    # Sauvegarde la config (nouveaux symboles ajoutés avec valeurs par défaut)
    save_symbol_config(symbol_config, CONFIG_FILE)
    new_symbols_after = set(symbol_config.keys()) - new_symbols_before
    if new_symbols_after:
        print(f"  → {len(new_symbols_after)} nouveau(x) symbole(s) ajouté(s) dans {CONFIG_FILE} "
              f"avec frais par défaut ({DEFAULT_FEE_PCT}%).\n")

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
    col_w = {"symbol": 18, "bars": 7, "avg": 12, "positive": 12, "negative": 12,
             "abs_avg": 12, "pct_up": 9, "fee_pct": 9, "net_abs": 12, "crossings": 11,
             "p40": 10, "p30": 10, "p20": 10, "p10": 10,
             "bt40": 12, "bt30": 12, "bt20": 12, "bt10": 12}
    header = (
        f"{'Symbole':<{col_w['symbol']}}"
        f"{'Bars':>{col_w['bars']}}"
        f"{'Moy. générale':>{col_w['avg']}}"
        f"{'Moy. positive':>{col_w['positive']}}"
        f"{'Moy. négative':>{col_w['negative']}}"
        f"{'Moy. absolue':>{col_w['abs_avg']}}"
        f"{'% haussier':>{col_w['pct_up']}}"
        f"{'Frais':>{col_w['fee_pct']}}"
        f"{'Net abs.':>{col_w['net_abs']}}"
        f"{'Croisements':>{col_w['crossings']}}"
        f"{'Top 40%':>{col_w['p40']}}"
        f"{'Top 30%':>{col_w['p30']}}"
        f"{'Top 20%':>{col_w['p20']}}"
        f"{'Top 10%':>{col_w['p10']}}"
        f"{'BT top40%':>{col_w['bt40']}}"
        f"{'BT top30%':>{col_w['bt30']}}"
        f"{'BT top20%':>{col_w['bt20']}}"
        f"{'BT top10%':>{col_w['bt10']}}"
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
        fee_str = f"{row['fee_pct']:.4f}%"
        net_str = f"{row['net_abs']:+.6f}%"
        cross_str = str(int(row['crossings']))
        p40_str  = f"{row['p40']:.4f}%"
        p30_str  = f"{row['p30']:.4f}%"
        p20_str  = f"{row['p20']:.4f}%"
        p10_str  = f"{row['p10']:.4f}%"
        bt40_str = f"{row['bt40']:+.2f}%"
        bt30_str = f"{row['bt30']:+.2f}%"
        bt20_str = f"{row['bt20']:+.2f}%"
        bt10_str = f"{row['bt10']:+.2f}%"
        print(
            f"{row['symbol']:<{col_w['symbol']}}"
            f"{int(row['bars']):>{col_w['bars']}}"
            f"{avg_str:>{col_w['avg']}}"
            f"{pos_str:>{col_w['positive']}}"
            f"{neg_str:>{col_w['negative']}}"
            f"{abs_str:>{col_w['abs_avg']}}"
            f"{up_str:>{col_w['pct_up']}}"
            f"{fee_str:>{col_w['fee_pct']}}"
            f"{net_str:>{col_w['net_abs']}}"
            f"{cross_str:>{col_w['crossings']}}"
            f"{p40_str:>{col_w['p40']}}"
            f"{p30_str:>{col_w['p30']}}"
            f"{p20_str:>{col_w['p20']}}"
            f"{p10_str:>{col_w['p10']}}"
            f"{bt40_str:>{col_w['bt40']}}"
            f"{bt30_str:>{col_w['bt30']}}"
            f"{bt20_str:>{col_w['bt20']}}"
            f"{bt10_str:>{col_w['bt10']}}"
        )

    print(sep)

    # Résumé global
    avg_all   = df_out["avg"].mean()
    avg_pos   = df_out["positive"].mean()
    avg_neg   = df_out["negative"].mean()
    avg_abs   = df_out["abs_avg"].mean()
    avg_up    = df_out["pct_up"].mean()
    avg_fee   = df_out["fee_pct"].mean()
    avg_net   = df_out["net_abs"].mean()
    avg_cross = df_out["crossings"].mean()
    avg_p40   = df_out["p40"].mean()
    avg_p30   = df_out["p30"].mean()
    avg_p20   = df_out["p20"].mean()
    avg_p10   = df_out["p10"].mean()
    avg_bt40  = df_out["bt40"].mean()
    avg_bt30  = df_out["bt30"].mean()
    avg_bt20  = df_out["bt20"].mean()
    avg_bt10  = df_out["bt10"].mean()
    print(
        f"{'MOYENNE GLOBALE':<{col_w['symbol']}}"
        f"{'':>{col_w['bars']}}"
        f"{avg_all:>+{col_w['avg'] - 1}.6f}%"
        f"{avg_pos:>+{col_w['positive'] - 1}.6f}%"
        f"{avg_neg:>+{col_w['negative'] - 1}.6f}%"
        f"{avg_abs:>{col_w['abs_avg'] - 1}.6f}%"
        f"{avg_up:>{col_w['pct_up'] - 1}.1f}%"
        f"{avg_fee:>{col_w['fee_pct'] - 1}.4f}%"
        f"{avg_net:>+{col_w['net_abs'] - 1}.6f}%"
        f"{avg_cross:>{col_w['crossings'] - 1}.1f}"
        f"{avg_p40:>{col_w['p40'] - 1}.4f}%"
        f"{avg_p30:>{col_w['p30'] - 1}.4f}%"
        f"{avg_p20:>{col_w['p20'] - 1}.4f}%"
        f"{avg_p10:>{col_w['p10'] - 1}.4f}%"
        f"{avg_bt40:>+{col_w['bt40'] - 1}.2f}%"
        f"{avg_bt30:>+{col_w['bt30'] - 1}.2f}%"
        f"{avg_bt20:>+{col_w['bt20'] - 1}.2f}%"
        f"{avg_bt10:>+{col_w['bt10'] - 1}.2f}%"
    )
    print()


if __name__ == "__main__":
    main()
