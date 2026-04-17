"""
Strategy Percentile — Mean Reversion vers la SMA
-------------------------------------------------
Stratégie de mean reversion basée sur les percentiles d'excursion close-based.

Logique :
  - Warmup de `warmup_years` ans : accumulation des excursions historiques,
    mais aucun trade.
  - Un run correspond à une suite de bougies du même côté de la SMA.
  - Après `min_bars` bougies confirmées dans un run, si l'excursion close-based
    dépasse un seuil historique (percentile), entrée à l'open de la bougie suivante.
  - Sortie lorsqu'un crossing est détecté sur la clôture d'une bougie :
    exécution à l'open de la bougie suivante.
  - Une seule entrée par run et par niveau.
  - Les positions encore ouvertes à la fin sont clôturées au dernier close.

Niveaux d'entrée :
  - bt40 : excursion > percentile 60 de l'historique
  - bt30 : excursion > percentile 70 de l'historique
  - bt20 : excursion > percentile 80 de l'historique
  - bt10 : excursion > percentile 90 de l'historique
"""

import numpy as np
import pandas as pd


def backtest_strategy(
    df: pd.DataFrame,
    period: int,
    fee_pct: float,
    min_bars: int = 3,
    warmup_years: int = 2,
    min_excursions_history: int = 10,
) -> dict:
    """Backteste une stratégie de mean reversion vers la SMA.

    Paramètres
    ----------
    df : pd.DataFrame
        Données OHLCV avec colonnes ``Datetime``, ``Open``, ``Close``.
    period : int
        Longueur de la SMA.
    fee_pct : float
        Frais par transaction en pourcentage (appliqués deux fois : entrée + sortie).
    min_bars : int
        Nombre minimal de bougies consécutives du même côté de la SMA pour
        qu'un run soit considéré comme valide.
    warmup_years : int
        Durée de la période de warmup (accumulation sans trades).
    min_excursions_history : int
        Nombre minimal d'excursions historiques pour calculer les seuils.

    Retourne
    --------
    dict avec les clés ``bt40``, ``bt30``, ``bt20``, ``bt10`` (PnL cumulé en %).
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
