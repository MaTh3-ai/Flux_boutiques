import pandas as pd
import numpy as np
from config import HISTORICAL_FILE
from app.utils.weather_fetcher import compute_custom_week_counts_for_period
import locale
import sqlite3
import os
locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")   # ou "fr_FR" sous Windows

def week_to_custom_date(year, week):
    first_jan = pd.Timestamp(year, 1, 1)
    first_jan_weekday = first_jan.weekday()
    if week == 1:
        return first_jan
    else:
        days_until_next_monday = (7 - first_jan_weekday) % 7
        first_monday = first_jan + pd.Timedelta(days=days_until_next_monday)
        week_start = first_monday + pd.Timedelta(weeks=week - 2)
        return week_start

def week_to_date(row):
    return week_to_custom_date(int(row['Annee']), int(row['Semaine']))

def load_historical_data(cible: str):
    """
    Retourne :
        y_hist   : série complète (hebdo) de la cible
        hist_n1  : y_hist décalé de –52 semaines
        hist_n2  : y_hist décalé de –104 semaines
        cal_df   : DataFrame [Date, Annee, Semaine] sur la même grille

    Toutes les dates sont déjà dans le calendrier « custom » et triées.
    """
    print("\n=== [DEBUG] Début load_historical_data ===")

    df = pd.read_excel(HISTORICAL_FILE)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    # sécurité : on impose int
    df = df.dropna(subset=["Annee", "Semaine"])
    df["Annee"] = df["Annee"].astype(int)
    df["Semaine"] = df["Semaine"].astype(int)

    # date ISO → calendrier custom
    df["Date"] = df.apply(week_to_date, axis=1)
    df = df.sort_values(["Annee", "Semaine", "Date"]).reset_index(drop=True)

    # Serie cible
    if cible not in df.columns:
        raise ValueError(f"Cible « {cible} » introuvable dans l’historique.")
    y_hist   = df.set_index("Date")[cible]
    hist_n1  = y_hist.shift(52)
    hist_n2  = y_hist.shift(104)

    cal_df = df[["Date", "Annee", "Semaine"]].copy()

    print("=== [DEBUG] Fin load_historical_data ===\n")
    return y_hist, hist_n1, hist_n2, cal_df


def build_weekly_lags(cal_df: pd.DataFrame, y_series: pd.Series, lags=(1, 2)) -> pd.DataFrame:
    base = cal_df.set_index("Date").copy()

    # Assure que y_series est indexée par les mêmes dates que cal_df
    if not isinstance(y_series.index, pd.DatetimeIndex):
        y_series = pd.Series(y_series.values, index=base.index)
    else:
        y_series = y_series.reindex(base.index)

    out = pd.DataFrame(index=base.index)
    for k in lags:
        shifted_index = base.index - pd.Timedelta(weeks=52 * k)
        out[f"Hist_N-{k}"] = y_series.reindex(shifted_index).values
    return out.reindex(base.index)

def get_all_boutiques(db_path: str | None = None) -> list[str]:
    """Retourne la liste des boutiques connues dans la base SQLite."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'database', 'boutiques.db')
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT nom_boutique FROM boutiques").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]
