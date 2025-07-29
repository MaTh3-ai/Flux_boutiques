import pandas as pd
import numpy as np
from config import HISTORICAL_FILE
from app.utils.weather_fetcher import compute_custom_week_counts_for_period
import locale
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

def load_historical_data(cible):
    print("\n=== [DEBUG] Début load_historical_data ===")

    df = pd.read_excel(HISTORICAL_FILE)
    print(f"[DEBUG] Taille initiale du df_hist : {df.shape}, NaN totaux : {df.isnull().sum().sum()}")

    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    assert {'Annee','Semaine'}.issubset(df.columns), "Colonnes Annee/Semaine manquantes"
    df = df.dropna(subset=['Annee','Semaine'])
    df['Annee'] = df['Annee'].astype(int)
    df['Semaine'] = df['Semaine'].astype(int)

    df['Date'] = df.apply(week_to_date, axis=1)
    print(f"[DEBUG] Date conversion : {df['Date'].isnull().sum()} valeurs manquantes")

    df = df.sort_values(['Annee','Semaine','Date']).reset_index(drop=True)

    if cible not in df.columns:
        raise ValueError(f"Cible '{cible}' introuvable dans l'historique.")

    y = df[cible]
    hist_n1 = y.shift(52)
    hist_n2 = y.shift(104)

    cal_df = df[['Date','Annee','Semaine']].copy()

    print(f"[DEBUG] y.shape       : {y.shape}")
    print(f"[DEBUG] hist_n1 NaN   : {hist_n1.isnull().sum()}, hist_n2 NaN : {hist_n2.isnull().sum()}")
    print(f"[DEBUG] cal_df.shape  : {cal_df.shape}")

    print("=== [DEBUG] Fin load_historical_data ===\n")
    return (
        y.reset_index(drop=True),
        hist_n1.reset_index(drop=True),
        hist_n2.reset_index(drop=True),
        cal_df.reset_index(drop=True)

    )

def build_weekly_lags(cal_df: pd.DataFrame, y_series: pd.Series, lags=(1, 2)) -> pd.DataFrame:
    base = cal_df.set_index("Date").copy()
    out  = pd.DataFrame(index=base.index)
    for k in lags:
        shifted = base.index - pd.Timedelta(weeks=52 * k)
        out[f"Hist_N-{k}"] = y_series.reindex(shifted).values
    return out.reindex(base.index)
