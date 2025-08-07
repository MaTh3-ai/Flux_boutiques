import pandas as pd
import numpy as np
from config import HISTORICAL_FILE
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

def load_historical_data(cible: str):
    """
    Charge l’historique de la cible, renvoie une série hebdo unique
    et un calendrier [Date, Année, Semaine], sans doublons.
    """
    print("\n=== [DEBUG] Début load_historical_data ===")

    df = pd.read_excel(HISTORICAL_FILE)
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = df.dropna(subset=["Annee", "Semaine"])
    df["Annee"] = df["Annee"].astype(int)
    df["Semaine"] = df["Semaine"].astype(int)

    df["Date"] = df.apply(week_to_date, axis=1)
    df = df.sort_values(["Annee", "Semaine", "Date"]).reset_index(drop=True)
    # Suppression explicite des doublons de Date (on garde le dernier)
    if df["Date"].duplicated().any():
        print("[DEBUG] Doublons trouvés, suppression :", df[df["Date"].duplicated(keep=False)][["Annee","Semaine","Date"]])
        df = df.drop_duplicates(subset="Date", keep="last")
    df = df.reset_index(drop=True)

    if cible not in df.columns:
        raise ValueError(f"Cible « {cible} » introuvable dans l’historique.")
    y_hist = df.set_index("Date")[cible]
    cal_df = df[["Date", "Annee", "Semaine"]].copy()

    # Décalage des lags par recherche SEMAINE+ANNEE et non pas juste index
    def lag_date(date, lag_years):
        year = date.year - lag_years
        week = date.isocalendar().week
        # On cherche la 1e date qui a même (année, semaine)
        match = cal_df[(cal_df["Annee"] == year) & (cal_df["Semaine"] == week)]
        return match["Date"].iloc[0] if not match.empty else pd.NaT

    hist_n1 = y_hist.reindex([lag_date(d, 1) for d in y_hist.index])
    hist_n2 = y_hist.reindex([lag_date(d, 2) for d in y_hist.index])
    hist_n1.index = y_hist.index
    hist_n2.index = y_hist.index

    print("=== [DEBUG] Fin load_historical_data ===\n")
    return y_hist, hist_n1, hist_n2, cal_df
