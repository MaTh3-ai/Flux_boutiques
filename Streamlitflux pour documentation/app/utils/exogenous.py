from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import holidays
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import requests
from app.utils.weather_fetcher import WeatherDataFetcher, compute_custom_week_counts_for_period
from config import LAT, LON, API_METEO_URL, PROXY_URL, HISTORICAL_EXOG



def is_vacation(date: datetime) -> int:
    vacation_periods = [
        ((2, 18), (3, 6)),    # Vacances d'hiver
        ((4, 15), (5, 2)),    # Vacances de printemps
        ((7, 1), (8, 31)),    # Vacances d'été
        ((10, 21), (11, 6)),  # Vacances de la Toussaint
        ((12, 23), (1, 8))    # Vacances de Noël
    ]
    for start, end in vacation_periods:
        start_date = datetime(date.year, start[0], start[1])
        end_date = datetime(date.year, end[0], end[1])
        if start[0] == 12 and end[0] == 1:
            end_date = datetime(date.year + 1, end[0], end[1])
        if start_date <= date <= end_date:
            return 1
    return 0

def is_sales_period(date: datetime) -> int:
    return int(date.month in [1, 7])

def is_new_iphone_launched(date: datetime) -> int:
    return int(date.month == 9)

def is_public_holiday(date: datetime) -> int:
    fr_holidays = holidays.France(years=date.year)
    return int(date in fr_holidays)

def add_exogenous_variables(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' not in df.columns:
        raise ValueError("La colonne 'date' est absente du DataFrame.")
    df['date'] = pd.to_datetime(df['date'])

    # Ajouter les colonnes Annee et Semaine
    start_date = df['date'].min()
    end_date = df['date'].max()
    week_counts = compute_custom_week_counts_for_period(start_date, end_date)

    def get_year_and_week(date):
        week_info = week_counts[(week_counts['week_start'] <= date) & (week_counts['week_end'] >= date)]
        if not week_info.empty:
            return week_info.iloc[0]['year'], week_info.iloc[0]['week']
        return None, None

    df['Annee'], df['Semaine'] = zip(*df['date'].apply(lambda x: get_year_and_week(x)))

    df['is_vacation'] = df['date'].apply(is_vacation)
    df['is_public_holiday'] = df['date'].apply(is_public_holiday)
    return df

def add_time_features(df):
    df = df.copy()
    # Utilisation du découpage custom pour année/semaine
    start_date = df['date'].min()
    end_date = df['date'].max()
    week_counts = compute_custom_week_counts_for_period(start_date, end_date)

    # Création des colonnes 'custom_year' et 'custom_week'
    def get_custom_year_and_week(date):
        week_info = week_counts[(week_counts['week_start'] <= date) & (week_counts['week_end'] >= date)]
        if not week_info.empty:
            return week_info.iloc[0]['year'], week_info.iloc[0]['week']
        return None, None

    df["custom_year"], df["custom_week"] = zip(*df["date"].apply(get_custom_year_and_week))
    # Prendre la base du nombre de semaines max trouvé dans week_counts (pour la périodicité)
    nb_weeks = week_counts['week'].max()
    df["sin_week"] = np.sin(2 * np.pi * df["custom_week"] / nb_weeks)
    df["cos_week"] = np.cos(2 * np.pi * df["custom_week"] / nb_weeks)
    return df

def get_custom_week_starts_covering(start_date, end_date):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    week_grid = compute_custom_week_counts_for_period(start_date, end_date)
    return pd.to_datetime(week_grid['week_start']).sort_values().unique()


def fetch_weather_forecast(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto"
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if "daily" not in data or "time" not in data["daily"]:
        return pd.DataFrame()
    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "temperature_max": data["daily"]["temperature_2m_max"],
        "temperature_min": data["daily"]["temperature_2m_min"],
        "precipitation": data["daily"]["precipitation_sum"]
    })
    df["date"] = pd.to_datetime(df["date"])
    df["source"] = "forecast"
    return df

def aggregate_daily_to_custom_week(df_daily):
    week_counts = compute_custom_week_counts_for_period(df_daily.date.min(),
                                                        df_daily.date.max())

    def get_week_start(d):
        row = week_counts[(week_counts.week_start <= d) & (week_counts.week_end >= d)]
        return row.iloc[0]["week_start"]

    df_daily["week_start"] = df_daily["date"].apply(get_week_start)
    weekly = (df_daily
              .groupby("week_start")
              .agg({
                 "temperature_max":"mean",
                 "temperature_min":"mean",
                 "precipitation":"mean",
                 # nouveaux indicateurs : on prend l'indicateur du **lundi** de la semaine
                 "is_vacation":"max",
                 "is_public_holiday":"max"
              })
              .reset_index()
              .rename(columns={"week_start":"date"}))
    # days_in_week = nombre réel de jours couverts par cette semaine dans le calendrier custom
    week_counts = compute_custom_week_counts_for_period(df_daily.date.min(),
                                                        df_daily.date.max())
    weekly["days_in_week"] = weekly["date"].apply(
        lambda d: week_counts.loc[week_counts.week_start == d,"days_in_week"].iloc[0]
    )
    return weekly

def impute_missing_weeks_ridge(df_hist, missing_dates):
    imputations = []
    if df_hist.empty:
        for d in missing_dates:
            imputations.append({
                "date": pd.Timestamp(d),
                "temperature_max": np.nan,
                "temperature_min": np.nan,
                "precipitation": np.nan,
                "source": "ridge"
            })
        return pd.DataFrame(imputations)
    # 1) Features temporelles
    df_hist = add_time_features(df_hist)
    feature_cols = ["sin_week", "cos_week", "custom_year"]   # <— correct

    # 2) Pipeline (imputer + régression) pour éviter tout NaN résiduel
    pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=1.0)
    )

    models = {}
    for var in ["temperature_max", "temperature_min", "precipitation"]:
        X = df_hist[feature_cols].values
        y = df_hist[var].values
        pipe.fit(X, y)
        models[var] = pipe
    df_missing = pd.DataFrame({"date": pd.to_datetime(missing_dates)})
    df_missing = add_time_features(df_missing)
    X_missing = df_missing[feature_cols].values

    preds = {var: models[var].predict(X_missing) for var in ["temperature_max",
                                                                "temperature_min",
                                                                "precipitation"]}

    for i, date_i in enumerate(df_missing["date"]):
        imputations.append({
            "date":           date_i,
            "temperature_max": preds["temperature_max"][i],
            "temperature_min": preds["temperature_min"][i],
            "precipitation":   max(0, preds["precipitation"][i]),  # pas de valeurs <0
            "source":          "ridge"
        })
    return pd.DataFrame(imputations)


def compute_custom_week_counts_for_period(start_date, end_date):
    """
    Pour la période [start_date, end_date], calcule pour chaque année
    le découpage en semaines avec les bornes réelles (en respectant les limites de l'année).
    Toutes les dates sont des pd.Timestamp.
    """
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    results = []

    for year in range(start_date.year, end_date.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        effective_start = max(year_start, start_date)
        effective_end = min(year_end, end_date)

        # Première semaine : du 1er janvier au premier dimanche
        first_week_start = effective_start
        first_week_end = first_week_start + timedelta(days=(6 - first_week_start.weekday()))
        if first_week_end > effective_end:
            first_week_end = effective_end
        if effective_start > first_week_start:
            first_week_start = effective_start

        week_number = 1
        results.append({
            'year': year,
            'week': week_number,
            'week_start': first_week_start,
            'week_end': first_week_end,
            'days_in_week': max(1, (first_week_end - first_week_start).days + 1)  # Garantir au moins 1 jour
        })

        # Semaines suivantes
        week_number += 1
        next_week_start = first_week_end + timedelta(days=1)
        while next_week_start <= effective_end:
            week_start = next_week_start
            week_end = week_start + timedelta(days=6)
            if week_end > effective_end:
                week_end = effective_end

            # Calculer days_in_week avec une contrainte minimale de 1 jour
            days_in_week = max(1, (week_end - week_start).days + 1)

            results.append({
                'year': year,
                'week': week_number,
                'week_start': week_start,
                'week_end': week_end,
                'days_in_week': days_in_week
            })

            week_number += 1
            next_week_start = week_end + timedelta(days=1)

    return pd.DataFrame(results)

def generate_custom_week_grid(start_date, end_date):
    """
    Génère une grille de semaines personnalisée entre start_date et end_date.
    """
    week_counts = compute_custom_week_counts_for_period(start_date, end_date)
    week_grid = pd.DataFrame({
        'Date': pd.to_datetime(week_counts['week_start']),
        'Annee': week_counts['year'],
        'Semaine': week_counts['week'],
        'days_in_week': week_counts['days_in_week']
    })
    week_grid = week_grid.sort_values(['Annee', 'Semaine', 'Date']).reset_index(drop=True)
    return week_grid

def exo_var(start_date, end_date) -> pd.DataFrame:
    print(f"\n=== [DEBUG] Début exo_var ===")
    print(f"[DEBUG] start_date: {start_date}, end_date: {end_date}")

    # Étape 1 : grille des semaines personnalisées
    week_grid = generate_custom_week_grid(pd.Timestamp(start_date), pd.Timestamp(end_date))
    df_weeks = week_grid.copy()
    df_weeks['Date'] = pd.to_datetime(df_weeks['Date'])

    print(f"[DEBUG] df_weeks shape : {df_weeks.shape}")
    print(f"[DEBUG] df_weeks - dates min/max : {df_weeks['Date'].min()} -> {df_weeks['Date'].max()}")

    # Étape 2 : chargement données historiques
    if os.path.exists(HISTORICAL_EXOG):
        df_hist = pd.read_excel(HISTORICAL_EXOG)
        df_hist['date'] = pd.to_datetime(df_hist['date'])
        df_hist = add_exogenous_variables(df_hist)
    else:
        df_hist = pd.DataFrame(columns=["date", "temperature_max", "temperature_min", "precipitation", "source", "Annee", "Semaine"])
    
    print(f"[DEBUG] df_hist shape : {df_hist.shape}, dates : {df_hist['date'].min()} -> {df_hist['date'].max()}")

    # Étape 3 : appel API pour forecast à court terme
    today = pd.Timestamp(datetime.today().date())
    forecast_end = min(today + pd.Timedelta(days=15), pd.Timestamp(end_date))
    df_forecast_weekly = pd.DataFrame()

    if today <= forecast_end:
        df_forecast = fetch_weather_forecast(LAT, LON, today, forecast_end)
        if not df_forecast.empty:
            df_forecast = add_exogenous_variables(df_forecast)
            df_forecast_weekly = aggregate_daily_to_custom_week(df_forecast)
    
    print(f"[DEBUG] df_forecast_weekly shape : {df_forecast_weekly.shape}, preview:\n{df_forecast_weekly.head()}")

    # Étape 4 : concat historique + forecast
    df_all = pd.concat([df_hist, df_forecast_weekly], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all = df_all.drop_duplicates(subset="date", keep="last")

    print(f"[DEBUG] df_all shape : {df_all.shape}")
    print(f"[DEBUG] df_all - dates min/max : {df_all['date'].min()} -> {df_all['date'].max()}")

    # Étape 5 : identifier les semaines manquantes
    known_weeks = set(df_all["date"].dt.normalize())
    missing_dates = [d for d in df_weeks['Date'].dt.normalize() if d not in known_weeks]
    print(f"[DEBUG] Nombre de dates manquantes : {len(missing_dates)}")

    # Étape 6 : imputations ridge
    df_ridge = impute_missing_weeks_ridge(df_all, missing_dates) if missing_dates else pd.DataFrame()
    if not df_ridge.empty:
        df_ridge = add_exogenous_variables(df_ridge)
    print(f"[DEBUG] df_ridge shape : {df_ridge.shape}")

    # Étape 7 : tout concaténer
    df_all = pd.concat([df_all, df_ridge], ignore_index=True)
    df_all['date'] = pd.to_datetime(df_all['date'])

    # DEBUG intermédiaire
    print("\n>>> Pré-merge - Aperçu de df_weeks:")
    print(df_weeks.head(10))
    print("\n>>> Pré-merge - Aperçu de df_all[['date','source']]:")
    print(df_all.head(10))

    # Étape 8 : merge exogène sur semaines
    df_final = df_weeks.merge(
        df_all,
        left_on='Date',
        right_on='date',
        how='left',
        suffixes=('', '_exog')
    )

    # Étape 9 : colonnes finales et imputations
    exog_cols = [
        'temperature_max', 'temperature_min', 'precipitation',
        'is_vacation', 'is_public_holiday', 'days_in_week'
    ]

    # Remplissage des colonnes manquantes
    if 'days_in_week' not in df_final.columns:
        df_final['days_in_week'] = 7
    df_final['Annee'] = df_final['Annee'].fillna(df_final['Annee_exog'])
    df_final['Semaine'] = df_final['Semaine'].fillna(df_final['Semaine_exog'])

    df_final = df_final.sort_values(['Annee', 'Semaine', 'Date']).reset_index(drop=True)
    df_final = df_final[['Date', 'Annee', 'Semaine'] + exog_cols]

    # Imputation ffill/bfill
    df_final[exog_cols] = df_final[exog_cols].ffill().bfill()

    # Dernier rempart : médiane si besoin
    if df_final[exog_cols].isnull().any().any():
        print("[WARNING] Imputation médiane appliquée !")
        df_final[exog_cols] = df_final[exog_cols].fillna(df_final[exog_cols].median())

    # Vérification finale
    if df_final[exog_cols].isnull().any().any():
        raise ValueError("NaN résiduels après imputation finale !")

    # Affichage final
    print("\n>>> EXOG FINAL SAMPLE:")
    print(df_final.head(10))
    print("\n>>> STATISTIQUES DESCRIPTIVES:")
    print(df_final.describe(include='all'))

    print("=== [DEBUG] Fin exo_var ===\n")

    return df_final


def verify_data_completeness(df_weeks, df_final):
    # Vérifier que toutes les semaines présentes dans l’historique sont aussi présentes dans les exogènes, et vice versa.
    weeks_hist = set(df_weeks['Date'].dt.normalize())
    weeks_exog = set(df_final['Date'].dt.normalize())
    if weeks_hist != weeks_exog:
        raise ValueError("Les semaines dans l'historique et les exogènes ne correspondent pas.")

    # Vérifier l'absence de valeurs NaN/NaT dans les colonnes critiques
    critical_columns = ['Date', 'Annee', 'Semaine', 'temperature_max', 'temperature_min', 'precipitation', 'is_vacation', 'is_public_holiday', 'days_in_week']
    if df_final[critical_columns].isnull().any().any():
        raise ValueError("Des valeurs NaN/NaT sont présentes dans les colonnes critiques.")

    return True
