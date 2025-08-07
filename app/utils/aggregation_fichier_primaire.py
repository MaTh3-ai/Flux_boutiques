import pandas as pd
from datetime import datetime, timedelta
from app.utils.weather_fetcher import WeatherDataFetcher
from config import HISTORICAL_EXOG, LAT, LON, HISTORICAL_FILE, RAW_HISTORICAL_FILE
from app.utils.exogenous import exo_var


# Dictionnaire mois français -> numéro
FRENCH_MONTHS = {
    'janvier': '01', 'février': '02', 'fevrier': '02', 'mars': '03',
    'avril': '04', 'mai': '05', 'juin': '06',
    'juillet': '07', 'août': '08', 'aout': '08', 'septembre': '09',
    'octobre': '10', 'novembre': '11', 'décembre': '12', 'decembre': '12'
}

def custom_week(dt):
    jan1 = datetime(dt.year, 1, 1)
    first_sun = jan1 + timedelta(days=(6 - jan1.weekday()))
    if dt <= first_sun:
        return 1
    return 2 + ((dt - first_sun - timedelta(days=1)).days // 7)

def process(input_path, output_path):
    # 1. Détecter la ligne d'en-tête contenant 'DATE'
    raw = pd.read_excel(input_path, header=None)
    mask = raw.iloc[:, 0].astype(str).str.strip().str.lower() == 'date'
    if not mask.any():
        raise ValueError("Ligne d'en-tête 'DATE' introuvable.")
    hdr = mask.idxmax()

    # 2. Charger avec header, supprimer colonnes vides ou 'Unnamed'
    df = (pd.read_excel(input_path, header=hdr)
            .loc[:, lambda d: ~d.columns.str.contains('^Unnamed')]
          )

    # 3. Convertir la première colonne en datetime (papier français)
    date_col = df.columns[0]
    df[date_col] = df[date_col].astype(str).str.lower()
    for fr, num in FRENCH_MONTHS.items():
        df[date_col] = df[date_col].str.replace(fr, num, regex=False)
    df['Date'] = pd.to_datetime(
        df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[0]
        + '-' + df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[1]
        + '-' + df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[2],
        format="%d-%m-%Y", errors='coerce'
    )

    # 4. Filtrer et préparer pour agrégation
    df = df.dropna(subset=['Date']).copy()
    df['Annee'] = df['Date'].dt.year
    df['Semaine'] = df['Date'].apply(custom_week)

    # 5. Somme par année/semaine (semaines partielles incluses)
    shops = df.columns.difference(['Date', 'Annee', 'Semaine'])
    cols_num = df[shops].select_dtypes(include='number').columns
    result = (
        df.groupby(['Annee', 'Semaine'])[cols_num]
        .sum()
        .reset_index()
    )

    # 6. Enregistrer
    result.to_excel(output_path, index=False)
    print(f"Enregistré dans {output_path}")


def update_all_historicals():
    # 1. Mise à jour de la donnée principale (boutiques/cartes)
    process(RAW_HISTORICAL_FILE, HISTORICAL_FILE)

    # 2. Mise à jour des données météo brutes
    print("🔄 Mise à jour du fichier météo…")
    histo_path = HISTORICAL_EXOG
    if not pd.io.common.file_exists(histo_path):
        raise FileNotFoundError(f"Fichier météo {histo_path} introuvable.")
    histo = pd.read_excel(histo_path)
    # Détection robuste de la colonne date
    date_col = next((col for col in histo.columns if col.lower() == 'date'), None)
    if not date_col:
        raise ValueError("Aucune colonne 'date' trouvée dans l'historique météo.")
    histo[date_col] = pd.to_datetime(histo[date_col])
    date_min = histo[date_col].min()
    date_max = histo[date_col].max()

   # 3. Date max fixée à aujourd'hui
    date_max = pd.Timestamp(datetime.today().date())

    # 4. Mise à jour des données météo brutes jusqu'à aujourd'hui
    print("🔄 Mise à jour du fichier météo…")
    histo_path = HISTORICAL_EXOG
    if not pd.io.common.file_exists(histo_path):
        raise FileNotFoundError(f"Fichier météo {histo_path} introuvable.")
    fetcher = WeatherDataFetcher(LAT, LON, proxy_url="http://localhost:3128")
    fetcher.update_historic_file(histo_path, date_max)


    # 4. Génération et sauvegarde du fichier exogène complet via exo_var
    print("🧩 Calcul des variables exogènes complètes…")
    exog_df = exo_var(date_min, date_max)

    # 5. Nettoyage (suppression des lignes avec valeurs manquantes)
    cols_obligatoires = ['Date', 'Annee', 'Semaine',
                         'temperature_max', 'temperature_min', 'precipitation',
                         'is_vacation', 'is_public_holiday', 'days_in_week']
    exog_df = exog_df.dropna(subset=cols_obligatoires, how='any')

    try:
        exog_df.to_excel(histo_path, index=False)
        print("✅ Fichier météo + exogènes mis à jour et complété.")
    except PermissionError:
        print(f"❌ Impossible d’écrire dans {histo_path}. Fermez le fichier Excel, puis réessayez.")
        raise
