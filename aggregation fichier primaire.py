import pandas as pd
from datetime import datetime, timedelta
import re

# Dictionnaire mois français -> numéro
FRENCH_MONTHS = {
    'janvier': '01', 'février': '02', 'fevrier': '02', 'mars': '03',
    'avril': '04', 'mai': '05', 'juin': '06',
    'juillet': '07', 'août': '08', 'aout': '08', 'septembre': '09',
    'octobre': '10', 'novembre': '11', 'décembre': '12', 'decembre': '12'
}

# Calcul de la semaine personnalisée

def custom_week(dt):
    jan1 = datetime(dt.year, 1, 1)
    first_sun = jan1 + timedelta(days=(6 - jan1.weekday()))
    if dt <= first_sun:
        return 1
    return 2 + ((dt - first_sun - timedelta(days=1)).days // 7)

# Lecture + parsing + agrégation hebdo

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
    # Remplacer le mois français par son numéro
    for fr, num in FRENCH_MONTHS.items():
        df[date_col] = df[date_col].str.replace(fr, num, regex=False)
    # Garder uniquement jour-numéro-mois-numéro-année
    df['DATE'] = pd.to_datetime(
        df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[0]
        + '-' + df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[1]
        + '-' + df[date_col].str.extract(r"(\d{1,2})[- ](\d{2})[- ](\d{4})")[2],
        format="%d-%m-%Y", errors='coerce'
    )

    # 4. Filtrer et préparer pour agrégation
    df = df.dropna(subset=['DATE']).copy()
    df['Annee'] = df['DATE'].dt.year
    df['Semaine'] = df['DATE'].apply(custom_week)

    # 5. Somme par année/semaine
    shops = df.columns.difference(['DATE', 'Annee', 'Semaine'])
    result = (df.groupby(['Annee', 'Semaine'])[shops]
                .sum()
                .reset_index()
             )

    # 6. Enregistrer
    result.to_excel(output_path, index=False)
    print(f"Enregistré dans {output_path}")

if __name__ == '__main__':
    process('C:/Users/NFGS3174/Downloads/streamlit_Flux/Flux_brut.xlsx', 'Flux_final.xlsx')