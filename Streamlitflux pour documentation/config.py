# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Données météo (valeurs par défaut, à surcharger dynamiquement si besoin)
LAT = 43.716667
LON = -1.050000
EXOG_FEATURES = [
    'temperature_max', 'temperature_min', 'precipitation',
    'is_vacation', 'is_public_holiday', 'days_in_week'
    # Ajoute ici toute autre variable exogène pertinente
]
# Données historiques
HISTORICAL_FILE = os.path.join(BASE_DIR, "Flux_final.xlsx")
HISTORICAL_EXOG = os.path.join(BASE_DIR, "Météo_SUD.xlsx")
# API météo et proxy
API_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
USE_PROXY = True
PROXY_URL = "http://127.0.0.1:3128" if USE_PROXY else None

# Utilitaire pour générer les chemins modèles dynamiquement
def get_model_paths(cible):
    model_path = os.path.join(BASE_DIR, 'models', f"{cible}_models")
    return {
        "MODEL_PATH": model_path,
        "MODEL_FILE": os.path.join(model_path, f"sarimax_model_{cible}.pkl"),
        "SCALER_EXOG_FILE": os.path.join(model_path, f"scaler_exog_{cible}.pkl"),
        "SCALER_TARGET_FILE": os.path.join(model_path, f"scaler_target_{cible}.pkl"),
        "PCA_FILE": os.path.join(model_path, f"pca_{cible}.pkl"),
    }
