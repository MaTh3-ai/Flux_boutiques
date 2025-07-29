import requests

# Définition du proxy CNTLM
proxies = {
    "http": "http://localhost:3128",
    "https": "http://localhost:3128",
}

# URL de l'API
API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Paramètres minimaux de la requête
params = {
    "latitude": 48.8566,  # Coordonnées de Paris en exemple
    "longitude": 2.3522,
    "start_date": "2024-01-01",
    "end_date": "2024-01-02",
    "daily": "temperature_2m_max",
    "timezone": "Europe/Paris"
}

try:
    response = requests.get(API_URL, params=params, proxies=proxies)
    response.raise_for_status()  # Vérifie si la requête a réussi
    print(response.json())  # Affiche la réponse en JSON
except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête : {e}")
