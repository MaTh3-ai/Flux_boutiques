import asyncio
import random
import os
from datetime import datetime, timedelta, date
import pandas as pd
import aiohttp

class WeatherDataFetcher:
    def __init__(self, lat, lon, api_url="https://archive-api.open-meteo.com/v1/archive", proxy_url=None):
        self.lat = lat
        self.lon = lon
        self.api_url = api_url
        self.proxy_url = proxy_url

    async def fetch_single_date_async(self, session, date):
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': date.strftime('%Y-%m-%d'),
            'end_date': date.strftime('%Y-%m-%d'),
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
            'timezone': 'auto'
        }
        try:
            async with session.get(self.api_url, params=params, proxy=self.proxy_url, timeout=15) as response:
                response.raise_for_status()
                data = await response.json()
                if 'daily' in data and data['daily'].get('time'):
                    return {
                        'date': data['daily']['time'][0],
                        'temperature_max': data['daily']['temperature_2m_max'][0],
                        'temperature_min': data['daily']['temperature_2m_min'][0],
                        'precipitation': data['daily']['precipitation_sum'][0]
                    }
                return None
        except Exception as e:
            print(f"❌ Erreur lors de la récupération de {date.strftime('%Y-%m-%d')} : {e}")
            return None

    async def fetch_dates_in_batch(self, dates, batch_size=10):
        all_data = []
        total_dates = len(dates)
        for i in range(0, total_dates, batch_size):
            batch = dates[i:i + batch_size]
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_single_date_async(session, date) for date in batch]
                batch_results = await asyncio.gather(*tasks)
                valid_results = [r for r in batch_results if r]
                all_data.extend(valid_results)
            await asyncio.sleep(random.uniform(1, 3))
        return all_data

    def fetch_weather_data_optimized(self, start_date, end_date, batch_size=10):
        start_date = start_date if isinstance(start_date, date) else start_date.date()
        end_date = end_date if isinstance(end_date, date) else end_date.date()
        today = datetime.today().date()
        if start_date > today:
            print(f"🚫 start_date {start_date} est dans le futur. Pas de données disponibles.")
            return pd.DataFrame()
        if end_date > today:
            print(f"⚠️ end_date {end_date} est dans le futur. Limitation à aujourd'hui : {today}.")
            end_date = today
        dates_to_fetch = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        if not dates_to_fetch:
            print("🚫 Aucun jour à récupérer. Vérifiez la plage de dates.")
            return pd.DataFrame()
        all_data = asyncio.run(self.fetch_dates_in_batch(dates_to_fetch, batch_size=batch_size))
        if not all_data:
            print(f"❗ Aucune donnée récupérée pour {start_date} -> {end_date} sur la zone ({self.lat}, {self.lon})")
            return pd.DataFrame()
        return pd.DataFrame(all_data)

    def update_historic_file(self, histo_path, start_date, end_date):
        """
        Met à jour automatiquement le fichier historique avec les nouvelles données disponibles.
        - Ajoute les nouveaux jours manquants.
        - Remplace les jours existants si la donnée API est disponible.
        """
        # Charger l'historique existant ou créer un DataFrame vide
        if os.path.exists(histo_path):
            histo = pd.read_excel(histo_path)
            histo['date'] = pd.to_datetime(histo['date'])
        else:
            histo = pd.DataFrame(columns=['date', 'temperature_max', 'temperature_min', 'precipitation'])

        # Générer la liste de toutes les dates à couvrir
        all_dates = pd.date_range(start_date, end_date, freq="D")
        all_dates = pd.to_datetime(all_dates)
        # Identifier les dates manquantes ou à mettre à jour
        histo_dates = set(histo['date']) if not histo.empty else set()
        missing_or_update_dates = [d for d in all_dates if d not in histo_dates]

        if not missing_or_update_dates:
            print("Aucune nouvelle donnée à mettre à jour.")
            return histo

        print(f"Récupération de {len(missing_or_update_dates)} jours manquants ou à mettre à jour...")
        new_data = self.fetch_weather_data_optimized(missing_or_update_dates[0], missing_or_update_dates[-1])
        if new_data is not None and not new_data.empty:
            new_data['date'] = pd.to_datetime(new_data['date'])
            # Fusionner : on remplace les jours existants par les nouveaux
            histo = pd.concat([histo[~histo['date'].isin(new_data['date'])], new_data], ignore_index=True)
            histo = histo.sort_values('date').reset_index(drop=True)
            histo.to_excel(histo_path, index=False)
        return histo

# app/utils/week_utils.py

from datetime import timedelta
import pandas as pd

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
            'days_in_week': max(1, (first_week_end - first_week_start).days + 1)
        })

        # Semaines suivantes
        week_number += 1
        next_week_start = first_week_end + timedelta(days=1)
        while next_week_start <= effective_end:
            week_start = next_week_start
            week_end = week_start + timedelta(days=6)
            if week_end > effective_end:
                week_end = effective_end

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
