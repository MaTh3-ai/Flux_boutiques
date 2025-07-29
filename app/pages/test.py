import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from app.utils.exogenous import exo_var
from app.utils.forecast import (
    load_model_and_scalers,
    forecast_future,
    weekly_forecast_to_selected_days_sum,
    sum_selected_days
)
from app.utils.data_loader import load_historical_data, week_to_date
from app.utils.data_loader import week_to_custom_date
from app.utils.visualizations import plot_forecast, plot_historical_data
from config import HISTORICAL_FILE
import numpy as np

def test():
    st.title("Page de Test")
    st.write("Ceci est une page de test pour vérifier le fonctionnement des imports et des fonctions.")

    # Exemple d'utilisation de week_to_date
    test_date = pd.DataFrame({
        'Annee': [2024],
        'Semaine': [1]
    })
    test_date['Date'] = test_date.apply(week_to_date, axis=1)
    st.write("Date de début de semaine 1 de 2024 :", test_date['Date'].iloc[0])

    # Exemple d'utilisation de la fonction sum_selected_days
    test_weeks = pd.DataFrame({
        'Annee': [2024, 2024, 2024, 2024],
        'Semaine': [1, 2, 3, 4]
    })
    test_weeks['Date'] = test_weeks.apply(week_to_date, axis=1)
    st.write("Test de la fonction week_to_date sur plusieurs semaines :")
    st.write(test_weeks)

    # Test de la fonction exo_var
    start_date = "2025-01-01"
    end_date = "2025-03-31"
    st.write("Test de la fonction exo_var :")
    df_test = exo_var(start_date, end_date)
    st.write(df_test.head(20))

    # Test de la fonction forecast_future
    if st.button("Tester la prévision"):
        try:
            # Charger les données historiques
            y_hist, _, _, _ = load_historical_data(None, "test_cible")

            # Charger le modèle et les scalers
            model, scaler_exog, scaler_target, pca = load_model_and_scalers("test_cible")

            # Générer les variables exogènes futures
            exog_future = exo_var(start_date, end_date)

            # Faire une prévision
            forecast_df = forecast_future(exog_future, model, scaler_exog, scaler_target, pca)

            # Afficher les résultats
            st.write("Résultats de la prévision :")
            st.write(forecast_df.head(20))

            # Visualisation
            st.write("Visualisation des prévisions :")
            fig = plot_forecast(forecast_df, None, None, 2025)
            st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")

