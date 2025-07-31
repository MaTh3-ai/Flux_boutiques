import streamlit as st
import pandas as pd
from app.utils.data_loader import load_historical_data
from app.utils.exogenous import exo_var  # Ajoute cet import !
from config import get_model_paths, BASE_DIR, EXOG_FEATURES  # Ajoute EXOG_FEATURES si tu l'as défini
from app.utils.model_optimiser import optimize_sarimax_model, save_model

def update_model_page():
    st.title("Mise à jour du modèle SARIMAX")

    cible = st.session_state['CIBLE']
    st.info(f"Boutique sélectionnée : **{cible}**")

    with st.spinner("Chargement des données historiques et exogènes..."):
        y, hist_n1, hist_n2, cal_df = load_historical_data( cible)
        y = y.to_frame(name=cible)  # Pour compatibilité SARIMAX

        # Ajout : récupération des exogènes historiques alignées sur la même grille
        exo_hist = exo_var(cal_df['Date'].min(), cal_df['Date'].max())
        # Merge sur Date (clé d'alignement custom)
        X = cal_df.merge(exo_hist, on='Date', how='left', suffixes=('', '_exo'))

        # Vérification d'alignement
        if not y.index.equals(X.index):
            st.warning("Attention : les index de la cible et des exogènes ne sont pas parfaitement alignés. Correction automatique appliquée.")
            X = X.reindex(y.index)
            X = X.fillna(method='ffill')

        # Vérification de la présence des exogènes attendues
        missing = [c for c in EXOG_FEATURES if c not in X.columns]
        if missing:
            st.error(f"Colonnes exogènes manquantes dans X : {missing}")
            st.stop()
        if X[EXOG_FEATURES].isnull().any().any():
            st.error("Des NaN dans les exogènes historiques après merge.")
            st.stop()

    st.subheader("Temps alloué à la recherche bayésienne")
    time_light = st.number_input("Temps alloué à la recherche (minutes)", min_value=1, max_value=60, value=10)

    if st.button("Entraîner et sauvegarder le modèle"):
        with st.spinner("Optimisation et entraînement du modèle..."):
            model_fit, best_order, scaler_exog, pca, scaler_target, aic = optimize_sarimax_model(
                y.squeeze(), X, time_light=time_light, cible=cible
            )
        if model_fit is not None:
            save_model(model_fit, scaler_exog, pca, scaler_target, cible)
            if aic < 600:
                st.success(f"Modèle sauvegardé avec succès !\nAIC = {aic:.2f} : Bon modèle")
            elif aic < 700:
                st.warning(f"Modèle sauvegardé avec succès !\nAIC = {aic:.2f} : Modèle moyen")
            else:
                st.error(f"Modèle sauvegardé avec succès !\nAIC = {aic:.2f} : Modèle mauvais")
        else:
            st.error("Erreur lors de l'entraînement du modèle.")

    if st.button("← Retour à la sélection"):
        st.session_state.page = 'selector'
        st.rerun()
