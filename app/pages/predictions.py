import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.utils.exogenous      import exo_var
from app.utils.forecast       import (
    load_model_and_scalers, forecast_future,
    aggregate_weekly_forecast, in_sample_prediction,
    auto_update_model_with_latest_data
)
from app.utils.data_loader    import load_historical_data
from app.utils.visualizations import plot_forecast, plot_historical_data
from config                   import HISTORICAL_FILE

def predictions_page() -> None:
    st.title("Prévisions hebdomadaires avec données historiques")

    # ───── Sélecteur de dates ────────────────────────────────────────────
    today      = datetime.today().date()
    default    = (today - timedelta(weeks=4), today + timedelta(weeks=4))
    date_sel   = st.date_input("Plage de prévision",
                               default,
                               min_value=today - timedelta(weeks=52),
                               max_value=today + timedelta(weeks=52))
    if not isinstance(date_sel, tuple) or len(date_sel) != 2:
        st.info("Choisissez la date de fin…")
        return
    start_date, end_date = date_sel
    if start_date >= end_date:
        st.error("La date de début doit précéder la date de fin.")
        return
    st.info(f"Période : **{start_date:%d/%m/%Y} → {end_date:%d/%m/%Y}**")

    # ───── Choix de la boutique ─────────────────────────────────────────
    cible = st.session_state.get("CIBLE")
    if not cible:
        st.error("Sélectionnez d’abord une boutique.")
        return

    if not st.button("Lancer la prévision 🚀"):
        return

    # ───── 1. Exogènes futures ──────────────────────────────────────────
    with st.spinner("Récupération des variables exogènes…"):
        exog_future = exo_var(start_date, end_date)
    if exog_future.empty:
        st.error("Impossible d’obtenir les exogènes.")
        return

    # ───── 2. Modèle + historiques ──────────────────────────────────────
    with st.spinner("Chargement du modèle et des historiques…"):
        model, scaler_exog, scaler_target, pca = load_model_and_scalers(cible)
        model = auto_update_model_with_latest_data(
            cible, model, scaler_exog, scaler_target, pca
        )

        y_hist, hist_n1, hist_n2, cal_df = load_historical_data(cible)

        # exogènes HISTORIQUES alignées sur le calendrier
        exog_hist = (exo_var(cal_df['Date'].min(), cal_df['Date'].max())
                    .set_index("Date")
                    .loc[cal_df['Date']])

        # Alignement défensif
        common_idx = y_hist.index.intersection(exog_hist.index)
        y_hist = y_hist.loc[common_idx]
        exog_hist = exog_hist.loc[common_idx]

        # Prédictions in-sample pour l'IC empirique
        pred_hist = in_sample_prediction(
            model, scaler_exog, pca, scaler_target, exog_hist
        )

        # ── 3. Prévision future ────────────────────────────────────
        exog_future_full = exog_future.set_index("Date")[scaler_exog.feature_names_in_]
        forecast_df = forecast_future(
            exog_future_full.reset_index(),
            model, scaler_exog, scaler_target, pca,
            train_data=y_hist, train_pred_mean=pred_hist
        )

    # ───── 4. Ajout des colonnes Hist_N‑1 / Hist_N‑2 ────────────────────
    forecast_dates = forecast_df["Date"]

    # On extrait Année/Semaine pour chaque date de prévision
    weeks = forecast_dates.apply(lambda d: (d.year, d.isocalendar().week))
    cal_df_hist = cal_df.set_index("Date")

    # Utilise la même logique de "lag recherche SEMAINE/ANNEE"
    def find_lag_value(year, week, lag):
        ref_year = year - lag
        match = cal_df_hist[(cal_df_hist["Annee"] == ref_year) & (cal_df_hist["Semaine"] == week)]
        if not match.empty:
            d = match.index[0]
            return y_hist.get(d, np.nan)
        return np.nan

    lag_n1 = [find_lag_value(y, w, 1) for y, w in weeks]
    lag_n2 = [find_lag_value(y, w, 2) for y, w in weeks]

    lag_table = pd.DataFrame({
        "Hist_N-1": lag_n1,
        "Hist_N-2": lag_n2
    }, index=forecast_dates)

    if lag_table.isnull().any().any():
        missing = lag_table[lag_table.isnull().any(axis=1)]
        st.warning(f"Valeurs manquantes dans l'historique décalé pour ces semaines : {missing.index.tolist()}")
        print("[DEBUG] Semaines avec lags manquants :", missing.index.tolist())

    forecast_df = pd.concat([forecast_df, lag_table.reset_index(drop=True)], axis=1)
    # ───── 5. Affichage tableau + courbe ────────────────────────────────
    st.subheader("Tableau de prévision")
    st.dataframe(forecast_df)

    fig = plot_forecast(
        forecast_df,
        forecast_df["Hist_N-1"],
        forecast_df["Hist_N-2"],
        start_date.year
    )
    st.plotly_chart(fig, use_container_width=True)


    # ───── 6. Synthèse numérique ────────────────────────────────────────
    days = pd.date_range(start_date, end_date, freq="D")
    tot  = aggregate_weekly_forecast(forecast_df, days, "Prévision")
    low  = aggregate_weekly_forecast(forecast_df, days, "Borne inférieure")
    high = aggregate_weekly_forecast(forecast_df, days, "Borne supérieure")
    st.metric("Flux total prédit",
              f"{tot:,.0f}",
              f"IC 70 % : {low:,.0f} → {high:,.0f}")

    # ───── 7. Historique complet (expander) ─────────────────────────────
    with st.expander("Afficher l’historique complet"):
        try:
            hist_full = pd.read_excel(HISTORICAL_FILE)
            hist_full["Date"] = pd.to_datetime(
                hist_full.apply(
                    lambda r: f"{int(r.Annee)}-W{int(r.Semaine):02d}-1", axis=1
                ), format="%G-W%V-%u"
            )
            hist_full = (hist_full.set_index("Date")
                         .drop(columns=["Annee", "Semaine"])
                         .sort_index())
            st.plotly_chart(
                plot_historical_data(hist_full, cible),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Impossible d’afficher l’historique complet : {e}")

    # ───── Retour sélection ────────────────────────────────────────────
    if st.button("← Retour à la sélection"):
        st.session_state.page = "selector"
        st.rerun()
