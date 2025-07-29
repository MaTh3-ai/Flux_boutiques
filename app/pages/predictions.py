import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from app.utils.exogenous      import exo_var
from app.utils.forecast       import (
    load_model_and_scalers, forecast_future,
    aggregate_weekly_forecast, in_sample_prediction, auto_update_model_with_latest_data
)
from app.utils.data_loader    import load_historical_data, build_weekly_lags
from app.utils.visualizations import plot_forecast, plot_historical_data
from config import HISTORICAL_FILE


def predictions_page() -> None:
    st.title("Prévisions hebdomadaires avec données historiques")

    today = datetime.today().date()
    default = (today - timedelta(weeks=4), today + timedelta(weeks=4))
    date_sel = st.date_input(
        "Plage de prévision", default,
        min_value=today - timedelta(weeks=52),
        max_value=today + timedelta(weeks=52),
        # locale="fr"  # <-- SUPPRIMER CETTE LIGNE
    )
    if not isinstance(date_sel, tuple) or len(date_sel) != 2:
        st.info("Choisissez d’abord une date de fin…")
        return
    start_date, end_date = date_sel
    if start_date >= end_date:
        st.error("La date de début doit précéder la date de fin.")
        return
    st.info(f"Période sélectionnée : **{start_date:%d/%m/%Y} → {end_date:%d/%m/%Y}**")

    # ───── Boutique ─────
    cible = st.session_state.get("CIBLE")
    if not cible:
        st.error("Veuillez d’abord sélectionner une boutique.")
        return

    if not st.button("Lancer la prévision 🚀"):
        return

    # 1. exogènes FUTURES -------------------------------------------------
    with st.spinner("Variables exogènes…"):
        exog_future = exo_var(start_date, end_date)
    if exog_future.empty:
        st.error("Exogènes vides.")
        return

    # 2. modèle + historiques --------------------------------------------
    with st.spinner("Modèle & historiques…"):
        model, scaler_exog, scaler_target, pca = load_model_and_scalers(cible)
        # — MISE A JOUR AUTO (nouvelle data ? modèle extendu auto) —
        model = auto_update_model_with_latest_data(cible, model, scaler_exog, scaler_target, pca)

        # série historique + grille calendrier
        y_hist, _, _, cal_df = load_historical_data(cible)

        # exogènes HISTORIQUES parfaitement alignées
        exog_hist = (exo_var(cal_df["Date"].min(), cal_df["Date"].max())
                     .set_index("Date")
                     .loc[cal_df["Date"]])

        # prédictions in‑sample (pour IC empirique)
        pred_hist = in_sample_prediction(model, scaler_exog, pca,
                                         scaler_target, exog_hist)

        # 3. prévision FUTURE ---------------------------------------------
        exog_future_full = (exog_future
                            .set_index("Date")[scaler_exog.feature_names_in_])
        forecast_df = forecast_future(
            exog_future_full.reset_index(),
            model, scaler_exog, scaler_target, pca,
            train_data=y_hist, train_pred_mean=pred_hist
        )

    # 4 ─ Ajout des colonnes historiques (lags) ──────────────────────────
    # Récupère les colonnes Hist_N-1 et Hist_N-2 alignées
    lag_cols = build_weekly_lags(cal_df, y_hist, lags=(1, 2))
    lag_future = lag_cols.iloc[-len(forecast_df):].reset_index(drop=True)
    # Supprime les colonnes déjà présentes pour éviter tout doublon
    forecast_df = pd.concat(
        [forecast_df.drop(columns=lag_future.columns, errors="ignore"), lag_future],
        axis=1
    )

    # 5 ─ Affichage (★ noms de colonnes ASCII simples) ───────────────────
    st.subheader("Tableau de prévision")
    st.dataframe(forecast_df)

    fig = plot_forecast(
        forecast_df,
        forecast_df["Hist_N-1"],
        forecast_df["Hist_N-2"],
        start_date.year
    )
    st.plotly_chart(fig, use_container_width=True)

    # Synthèse
    days = pd.date_range(start_date, end_date, freq="D")
    tot   = aggregate_weekly_forecast(forecast_df, days, "Prévision")
    low   = aggregate_weekly_forecast(forecast_df, days, "Borne inférieure")
    high  = aggregate_weekly_forecast(forecast_df, days, "Borne supérieure")
    st.metric("Flux total prédit", f"{tot:,.0f}",
              f"IC 70 % : {low:,.0f} → {high:,.0f}")

    # 6. Historique complet (expander) -----------------------------------
    with st.expander("Afficher l’historique complet"):
        try:
            hist_full = pd.read_excel(HISTORICAL_FILE)
            hist_full["Date"] = pd.to_datetime(
                hist_full.apply(
                    lambda r: f"{int(r.Annee)}-W{int(r.Semaine):02d}-1", axis=1
                ), format="%G-W%V-%u")
            hist_full = (hist_full.set_index("Date")
                         .drop(columns=["Annee", "Semaine"])
                         .sort_index())
            st.plotly_chart(plot_historical_data(hist_full, cible),
                            use_container_width=True)
        except Exception as e:
            st.error(f"Impossible d’afficher l’historique complet : {e}")

    # retour
    if st.button("← Retour à la sélection"):
        st.session_state.page = "selector"
        st.rerun()
