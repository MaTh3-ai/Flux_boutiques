import streamlit as st
from datetime import datetime, timedelta
import pandas as pd

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

    # Assurer l'unicité des dates dans y_hist (sinon reindex échoue)
    if not y_hist.index.is_unique:
        duplicated = y_hist.index[y_hist.index.duplicated()].unique()
        st.error(f"Doublons détectés dans l'historique : {list(duplicated)}")
        print("[DEBUG] Dates dupliquées dans y_hist :", duplicated)
        # Si besoin, on garde le dernier enregistrement pour chaque date
        y_hist = y_hist[~y_hist.index.duplicated(keep='last')]

    # Calcul des dates lag avec calendrier custom
    dates_n1 = []
    dates_n2 = []
    for d in forecast_dates:
        # Recherche de la semaine correspondante dans le calendrier (custom !)
        if d in y_hist.index:
            year, week = d.isocalendar().year, d.isocalendar().week
            # Semaine N-1 et N-2 selon le calendrier
            prev1 = y_hist.index[(y_hist.index.year == year-1) & (y_hist.index.isocalendar().week == week)]
            prev2 = y_hist.index[(y_hist.index.year == year-2) & (y_hist.index.isocalendar().week == week)]
            # On prend le premier match ou NaT
            dates_n1.append(prev1[0] if len(prev1) else pd.NaT)
            dates_n2.append(prev2[0] if len(prev2) else pd.NaT)
        else:
            dates_n1.append(pd.NaT)
            dates_n2.append(pd.NaT)

    lag_table = pd.DataFrame({
        "Hist_N-1": y_hist.reindex(dates_n1).values,
        "Hist_N-2": y_hist.reindex(dates_n2).values
    }, index=forecast_dates)

    # DEBUG: Vérification finale
    if lag_table.isnull().any().any():
        missing = lag_table[lag_table.isnull().any(axis=1)]
        st.warning(f"Valeurs manquantes dans l'historique décalé pour ces dates : {missing.index.tolist()}")
        print("[DEBUG] Dates avec valeurs manquantes dans l'historique décalé :", missing.index.tolist())

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
