import streamlit as st
from app.utils.data_loader import  load_historical_data
from app.database.database_manager import get_all_boutiques
from app.utils.exogenous import exo_var
from app.utils.model_optimiser import optimize_sarimax_model, save_model
from config import EXOG_FEATURES  # Nécessaire pour vérification des colonnes

def update_all_models_page():
    st.title("Mise à jour globale des modèles boutiques")
    boutiques = get_all_boutiques()
    st.write(f"{len(boutiques)} boutiques détectées.")

    # Paramètre unique pour toutes les boutiques
    time_light = st.number_input("Temps alloué à la recherche (minutes)", min_value=1, max_value=60, value=10)

    if 'update_logs' not in st.session_state:
        st.session_state.update_logs = []

    if st.button("Mettre à jour toutes les boutiques 🚀"):
        st.session_state.update_logs = []  # Reset logs
        progress_bar = st.progress(0)
        total = len(boutiques)
        for i, cible in enumerate(boutiques):
            log = f"**Boutique : {cible}**\n"
            try:
                # Données historiques
                y, _, _, cal_df = load_historical_data(cible)
                y = y.to_frame(name=cible)

                # Exogènes sur la même période
                exog = exo_var(cal_df["Date"].min(), cal_df["Date"].max())
                X = cal_df.merge(exog, on="Date", how="left")

                # Vérification alignement Date
                dates_y = cal_df["Date"]
                dates_x = X["Date"] if "Date" in X.columns else X.index

                if not dates_y.equals(dates_x):
                    log += "⚠️ Dates de la cible et des exogènes non alignées. Correction appliquée.\n"
                    X = X.set_index("Date").reindex(dates_y).reset_index()
                    X = X.ffill().bfill()
                    missing_dates = dates_y[~dates_y.isin(dates_x)]
                    if not missing_dates.empty:
                        log += f"  - Dates manquantes exogènes corrigées : {missing_dates.tolist()}\n"

                # Vérification des colonnes exogènes requises
                missing = [c for c in EXOG_FEATURES if c not in X.columns]
                if missing:
                    log += f"❌ Colonnes exogènes manquantes : {missing}\n"
                    st.session_state.update_logs.append(log)
                    continue
                if X[EXOG_FEATURES].isnull().any().any():
                    log += "❌ Des NaN dans les exogènes après merge (malgré correction ffill/bfill).\n"
                    st.session_state.update_logs.append(log)
                    continue

                # Optimisation/Entraînement
                model_fit, best_order, scaler_exog, pca, scaler_target, aic = optimize_sarimax_model(
                    y.squeeze(), X, time_light=time_light, cible=cible
                )
                if model_fit is not None:
                    save_model(model_fit, scaler_exog, pca, scaler_target, cible)
                    if aic < 600:
                        log += f"✅ Modèle sauvegardé - Bon (AIC={aic:.2f})"
                    elif aic < 700:
                        log += f"🟠 Modèle sauvegardé - Moyen (AIC={aic:.2f})"
                    else:
                        log += f"🔴 Modèle sauvegardé - Mauvais (AIC={aic:.2f})"
                else:
                    log += "❌ Erreur lors de l'entraînement."

            except Exception as e:
                log += f"❌ Exception : {e}"
            st.session_state.update_logs.append(log)
            progress_bar.progress((i + 1) / total)
            st.markdown("\n\n".join(st.session_state.update_logs))
        st.success("Mise à jour terminée pour toutes les boutiques.")

    # Affichage des logs même après la boucle
    if st.session_state.get('update_logs'):
        st.markdown("---")
        st.subheader("Suivi de la mise à jour")
        st.markdown("\n\n".join(st.session_state.update_logs))

    if st.button("← Retour à la sélection"):
        st.session_state.page = 'selector'
        st.rerun()
