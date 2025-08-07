import joblib
import pandas as pd
import numpy as np
import os
from config import BASE_DIR
from app.utils.data_loader import load_historical_data
from app.utils.exogenous    import exo_var

def load_model_and_scalers(cible):
    folder = os.path.join(BASE_DIR, 'models', f"{cible}_models")
    model = joblib.load(os.path.join(folder, f"sarimax_model_{cible}.pkl"))
    scaler_exog = joblib.load(os.path.join(folder, f"scaler_exog_{cible}.pkl"))
    scaler_target = joblib.load(os.path.join(folder, f"scaler_target_{cible}.pkl"))
    pca = joblib.load(os.path.join(folder, f"pca_{cible}.pkl"))
    return model, scaler_exog, scaler_target, pca

def in_sample_prediction(
    model,
    scaler_exog,
    pca,
    scaler_target,
    exog_hist: pd.DataFrame,
) -> pd.Series:
    """
    Retourne les prédictions *in‑sample* du modèle, **dé‑normalisées**,
    indexées exactement comme ``exog_hist``.
    """
    X_pca = pca.transform(
        scaler_exog.transform(exog_hist[scaler_exog.feature_names_in_])
    )
    y_pred_norm = model.get_prediction(exog=X_pca).predicted_mean
    y_pred_real = scaler_target.inverse_transform(
        y_pred_norm.to_numpy().reshape(-1, 1)
    ).ravel()

    # 🛠 Correction : aligner explicitement les longueurs
    n = min(len(y_pred_real), len(exog_hist.index))
        # === DEBUG ALIGNEMENT AVANT RETOUR ===
    idx_pred = pd.Index(range(len(y_pred_real)))
    idx_exog = exog_hist.index

    if len(y_pred_real) != len(idx_exog):
        print("[DEBUG ALIGNEMENT] Mismatch de taille :")
        print(f"  - Prédictions : {len(y_pred_real)}")
        print(f"  - Index exogènes : {len(idx_exog)}")

        # Si index est un DatetimeIndex, afficher les dates manquantes/supplémentaires
        if isinstance(idx_exog, pd.DatetimeIndex):
            print(f"  - Dates exogènes : {idx_exog.min()} à {idx_exog.max()} (total: {len(idx_exog)})")
        else:
            print(f"  - Index exogènes : {list(idx_exog)[:10]} ...")
        # Afficher quelques valeurs de y_pred_real si utile
        print(f"  - Valeurs y_pred_real (échantillon): {y_pred_real[:10]}")
    # Si les index sont DatetimeIndex, comparer les dates
    if hasattr(exog_hist, "index") and isinstance(exog_hist.index, pd.DatetimeIndex):
        if hasattr(exog_hist, "index"):
            dates_exog = set(exog_hist.index)
            n_pred = len(y_pred_real)
            n_exog = len(exog_hist.index)
            if n_pred != n_exog:
                print("[DEBUG] Index non alignés !")
                # Impossible de donner les "dates" côté y_pred_real si ce n'est pas un Series, mais on peut donner la longueur
                print(f"  - y_pred_real: {n_pred}, exogènes: {n_exog}")
                print(f"  - Dates exogènes: {sorted(dates_exog)[:5]} ... {sorted(dates_exog)[-5:]}")
                # Optionnel : Afficher des exemples de dates dans les historiques si dispo

    return pd.Series(y_pred_real[:n], index=exog_hist.index[:n], name="y_hat")


def compute_empirical_bounds(
    train_data: pd.Series,
    train_pred: pd.Series,
    alpha: float = 0.70,
    window_weeks: int = 104,
) -> tuple[float, float]:
    """
    Calcule un intervalle de confiance empirique *dans l’échelle d’origine*
    à partir des résidus des 2 dernières années (par défaut).

    • On ne normalise plus → largeur d’IC plus réaliste.  
    • Bornes = quantiles (1‑alpha)/2 et (1+alpha)/2 du jeu de résidus récent.
    """
    resid = (train_data.iloc[: len(train_pred)] - train_pred).dropna()
    if len(resid) < max(10, window_weeks):
        raise ValueError("Pas assez de résidus pour estimer l’IC.")

    recent = resid.tail(window_weeks)
    q_lo, q_hi = np.percentile(recent, [(1 - alpha) / 2 * 100,
                                        (1 + alpha) / 2 * 100])
    return q_lo, q_hi


def verify_completeness(df, columns):
    if df[columns].isnull().any().any():
        missing_values = df[df[columns].isnull().any(axis=1)]
        print("[WARNING] Valeurs manquantes détectées :")
        print(missing_values)
        raise ValueError("Des valeurs NaN sont présentes dans les colonnes critiques.")

def forecast_future(
    exog_future: pd.DataFrame,
    model,
    scaler_exog,
    scaler_target,
    pca,
    train_data: pd.Series | None = None,
    train_pred_mean: pd.Series | None = None,
    alpha: float = 0.70
) -> pd.DataFrame:
    print("\n=== [DEBUG] Début forecast_future ===")

    # --- 0. vérifs rapides ------------------------------------
    if "Date" not in exog_future.columns:
        raise ValueError("Colonne 'Date' absente de exog_future")
    dates = pd.to_datetime(exog_future["Date"])
    if dates.isnull().any():
        raise ValueError("Dates NaN dans exog_future")

    exog_df = exog_future.set_index(dates).copy()

    # --- 1. features  →  PCA ----------------------------------
    feat_cols    = [c for c in scaler_exog.feature_names_in_ if c in exog_df.columns]
    X_pca        = pca.transform(scaler_exog.transform(exog_df[feat_cols]))

    # index RangeIndex attendu par statsmodels
    start_idx = model.nobs
    end_idx   = start_idx + len(X_pca) - 1
    exog_pca  = pd.DataFrame(
        X_pca,
        index=pd.RangeIndex(start_idx, end_idx + 1),
        columns=[c for c in model.model.exog_names if c not in ("const", "intercept")]
    )

    # --- 2. prévision (échelle normalisée) --------------------
    y_pred_norm = model.predict(start=start_idx, end=end_idx, exog=exog_pca)

    # --- 3. dé‑normalisation tout de suite --------------------
    y_hat = scaler_target.inverse_transform(y_pred_norm.values.reshape(-1, 1)).ravel()
    y_hat = pd.Series(y_hat, index=dates, name="y_hat")

    # --- 4. bornes empiriques (échelle réelle) ----------------
    if train_data is not None and train_pred_mean is not None:
        low_d, up_d = compute_empirical_bounds(
            train_data.reset_index(drop=True),
            train_pred_mean.reset_index(drop=True),
            alpha=alpha
        )
        y_lower = y_hat + low_d
        y_upper = y_hat + up_d
    else:
        y_lower = pd.Series(np.nan, index=y_hat.index)
        y_upper = pd.Series(np.nan, index=y_hat.index)

    # --- 5. assemblage final ----------------------------------
    df_out = pd.DataFrame({
        "Date": dates.values,
        "Prévision":          y_hat.values,
        "Borne inférieure":   y_lower.values,
        "Borne supérieure":   y_upper.values,
    })

    print(f"[DEBUG] NaN par colonne :\n{df_out.isnull().sum()}")
    if df_out.isnull().any().any():
        raise ValueError("NaN détecté dans df_out — voir debug ci‑dessus")

    print("=== [DEBUG] Fin forecast_future ===\n")
    return df_out



def aggregate_weekly_forecast(
    forecast_weekly: pd.DataFrame,
    selected_days: pd.DatetimeIndex,
    column: str = 'Prévision'
) -> float:
    if forecast_weekly[column].isnull().any():
        raise ValueError("Des NaN sont présents dans les prévisions hebdomadaires.")

    weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1.2, 6: 0}
    total_weight = sum(weights.values())
    selected_set = set(selected_days.date)

    total_flux = 0.0
    for _, row in forecast_weekly.iterrows():
        week_start = pd.to_datetime(row['Date'])
        days = pd.date_range(start=week_start, periods=7)
        common_days = set(days.date).intersection(selected_set)
        if not common_days:
            continue
        daily_flux = row[column] / total_weight
        for day in common_days:
            total_flux += daily_flux * weights[day.weekday()]

    return total_flux

def auto_update_model_with_latest_data(cible, model, scaler_exog, scaler_target, pca):
    """
    Extend the SARIMAX model with any new weekly observations that have become 
    available since the model was last trained. This function appends new data 
    (with proper scaling and PCA transformation for exogenous variables) to the 
    model and saves the updated model.
    """
    # Load the full historical data to identify new observations
    y_hist, _, _, cal_df = load_historical_data(cible)
    total_obs = len(y_hist)
    old_nobs = model.nobs  # number of observations the model was originally trained on
    
    # If no new data, return the model unchanged
    if total_obs <= old_nobs:
        return model
    
    new_points_count = total_obs - old_nobs
    print(f"[DEBUG] New data detected: {new_points_count} new weeks will be appended to the model.")
    
    # Prepare new endogenous (target) data – scale it using the existing scaler_target
    new_endog = y_hist.iloc[old_nobs:]           # new target values (real scale)
    new_endog_norm = scaler_target.transform(new_endog.to_numpy().reshape(-1, 1)).ravel()
    # Create a series for new endogenous data with index continuing from old_nobs
    # Harmoniser le nom de la colonne avec l'endogène d'origine
    original_name = model.model.data.orig_endog.name if hasattr(model.model.data.orig_endog, 'name') else "y"

    new_endog_series = pd.Series(
        new_endog_norm,
        index=pd.RangeIndex(start=old_nobs, stop=total_obs),
        name=original_name
    )

    # Prepare new exogenous data for the corresponding new dates
    new_dates = cal_df["Date"].iloc[old_nobs:]   # dates for new observations
    start_new_date = new_dates.min()
    end_new_date   = new_dates.max()
    exo_new = exo_var(start_new_date, end_new_date)
    if exo_new.empty:
        # If we cannot retrieve exogenous data for the new period, skip the update
        print("[WARNING] No exogenous data for new dates; model update skipped.")
        return model
    # Align exogenous data to the weekly dates of new observations
    exo_new = exo_new.set_index("Date")
    new_exog_aligned = exo_new.loc[new_dates]
    
    # Ensure exogenous features match those used in training
    feature_cols = [c for c in scaler_exog.feature_names_in_ if c in new_exog_aligned.columns]
    missing_cols = [c for c in scaler_exog.feature_names_in_ if c not in new_exog_aligned.columns]
    if missing_cols:
        raise ValueError(f"Les variables exogènes manquantes pour les nouvelles données : {missing_cols}")
    
    # Apply scaling and PCA transformation to new exogenous features
    X_new_scaled = scaler_exog.transform(new_exog_aligned[feature_cols])
    X_new_pca = pca.transform(X_new_scaled)
    exog_expected = [c for c in model.model.exog_names if c not in ("const", "intercept")]
    if X_new_pca.shape[1] != len(exog_expected):
        raise ValueError("Mismatch in PCA output dimensions vs model exog features during update.")
    # Create DataFrame for new exogenous PCA data with proper index
    X_new_pca_df = pd.DataFrame(X_new_pca, 
                                index=pd.RangeIndex(start=old_nobs, stop=total_obs), 
                                columns=exog_expected)
    
    # Append new observations to the model without refitting parameters
    updated_model = model.append(new_endog_series, exog=X_new_pca_df, refit=False)
    print("[DEBUG] Model successfully extended with new data.")
    
    # Save the updated model back to disk (overwriting the old model file)
    model_path = os.path.join(BASE_DIR, 'models', f"{cible}_models", f"sarimax_model_{cible}.pkl")
    joblib.dump(updated_model, model_path)
    print(f"[DEBUG] Updated model saved to {model_path}")
    
    return updated_model
