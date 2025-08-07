import time
import warnings
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import BASE_DIR, EXOG_FEATURES

warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

def print_index_debug(idx, label):
    print(f"[DEBUG] {label} - min: {idx.min()}, max: {idx.max()}, len: {len(idx)}")

def optimize_sarimax_model(train_data, train_exog, orders=None, cible=None, time_light=10):
    # Vérification / sélection explicite des colonnes exogènes
    missing = [c for c in EXOG_FEATURES if c not in train_exog.columns]
    if missing:
        raise ValueError(f"Colonnes exogènes manquantes dans train_exog : {missing}")
    exog_cols = EXOG_FEATURES.copy()
    target_col = getattr(train_data, 'name', None)
    if target_col in exog_cols:
        exog_cols.remove(target_col)

    # Préparation de X avec seulement les vraies exogènes
    X = train_exog[exog_cols]
    print(f"[DEBUG] Forme de X (données d'entraînement) : {X.shape}")  # Debug print

    # Entraînement du scaler_exog et PCA
    scaler_exog = StandardScaler().fit(X)
    print("[DEBUG] Exogènes scalées :", scaler_exog.feature_names_in_)
    X_scaled = scaler_exog.transform(X)
    print(f"[DEBUG] Forme de X_scaled après transformation : {X_scaled.shape}")  # Debug print

    pca = PCA(n_components=min(X_scaled.shape[1], 5)).fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # Entraînement du scaler_target
    y = train_data.values.ravel()
    scaler_target = StandardScaler().fit(y.reshape(-1, 1))
    y_scaled = scaler_target.transform(y.reshape(-1, 1)).flatten()

    # Construction des séries pour SARIMAX
    series_train = pd.Series(y_scaled).reset_index(drop=True)
    train_exog_pca = pd.DataFrame(X_pca).reset_index(drop=True)
    print(f"[DEBUG] series_train : index 0→{series_train.index.max()} / len={len(series_train)}")
    print(f"[DEBUG] exog_pca     : index 0→{train_exog_pca.index.max()} / len={len(train_exog_pca)}")

    # Le reste de votre fonction...
    if orders is None:
        orders = {
            'p': 1, 'd': 1, 'q': 1,
            'P': 1, 'D': 0, 'Q': 1, 's': 53
        }
    d = orders['d']
    D = orders['D']
    s = orders['s']
    TIME_LIGHT = int(time_light) * 60
    maxiter_light = 5
    tol_light = 1e-1
    maxiter_full = 100
    tol_full = 1e-5
    K_LIGHT = 5
    ESTIMATED_FIT_TIME = 60
    cache_results = {}
    best_trials = []
    start_time_light = time.time()
    search_space = [
        Integer(0, 4, name='p'),
        Integer(0, 3, name='q'),
        Integer(0, 3, name='P'),
        Integer(0, 2, name='Q')
    ]
    def format_order(order):
        return tuple(map(int, order))
    def compute_score_from_result(res):
        try:
            fitted_train = res.fittedvalues.copy()
            common_index = series_train.index.intersection(fitted_train.index)
            fitted_train_aligned = fitted_train.loc[common_index]
            true_train_aligned = series_train.loc[common_index]
            corr_train = np.corrcoef(fitted_train_aligned, true_train_aligned)[0, 1]
            return corr_train
        except Exception as e:
            st.error(f"Erreur lors du calcul des métriques : {e}")
            return -np.inf

    @use_named_args(search_space)
    def objective(p, q, P, Q):
        order = (int(p), int(q), int(P), int(Q))
        if (int(p) + orders['d'] + int(q) + int(P) + orders['D'] + int(Q)) > 9:
            return 1e6
        if order in cache_results:
            aic_cached = cache_results[order]['aic']
            corr_tr_cached = cache_results[order].get('corr_tr', np.nan)
            elapsed = int(time.time() - start_time_light)
            st.write(f"(cache) Bayes trial order={format_order(order)}, AIC={aic_cached:.2f}, corr_tr={corr_tr_cached:.3f}, elapsed={elapsed}s")
            return aic_cached
        try:
            print_index_debug(series_train.index, "series_train (objective)")
            print_index_debug(train_exog_pca.index, "train_exog_pca (objective)")
            model = sm.tsa.SARIMAX(
                series_train,
                exog=train_exog_pca,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            res = model.fit(disp=False, maxiter=maxiter_light, tol=tol_light)
            aic = res.aic
            fitted_train = res.fittedvalues
            common_idx = series_train.index.intersection(fitted_train.index)
            corr_tr = series_train.loc[common_idx].corr(fitted_train.loc[common_idx])
            cache_results[order] = {
                'aic': aic,
                'params': res.params.copy(),
                'corr_tr': corr_tr
            }
            best_trials.append((aic, corr_tr, order, res.params.copy()))
            elapsed = int(time.time() - start_time_light)
            st.write(f"Bayes trial #{len(best_trials)}: order={format_order(order)}, AIC={aic:.2f}, corr_tr={corr_tr:.3f}, elapsed={elapsed}s")
            return aic
        except Exception as e:
            st.error(f"Erreur dans objective pour {order}: {e}")
            return 1e6

    def pareto_frontier_multi(objectives):
        objectives = np.asarray(objectives)
        n = objectives.shape[0]
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_pareto[i] = False
                    break
        return is_pareto

    def complexite(order):
        p, q, P, Q = order
        return p + q + 2 * (P + Q)

    def select_model_with_pareto(finalists, alpha=1, beta=50, gamma=10, delta_weight=1, k=50):
        if not finalists:
            return None
        aics = np.array([ft[0] for ft in finalists])
        corrs = np.array([ft[1] for ft in finalists])
        orders = [ft[2] for ft in finalists]
        complexities = np.array([complexite(o) for o in orders])

        # Objectifs (AIC à minimiser, complexité à minimiser, corr à maximiser → 1-corr à minimiser)
        objs = np.vstack([aics, complexities, 1-corrs]).T
        is_pareto = pareto_frontier_multi(objs)
        pareto_indices = [i for i, ok in enumerate(is_pareto) if ok]
        if not pareto_indices:
            best_idx = np.argmin(aics)
            return orders[best_idx]

        # Critères sur le front de Pareto
        pareto_aics = aics[pareto_indices]
        pareto_comps = complexities[pareto_indices]
        pareto_corrs = corrs[pareto_indices]
        pareto_objs = objs[pareto_indices]

        # Distance à la droite AIC = k × Complexité
        penalty = np.sqrt((pareto_aics - k * pareto_comps) ** 2)

        # Score pondéré
        scores = (
            alpha * pareto_aics +
            beta  * pareto_comps +
            gamma * pareto_corrs +     # corr_tr à maximiser (positif = meilleure corrélation)
            delta_weight * penalty
        )
        winner_on_front = np.argmin(scores)
        idx = pareto_indices[winner_on_front]
        return orders[idx]



    def get_initial_points(dimensions, n_points):
        bounds = [(dim.low, dim.high) for dim in dimensions]
        sampler = qmc.LatinHypercube(d=len(dimensions))
        sample = sampler.random(n=n_points)
        for i, (low, high) in enumerate(bounds):
            sample[:, i] = (sample[:, i] * (high - low + 1) + low).astype(int)
        return sample.tolist()

    st.subheader(f"Optimisation du modèle pour {cible if cible else '[cible non précisée]'}")
    approx_max = int(TIME_LIGHT / ESTIMATED_FIT_TIME)
    n_calls = min(30, approx_max) if approx_max > 0 else 1
    st.write(f"Lancement optimisation bayésienne (~{n_calls} appels) sur {TIME_LIGHT//60} min")
    acq_funcs = ["EI", "LCB", "PI"]
    acq_func = np.random.choice(acq_funcs)
    n_initial_points = min(8, n_calls//2 if n_calls >= 2 else 1)
    initial_points = get_initial_points(search_space, n_initial_points)

    with st.spinner("Optimisation bayésienne en cours..."):
        res_gp = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            acq_func=acq_func,
            random_state=42,
            x0=initial_points,
            n_initial_points=0
        )

    best_trials.sort(key=lambda x: x[0])
    finalists = best_trials[:K_LIGHT]
    st.write("Finalistes (AIC asc):")
    for aic, corr_tr, order, _ in finalists:
        st.write(f"  Order={format_order(order)}, AIC={aic:.2f}, corr_tr={corr_tr:.3f}, complexité={complexite(order)}")

    best_order = select_model_with_pareto(finalists)
    if best_order is None:
        st.error("Aucun finaliste valide pour Pareto.")
        return None, None, scaler_exog, pca, scaler_target

    st.write(f"Ordre choisi parmi les finalistes (Pareto AIC vs complexité vs corr_tr): {format_order(best_order)} (complexité={complexite(best_order)})")
    p, q, P, Q = best_order
    try:
        with st.spinner("Entraînement final du modèle..."):
            print_index_debug(series_train.index, "series_train (fit final)")
            print_index_debug(train_exog_pca.index, "train_exog_pca (fit final)")
            model = sm.tsa.SARIMAX(
                series_train,
                exog=train_exog_pca,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            hyperparams = {
                'maxiter': maxiter_full,
                'tol': tol_full,
                'method': np.random.choice(['bfgs', 'nm', 'cg'])
            }
            start_params = cache_results.get(best_order, {}).get('params', None)
            if start_params is not None:
                res = model.fit(start_params=start_params, disp=False, **hyperparams)
            else:
                res = model.fit(disp=False, **hyperparams)
            corr_tr_f = compute_score_from_result(res)
            st.success(f"Full fit terminé: AIC={res.aic:.2f}, corr_tr={corr_tr_f:.3f}")
            return res, best_order, scaler_exog, pca, scaler_target, res.aic
    except Exception as e:
        st.error(f"Échec full fit pour {format_order(best_order)}: {e}")
        return None, best_order, scaler_exog, pca, scaler_target


def save_model(model_fit, scaler_exog, pca, scaler_target, cible):
    """
    Sauvegarde le modèle et les transformateurs dans le dossier approprié.
    :param model_fit: Modèle SARIMAX entraîné
    :param scaler_exog: Scaler pour les variables exogènes
    :param pca: Composant PCA pour la réduction de dimension
    :param scaler_target: Scaler pour la variable cible
    :param cible: Nom de la cible (boutique)
    """
    if not cible:
        raise ValueError("Le nom de la cible (boutique) doit être fourni à save_model.")
    target_folder = os.path.join(BASE_DIR, 'models', f"{cible}_models")
    os.makedirs(target_folder, exist_ok=True)
    joblib.dump(model_fit, os.path.join(target_folder, f"sarimax_model_{cible}.pkl"))
    joblib.dump(scaler_exog, os.path.join(target_folder, f"scaler_exog_{cible}.pkl"))
    joblib.dump(pca, os.path.join(target_folder, f"pca_{cible}.pkl"))
    joblib.dump(scaler_target, os.path.join(target_folder, f"scaler_target_{cible}.pkl"))
    st.success(f"Modèle et transformateurs sauvegardés pour la cible : {cible}")
