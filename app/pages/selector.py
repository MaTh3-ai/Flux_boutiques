import streamlit as st
import sqlite3
import pandas as pd
import os
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'boutiques.db')


def load_data():
    conn = sqlite3.connect(db_path)
    secteurs = pd.read_sql("SELECT * FROM secteurs", conn)
    boutiques = pd.read_sql("SELECT * FROM boutiques", conn)
    conn.close()
    return secteurs, boutiques



def selector_page():
    st.title("Sélection Secteur et Boutique")
    secteurs, boutiques = load_data()
    secteur = st.selectbox(
        "Sélectionnez un secteur",
        options=secteurs['nom_secteur'].unique()
    )
    boutiques_secteur = boutiques[
        boutiques['id_secteur'] == secteurs[
            secteurs['nom_secteur'] == secteur
        ]['id_secteur'].iloc[0]
    ]
    boutique = st.selectbox(
        "Sélectionnez une boutique",
        options=boutiques_secteur['nom_boutique']
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Faire une prévision"):
            st.session_state['CIBLE'] = boutique
            st.session_state.page = 'prediction'
            st.rerun()
    with col2:
        if st.button("Mettre à jour le modèle"):
            st.session_state['CIBLE'] = boutique
            st.session_state.page = 'update_model'
            st.rerun()
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Prévision globale"):
            pass
    with col4:
        if st.button("Mise à jour globale"):
            st.session_state.page = 'update_all_models'
            st.rerun()
    st.markdown("---")
    # Ajout du bouton de gestion des boutiques
    if st.button("Gérer les boutiques"):
        st.session_state.page = 'manage_boutiques'
        st.rerun()

    st.markdown("---")
    if st.button("Mettre à jour les fichiers historiques"):
        with st.spinner("Mise à jour des fichiers historiques en cours…"):
            from app.utils.aggregation_fichier_primaire import update_all_historicals
            update_all_historicals()
        st.success("✅ Données mises à jour avec succès.")

