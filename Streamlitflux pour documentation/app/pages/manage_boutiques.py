import streamlit as st
import pandas as pd
import os
import shutil
from config import get_model_paths
from app.database.database_manager import DatabaseManager  # adapte le chemin si besoin
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'database', 'boutiques.db')

db = DatabaseManager(db_path)

# --- Gestion des secteurs (méthodes à ajouter dans DatabaseManager) ---
def add_secteur(nom_secteur):
    db.add_secteur(nom_secteur)

def delete_secteur(secteur_id):
    try:
        db.delete_secteur(secteur_id)
        return True, ""
    except Exception as e:
        return False, str(e)

# --- Gestion des boutiques ---
def add_boutique(nom_boutique, secteur_id):
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO boutiques (nom_boutique, id_secteur) VALUES (?, ?)", (nom_boutique, secteur_id))
        conn.commit()

def delete_boutique(nom_boutique):
    # Supprimer la boutique de la base
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM boutiques WHERE nom_boutique = ?", (nom_boutique,))
        conn.commit()
    # Supprimer le dossier de modèles associé
    model_dir = get_model_paths(nom_boutique)["MODEL_PATH"]
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            st.info(f"Dossier de modèles supprimé : {model_dir}")
        except Exception as e:
            st.warning(f"Erreur lors de la suppression du dossier de modèles : {e}")

def manage_boutiques_page():
    st.title("Gestion des boutiques et des secteurs")

    # --- Ajout d'un secteur ---
    st.header("Ajouter un secteur")
    nom_secteur = st.text_input("Nom du nouveau secteur")
    if st.button("Ajouter le secteur"):
        if nom_secteur.strip() == "":
            st.error("Le nom du secteur ne peut pas être vide.")
        else:
            try:
                add_secteur(nom_secteur.strip())
                st.success(f"Secteur '{nom_secteur}' ajouté.")
            except Exception as e:
                st.error(f"Erreur lors de l'ajout : {e}")

    st.markdown("---")

    # --- Suppression d'un secteur ---
    st.header("Supprimer un secteur")
    secteurs = db.get_all_secteurs()
    secteurs_df = pd.DataFrame(secteurs, columns=["id_secteur", "nom_secteur"]).drop_duplicates("nom_secteur")
    if len(secteurs_df) == 0:
        st.info("Aucun secteur à supprimer.")
    else:
        secteur_nom = st.selectbox("Secteur à supprimer", secteurs_df['nom_secteur'])
        secteur_id = secteurs_df[secteurs_df['nom_secteur'] == secteur_nom]['id_secteur'].iloc[0]
        confirm_sec = st.checkbox(f"Je confirme la suppression du secteur '{secteur_nom}'.")
        if st.button("Supprimer le secteur"):
            if confirm_sec:
                ok, msg = delete_secteur(secteur_id)
                if ok:
                    st.success(f"Secteur '{secteur_nom}' supprimé.")
                else:
                    st.error(msg)
            else:
                st.warning("Merci de cocher la case de confirmation avant de supprimer.")

    st.markdown("---")

    # --- Ajout d'une boutique ---
    st.header("Ajouter une boutique")
    secteurs_df = pd.DataFrame(db.get_all_secteurs(), columns=["id_secteur", "nom_secteur"]).drop_duplicates("nom_secteur")
    nom_boutique = st.text_input("Nom de la nouvelle boutique")
    secteur_nom = st.selectbox("Secteur", secteurs_df['nom_secteur'], key="secteur_boutique")
    secteur_id = secteurs_df[secteurs_df['nom_secteur'] == secteur_nom]['id_secteur'].iloc[0]

    if st.button("Ajouter la boutique"):
        if nom_boutique.strip() == "":
            st.error("Le nom de la boutique ne peut pas être vide.")
        else:
            add_boutique(nom_boutique.strip(), secteur_id)
            st.success(f"Boutique '{nom_boutique}' ajoutée au secteur '{secteur_nom}'.")

    st.markdown("---")

    # --- Suppression d'une boutique ---
    st.header("Supprimer une boutique")
    with db.get_connection() as conn:
        boutiques = pd.read_sql("SELECT * FROM boutiques", conn)
    if len(boutiques) == 0:
        st.info("Aucune boutique à supprimer.")
    else:
        boutique_nom = st.selectbox("Boutique à supprimer", boutiques['nom_boutique'])
        confirm = st.checkbox(f"Je confirme la suppression de la boutique '{boutique_nom}' et de ses modèles associés.")
        if st.button("Supprimer la boutique"):
            if confirm:
                delete_boutique(boutique_nom)
                st.success(f"Boutique '{boutique_nom}' et ses modèles associés supprimés.")
            else:
                st.warning("Merci de cocher la case de confirmation avant de supprimer.")

    st.markdown("---")

    # --- Liste des secteurs ---
    st.header("Liste des secteurs")
    secteurs_df = pd.DataFrame(db.get_all_secteurs(), columns=["id_secteur", "nom_secteur"]).drop_duplicates("nom_secteur")
    st.dataframe(secteurs_df)

    # --- Liste des boutiques actuelles ---
    st.header("Liste des boutiques")
    st.dataframe(
        boutiques[['nom_boutique', 'id_secteur']].merge(
            secteurs_df, on='id_secteur'
        )[['nom_boutique', 'nom_secteur']]
    )

    if st.button("← Retour à la sélection"):
        st.session_state.page = 'selector'
        st.rerun()
