import sqlite3
from typing import List, Tuple
import os

class DatabaseManager:
    def __init__(self, db_path: str = "app/database/boutiques.db"):
        self.db_path = db_path

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def get_all_secteurs(self) -> List[Tuple]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM secteurs")
            return cursor.fetchall()

    def get_all_boutiques(self) -> List[Tuple]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM boutiques")
            return cursor.fetchall()

    def get_boutiques_by_secteur(self, secteur_id: int) -> List[Tuple]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM boutiques WHERE id_secteur = ?", (secteur_id,))
            return cursor.fetchall()

    def get_boutique_by_name(self, nom_boutique: str) -> Tuple:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM boutiques WHERE nom_boutique = ?", (nom_boutique,))
            return cursor.fetchone()

    def add_secteur(self, nom_secteur: str):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO secteurs (nom_secteur) VALUES (?)", (nom_secteur,))
            conn.commit()

    def delete_secteur(self, secteur_id: int):
        with self.get_connection() as conn:
            cur = conn.cursor()
            # Vérifier s'il y a des boutiques associées
            cur.execute("SELECT COUNT(*) FROM boutiques WHERE id_secteur = ?", (secteur_id,))
            count = cur.fetchone()[0]
            if count > 0:
                raise Exception("Impossible de supprimer ce secteur : des boutiques y sont encore rattachées.")
            cur.execute("DELETE FROM secteurs WHERE id_secteur = ?", (secteur_id,))
            conn.commit()
def get_all_boutiques(db_path: str | None = None) -> list[str]:
    """Retourne la liste des boutiques connues dans la base SQLite."""
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'database', 'boutiques.db')
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT nom_boutique FROM boutiques").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]
