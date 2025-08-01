"""
import sqlite3
import os

if os.path.exists('app/database/boutiques.db'):
    os.remove('app/database/boutiques.db')


def init_database():
    # Vérifier si le dossier app/database existe, sinon le créer
    if not os.path.exists('app/database'):
        os.makedirs('app/database')
    
    # Connexion à la base de données (la crée si elle n'existe pas)
    conn = sqlite3.connect('app/database/boutiques.db')
    cursor = conn.cursor()
    
    # Création de la table secteurs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS secteurs (
        id_secteur INTEGER PRIMARY KEY AUTOINCREMENT,
        nom_secteur TEXT NOT NULL
    )
    """)

    # Création de la table boutiques
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS boutiques (
        id_boutique INTEGER PRIMARY KEY AUTOINCREMENT,
        nom_boutique TEXT NOT NULL,
        id_secteur INTEGER NOT NULL,
        FOREIGN KEY (id_secteur) REFERENCES secteurs(id_secteur)
    )
    """)

    # Données des secteurs
    secteurs = [
        ("CENTRE AQUITAINE",),
        ("NORD AQUITAINE",),
        ("SUD AQUITAINE",)
    ]

    # Données des boutiques avec leur secteur
    boutiques = [
        # CENTRE AQUITAINE (id_secteur = 1)
        ("BORDEAUX INTENDANCE", 1),
        ("BORDEAUX SAINTE CATHERINE", 1),
        ("LA TESTE", 1),
        ("LANGON", 1),
        ("LIBOURNE", 1),
        ("MERIGNAC", 1),
        ("BEGLES RIVES D' ARCINS", 1),

        # NORD AQUITAINE (id_secteur = 2)
        ("ANGOULEME CASINO", 2),
        ("LA ROCHELLE BEAULIEU", 2),
        ("LIMOGES CLOCHER", 2),
        ("POITIERS CASINO", 2),
        ("ROYAN", 2),
        ("SAINTES", 2),
        
        # SUD AQUITAINE (id_secteur = 3)
        ("ANGLET BAB 2", 3),
        ("BERGERAC LA CAVAILLE", 3),
        ("MONT DE MARSAN", 3),
        ("PAU LESCAR", 3),
        ("SAINT PAUL LES DAX", 3),
        ("TRELISSAC LA FEUILLERAIE", 3)
    ]

    # Insertion des données
    cursor.executemany("INSERT OR IGNORE INTO secteurs (nom_secteur) VALUES (?)", secteurs)
    cursor.executemany("INSERT OR IGNORE INTO boutiques (nom_boutique, id_secteur) VALUES (?, ?)", boutiques)

    # Commit et fermeture
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_database()
    print("Base de données initialisée avec succès!")
"""
