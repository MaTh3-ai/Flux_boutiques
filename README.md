
---
# Flux Boutiques – Application de Prévisions et d’Analyse

## Présentation

**Flux Boutiques** est une application de prévisions hebdomadaires développée avec Streamlit, conçue pour analyser et anticiper l’activité des boutiques à partir de données historiques enrichies de variables exogènes (météo, jours fériés, etc.).
L’application intègre une base de données SQLite, des modèles statistiques avancés (SARIMAX, PCA…), et permet un traitement semi-automatisé des flux.

## Fonctionnalités principales

* Chargement et visualisation des historiques (ventes, météo…)
* Prévisions hebdomadaires avec graphiques interactifs
* Actualisation automatique ou manuelle des modèles et historiques
* Interface utilisateur claire, orientée métier

## Prérequis techniques

* Python 3.10 ou supérieur
* Installation des dépendances :

  ```
  pip install -r requirements.txt
  ```
* Accès au fichier de données principal : **Flux\_brut.xlsx**

## Procédure d’utilisation

### 1. Mise à jour manuelle des données

* **Obligation :**

  * Il est **impératif** de mettre à jour à la main le fichier `Flux_brut.xlsx` avec les nouvelles données avant toute exécution de l’application ou recalcul des modèles.

* **Attention :**

  * **Tous les fichiers Excel liés à l’application doivent être fermés** sur votre poste avant de lancer l’application ou toute opération de mise à jour.
    Si un fichier est ouvert, la mise à jour échouera ou pourra corrompre les données.

### 2. Lancement de l’application

* Exécuter la commande suivante dans le répertoire racine du projet :

  ```
  streamlit run app.py
  ```
* Suivre les instructions affichées dans l’interface pour paramétrer et lancer les prévisions.

### 3. Mise à jour des historiques et des modèles

* Utiliser la page dédiée dans l’application pour actualiser l’historique ou les modèles.
* **Toujours s’assurer qu’aucun fichier Excel n’est ouvert** avant d’effectuer cette opération.

## Recommandations et bonnes pratiques

* Ne jamais inclure dans l’archive ou le partage :

  * La base de données `boutiques.db`
  * Les fichiers Excel générés automatiquement par l’application
* Sauvegarder régulièrement une copie du fichier `Flux_brut.xlsx` après chaque ajout ou modification.
* Compléter le fichier `requirements.txt` si de nouveaux modules Python sont ajoutés au projet.
* Adapter les chemins d’accès dans `config.py` en fonction de l’emplacement réel des fichiers sur votre poste.

## Architecture simplifiée du projet

```
Streamlitflux pour documentation/
├── app/
│   ├── pages/
│   ├── utils/
│   ├── database/
│   └── ...
├── config.py
├── requirements.txt
├── Flux_brut.xlsx    (à mettre à jour manuellement)
└── README.md
```

---

**RAPPELS ESSENTIELS**

* Mettre à jour manuellement le fichier de données **avant toute utilisation** de l’application.
* **Fermer systématiquement tous les fichiers Excel** avant toute opération dans l’application.

---

Si tu veux une version avec des exemples concrets (exemple de structure de `Flux_brut.xlsx`, consignes pour la mise à jour, etc.), je peux l’ajouter sur demande.
