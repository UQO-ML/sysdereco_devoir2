"""
Module : loader
Rôle : Chargement et prétraitement des données pour le système de recommandation.

Structure prévue :
--------------------
1. Imports (pandas, numpy, chemins, config YAML).
2. Constantes / chemins par défaut (répertoire data, noms de colonnes attendus).
3. load_raw(path) : charge le jeu de données brut (CSV/JSON) depuis path.
4. preprocess(df) : nettoie et normalise les données (valeurs manquantes, types, encodage).
5. get_item_features(df) : construit la matrice ou le tableau des caractéristiques des items.
6. get_user_interactions(df) : construit les interactions utilisateur–item (ratings, clics, etc.).
7. load_config(config_path) : charge la config (ex. default.yaml) pour chemins et options.
8. Point d’entrée cli ou appel depuis run_pipeline : charger config → load_raw → preprocess → retourner données prêtes.
"""

# TODO: Implémenter les fonctions selon le format de données du projet (voir énoncé).
