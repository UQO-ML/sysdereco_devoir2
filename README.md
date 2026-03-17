# INF6083 - Projet P2 : Recommandation basée sur le contenu

Projet 2 du cours *Systèmes de recommandation* (INF6083). Recommandation **basée sur le contenu** à partir des métadonnées des livres (Amazon Reviews 2023, catégorie Books), avec un **notebook** comme livrable principal.

---

## Jeu de données

- **Source :** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), catégorie **Books** (même sous-ensemble que le projet P1).
- **Interactions :** `Books.jsonl` — 29.5M reviews (user_id, parent_asin, rating, timestamp, text, …).
- **Métadonnées :** `meta_Books.jsonl` — 4.4M fiches produit (parent_asin, title, author, description, categories, features, details, price, …).
- Les sous-ensembles d'interactions P1 (active users / temporal) sont réutilisés tels quels pour P2 — aucun nouvel échantillonnage massif n'est effectué.

---

## Prérequis

- **Python** 3.10 à 3.13
- **OS :** Linux recommandé (testé sur Arch/CachyOS). Windows possible via WSL.
- **RAM :** 16 Go minimum (le pipeline gère la mémoire via `gc.collect()` et chargement par colonnes).
- **Disque :** ~15 Go pour les données brutes + fichiers intermédiaires.
- **Optionnel :** GPU NVIDIA avec CUDA 12 pour l'échantillonnage accéléré via RAPIDS (cuDF, RMM, CuPy).

---

## Installation

### 1. Environnement virtuel et dépendances

```bash
cd sysdereco_devoir2
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. *(Optionnel, GPU NVIDIA CUDA 12)* - Pour l’échantillonnage RAPIDS dans `scripts/precursor.py` :

```bash
pip install -r requirements-rapids.txt
```

---

## Structure du projet


```
data/raw/jsonl/
├── Books.jsonl          # Interactions (reviews)
└── meta_Books.jsonl     # Métadonnées produits
```

```
sysdereco_devoir2/
├── README.md
├── requirements.txt                            # Dépendances principales (pandas, pyarrow, scikit-learn, …)
├── requirements-rapids.txt                     # Optionnel : RAPIDS (cudf, rmm, cupy) pour GPU
├── main.py                                     # Orchestrateur : précurseur (P1) + jointure (P2)
├── notebook_recommandation_contenu.ipynb        # Notebook principal (livrable)
├── INF6083_projet_p2.pdf                       # Énoncé du projet
├── scripts/
│   ├── precursor.py                            # P1 : conversion JSONL→Parquet, échantillonnage,
│   │                                           #       nettoyage, filtrage itératif, split train/test
│   ├── joining.py                              # P2 : jointure interactions↔metadata, normalisation,
│   │                                           #       diagnostics qualité, production datasets finaux
│   └── check_env.py                            # Vérification de l'environnement (GPU, libs, …)
├── config/
│   └── default.yaml                            # Configuration (seuils, chemins, …)
├── data/
│   ├── raw/
│   │   ├── jsonl/                              # Sources brutes (Books.jsonl, meta_Books.jsonl)
│   │   └── parquet/                            # Conversion automatique (Books.parquet, meta_Books.parquet)
│   ├── processed/
│   │   ├── sample-active-users/                # Échantillon utilisateurs actifs
│   │   │   ├── active_users_original.parquet   #   Brut après échantillonnage
│   │   │   ├── active_users_cleaned.parquet    #   Après nettoyage
│   │   │   ├── active_users_filtered.parquet   #   Après filtrage itératif (k-core)
│   │   │   └── splits/                         #   train.parquet, test.parquet, matrices CSR
│   │   └── sample-temporal/                    # Échantillon temporel (même structure)
│   └── joining/                                # Datasets joints finaux (P2)
│       ├── active_pre_split_joined.parquet
│       ├── active_post_split_union_joined.parquet
│       ├── temporal_pre_split_joined.parquet
│       ├── temporal_post_split_union_joined.parquet
│       ├── active_pre_split/
│       │   ├── train_interactions.parquet            
│       │   └── test_interactions.parquet             
│       └── temporal_pre_split/
│           ├── train_interactions.parquet            
│           └── test_interactions.parquet             
├── results/
│   └── joining/
│       ├── joining_diagnostics.json            # Diagnostic complet (machine-readable)
│       └── joining_diagnostics.md              # Diagnostic lisible (sections A-G)
└── models/                                     # Modèles entraînés (Tâches 1-3)
```

**Remise :** renommer le notebook en `INF6083-P2-EquipeN-Code.ipynb` (N = numéro d’équipe) et l’inclure dans le zip avec le rapport PDF et le `README.md`.

---

## Exécution

1. Données : placer dans `data/raw/jsonl/` les fichiers sources (ex. `Books.jsonl`, JSONL/Parquet d’interactions). Ou
2. python main.py

## Pipeline d'exécution

Le pipeline complet est lancé via `python main.py`. Il enchaîne deux phases.
Si les fichiers intermédiaires existent déjà, chaque phase est automatiquement sautée.

---

### Phase 1 — Précurseur (`scripts/precursor.py`)

Réutilise le pipeline P1 si les fichiers existent déjà dans `data/processed/`, sinon exécute :

| Étape | Description |
|:-----:|-------------|
| 0 | Conversion JSONL → Parquet (Polars) |
| 1 | Échantillonnage utilisateurs actifs (≥ 20 reviews, 50k users) — GPU ou CPU |
| 2 | Échantillonnage temporel (2020-2023, users ≥ 20 reviews) — GPU ou CPU |
| 3 | Nettoyage (rating/timestamp invalides) |
| 4 | Filtrage itératif k-core (user ≥ 20, item ≥ 5) jusqu'à convergence |
| 5 | Split train/test (80/20) + matrices CSR + sauvegarde |

### Phase 2 — Jointure (`scripts/joining.py`)

Fusionne les interactions P1 avec les métadonnées `meta_Books`. Vérifié automatiquement via `data/joining/` :

| Étape | Description |
|:-----:|-------------|
| 1 | Chargement metadata (colonnes ciblées uniquement pour économiser la RAM) |
| 2 | Vérification schéma et clés (`parent_asin`) sur chaque source |
| 3 | Calcul qualité de jointure (couverture, metadata orphelines) |
| 4 | Identification des attributs exploitables + justifications |
| 5 | Rapport de valeurs manquantes (NaN + listes vides) — global et par sous-ensemble |
| 6 | Normalisation des colonnes complexes (structs → scalaires, listes → strings) |
| 7 | Imputation médiane pour `price` |
| 8 | Sauvegarde des datasets joints en Parquet + diagnostics JSON/MD |

---

## Colonnes des datasets joints

Chaque fichier `*_joined.parquet` dans `data/joining/` contient :

| Colonne | Source | Type final | Description |
|---------|--------|:----------:|-------------|
| `user_id` | interactions | string | Identifiant utilisateur anonymisé |
| `parent_asin` | interactions | string | Identifiant produit (clé de jointure) |
| `rating` | interactions | float | Note 1-5 attribuée par l'utilisateur |
| `timestamp` | interactions | int/datetime | Horodatage de la review |
| `title` | metadata | string | Titre du livre |
| `subtitle` | metadata | string | Sous-titre (format, édition) |
| `description` | metadata | string | Description éditoriale (éléments joints par ` \| `) |
| `categories` | metadata | string | Taxonomie Amazon (éléments joints par ` \| `) |
| `features` | metadata | string | Points clés marketing (éléments joints par ` \| `) |
| `author_name` | metadata | string | Nom de l'auteur (extrait du struct `author.name`) |
| `details_publisher` | metadata | string | Éditeur (extrait du struct `details.Publisher`) |
| `details_language` | metadata | string | Langue (extrait du struct `details.Language`) |
| `average_rating` | metadata | float | Note moyenne agrégée |
| `rating_number` | metadata | float | Nombre total de notes |
| `price` | metadata | float | Prix (manquants imputés par la médiane) |

---

## Diagnostics produits

Après exécution, `results/joining/joining_diagnostics.md` contient :

| Section | Contenu |
|:-------:|---------|
| **A** | Réutilisation du sous-ensemble P1 (note méthodologique) |
| **B** | Documentation des sources (chemins, format, shape, colonnes) |
| **C** | Vérifications schéma et clés (`parent_asin`) |
| **D** | Qualité de jointure (couverture, metadata orphelines, interprétation) |
| **E** | Attributs exploitables (retenus, ignorés, justifications) |
| **F** | Valeurs manquantes et stratégie (global meta + sous-ensemble joint, NaN + vides effectifs) |
| **G** | Jeux de données finaux (chemins, dimensions) |

Le fichier `results/joining/joining_diagnostics.json` contient les mêmes données en format machine-readable (exploitable depuis le notebook).

---

## Tâches (résumé)

| Tâche | Contenu |
|-------|--------|
| **0** | Préparation : jointure interactions/métadonnées, split train/test temporel, représentations d’items (TF-IDF), profils utilisateurs. |
| **1** | Recommandation par similarité de contenu (profil-item), top-N, analyse qualitative. |
| **2** | Représentations latentes (SVD tronquée), projection des profils, comparaison avec Tâche 1. |
| **3** | Apprentissage d’un score de pertinence (variables explicatives, au moins 2 modèles). |
| **4** | Évaluation : métriques top-N, analyse par profils d’utilisateurs, diversité et surspécialisation. |
| **5** | Discussion : synthèse, limites, cold start, hybridation avec P1. |

---

## Livrables (rappel)

- **Rapport PDF** : `INF6083-P2-EquipeN-Rapport.pdf` (max 30 pages).
- **Code** : `INF6083-P2-EquipeN-Code.ipynb` (ou fichiers .py), `requirements.txt`, `README.md`.
- Date de remise : **22 mars 2026**, 23h59.

---

## Licence

Voir le fichier `LICENSE` à la racine du projet.
