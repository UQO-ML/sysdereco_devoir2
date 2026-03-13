# INF6083 — Projet P2 : Recommandation basée sur le contenu

Projet 2 du cours *Systèmes de recommandation* (INF6083). Recommandation **basée sur le contenu** à partir des métadonnées des livres (Amazon Reviews 2023, catégorie Books), avec un **notebook** comme livrable principal.

---

## Jeu de données

- **Source :** Amazon Reviews 2023, catégorie **Books** (même sous-ensemble que le projet P1).
- **Fichier central :** `meta_Books.jsonl.gz` (métadonnées : `parent_asin`, `title`, `description`, `categories`, `average_rating`, `rating_number`, `price`).
- Les **interactions** utilisateur–item proviennent du sous-ensemble de travail P1 (à placer dans `data/` ou à indiquer en début de notebook).

---

## Prérequis

- **Python** 3.10+ max 3.13
- **Environnement** : venv recommandé (Linux / Windows).
- **Optionnel** : CUDA pour accélération GPU (Cupy, CuDl, Rmm, PyTorch).

---

## Installation

**Ordre recommandé :**

1. Créer et activer le venv, puis installer les dépendances principales :

```bash
cd sysdereco_devoir2
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. *(Optionnel, GPU NVIDIA CUDA 12)* — Pour l’échantillonnage RAPIDS dans `scripts/precursor.py` :

```bash
pip install -r requirements-rapids.txt
```

---

## Structure du projet

```
sysdereco_devoir2/
├── README.md
├── requirements.txt
├── requirements-rapids.txt                 # Optionnel : RAPIDS (cudf, rmm, cupy) pour GPU
├── main.py                                 # Point d’entrée principal (si utilisé)
├── notebook_recommandation_contenu.ipynb   # Notebook principal (à renommer pour remise)
├── scripts/
│   ├── precursor.py                        # Échantillonnage et prétraitement (GPU ou CPU)
│   └── check_env.py                        # Vérification de l’environnement
├── config/
│   └── default.yaml
├── data/                                   # Données (meta_Books, interactions P1, processed/)
├── models/
└── results/
```

**Remise :** renommer le notebook en `INF6083-P2-EquipeN-Code.ipynb` (N = numéro d’équipe) et l’inclure dans le zip avec le rapport PDF et le `README.md`.

---

## Exécution

1. *(Optionnel)* Vérifier l’environnement : `python scripts/check_env.py`
2. Données : placer dans `data/` les fichiers sources (ex. `meta_Books.jsonl.gz`, JSONL/Parquet d’interactions). L’échantillonnage et le prétraitement peuvent être lancés via `python scripts/precursor.py` (voir docstring dans le script).
3. Notebook : `jupyter notebook notebook_recommandation_contenu.ipynb` (ou JupyterLab / VS Code). Exécuter les cellules dans l’ordre (Tâche 0 → … → Tâche 5).
4. *(Optionnel)* Point d’entrée alternatif : `main.py` si le projet utilise un pipeline scripté.

---

## Tâches (résumé)

| Tâche | Contenu |
|-------|--------|
| **0** | Préparation : jointure interactions/métadonnées, split train/test temporel, représentations d’items (TF-IDF), profils utilisateurs. |
| **1** | Recommandation par similarité de contenu (profil–item), top-N, analyse qualitative. |
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
