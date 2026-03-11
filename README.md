# INF6083 — Projet P2 : Recommandation basée sur le contenu

Projet 2 du cours *Systèmes de recommandation* (INF6083). Recommandation **basée sur le contenu** à partir des métadonnées des livres (Amazon Reviews 2023, catégorie Books), avec un **notebook** comme livrable principal.

---

## Jeu de données

- **Source :** Amazon Reviews 2023, catégorie **Books** (même sous-ensemble que le projet P1).
- **Fichier central :** `meta_Books.jsonl.gz` (métadonnées : `parent_asin`, `title`, `description`, `categories`, `average_rating`, `rating_number`, `price`).
- Les **interactions** utilisateur–item proviennent du sous-ensemble de travail P1 (à placer dans `data/` ou à indiquer en début de notebook).

---

## Prérequis

- **Python** 3.10+
- **Environnement** : venv recommandé (Linux / Windows).
- **Optionnel** : CUDA pour accélération GPU (PyTorch).

---

## Installation

```bash
cd sysdereco_devoir2
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Structure du projet (format simple)

```
sysdereco_devoir2/
├── README.md
├── requirements.txt
├── notebook_recommandation_contenu.ipynb   # Notebook principal (à renommer pour remise)
├── data/                                   # Données (meta_Books.jsonl.gz + interactions P1)
├── config/
│   └── default.yaml                        # Optionnel : paramètres (référencés dans le notebook si besoin)
├── models/                                  # Modèles sauvegardés (optionnel)
└── results/                                # Résultats / figures pour le rapport
```

**Remise :** renommer le notebook en `INF6083-P2-EquipeN-Code.ipynb` (N = numéro d’équipe) et l’inclure dans le zip avec le rapport PDF et le `README.md`.

---

## Exécution

1. *(Optionnel)* Vérifier l’environnement : `python scripts/check_env.py`
2. Placer les données dans `data/` : `meta_Books.jsonl.gz` et le fichier d’interactions (sous-ensemble P1).
3. Ouvrir le notebook dans Jupyter :  
   `jupyter notebook notebook_recommandation_contenu.ipynb`  
   ou avec JupyterLab / VS Code.
4. Exécuter les cellules dans l’ordre (Tâche 0 → … → Tâche 5).

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
