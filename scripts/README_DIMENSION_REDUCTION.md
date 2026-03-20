# Scripts - Tâche 2: Réduction de Dimension

Ce répertoire contient les scripts pour la **Tâche 2** du projet: représentations latentes et réduction de dimension.

---

## Fichiers Principaux

### `dimension_reduction.py` ⭐ NOUVEAU

**Pipeline complet de réduction de dimension avec TruncatedSVD.**

#### Fonctionnalités:
- ✅ Teste 4 dimensions latentes: **50, 100, 200, 300**
- ✅ S'entraîne **uniquement sur train** (évite data leakage)
- ✅ Transforme items train ET test dans espace latent
- ✅ Calcule **variance expliquée** cumulée
- ✅ Mesure **coût computationnel** (fit + inference)
- ✅ Génère **rapport comparatif** avec recommandation automatique
- ✅ Sauvegarde modèles SVD + matrices réduites

#### Utilisation:

```bash
# Prérequis: matrices TF-IDF générées
python scripts/item_representation.py

# Exécuter réduction de dimension
python scripts/dimension_reduction.py
```

#### Sorties:

Pour chaque variant (ex: `active_pre_split`):

```
data/joining/active_pre_split/
├── items_reduced_svd_50d.npy       # Matrice réduite 50D (n_items × 50)
├── items_reduced_svd_100d.npy      # Matrice réduite 100D
├── items_reduced_svd_200d.npy      # Matrice réduite 200D
├── items_reduced_svd_300d.npy      # Matrice réduite 300D
├── reducer_svd_50d.pkl             # Modèle TruncatedSVD 50D
├── reducer_svd_100d.pkl            # Modèle TruncatedSVD 100D
├── reducer_svd_200d.pkl            # Modèle TruncatedSVD 200D
├── reducer_svd_300d.pkl            # Modèle TruncatedSVD 300D
├── item_ids.npy                    # IDs items (ordre des lignes)
├── metrics_svd_50d.json            # Métriques détaillées 50D
├── metrics_svd_100d.json           # Métriques détaillées 100D
├── metrics_svd_200d.json           # Métriques détaillées 200D
├── metrics_svd_300d.json           # Métriques détaillées 300D
└── dimension_comparison.json       # ⭐ Rapport comparatif + recommandation
```

#### Structure du Rapport (`dimension_comparison.json`):

```json
{
  "variant": "active_pre_split",
  "method": "svd",
  "methodology": {
    "chosen_method": "TruncatedSVD",
    "reasons": ["Optimisée matrices creuses", "Efficace mémoire", ...],
    "alternatives": {
      "NMF": {"advantages": [...], "limitations": [...]},
      "PCA": {"advantages": [...], "limitations": [...]}
    },
    "svd_advantages": [...],
    "svd_limitations": [...]
  },
  "dimensions_tested": [50, 100, 200, 300],
  "comparison_results": [
    {
      "method": "svd",
      "n_components": 50,
      "input_shape": [10123, 29847],
      "output_shape": [10123, 50],
      "fit_time_s": 1.8421,
      "transform_time_per_sample_ms": 0.0234,
      "variance_explained": 0.3214,
      "variance_explained_pct": 32.14,
      "variance_per_component": [0.0423, 0.0312, ...],
      "singular_values": [42.31, 38.12, ...]
    },
    ...
  ],
  "analysis": {
    "summary": [
      {"dimension": 50, "variance_pct": 32.14, "fit_time_s": 1.84},
      {"dimension": 100, "variance_pct": 44.72, "fit_time_s": 3.21},
      {"dimension": 200, "variance_pct": 59.31, "fit_time_s": 5.12},
      {"dimension": 300, "variance_pct": 68.24, "fit_time_s": 7.43}
    ],
    "marginal_gains": [
      {"from_dim": 50, "to_dim": 100, "variance_gain_pct": 12.58},
      {"from_dim": 100, "to_dim": 200, "variance_gain_pct": 14.59},
      {"from_dim": 200, "to_dim": 300, "variance_gain_pct": 8.93}
    ],
    "recommendation": {
      "dimension": 100,
      "variance_pct": 44.72,
      "fit_time_s": 3.21,
      "rationale": [
        "Variance expliquée: 44.7%",
        "Temps d'entraînement acceptable: 3.2s",
        "Bon compromis variance/coût computationnel",
        "Gain marginal décroissant au-delà de cette dimension"
      ]
    }
  },
  "build_time_s": 18.34
}
```

---

### `item_representation.py`

**Génération matrices TF-IDF (Tâche 1).**

- Charge datasets joints (`*_clean_joined.parquet`)
- Prétraitement texte (nettoyage, stopwords, etc.)
- Vectorisation TF-IDF avec `TfidfVectorizer`
- Fusion avec attributs numériques (rating, price)
- Sauvegarde matrices creuses (`.npz`)

**Artéfacts produits:**
```
data/joining/{variant}/
├── books_representation_sparse.npz  # Matrice TF-IDF + numeric (sparse)
└── vocabulary_tfidf.txt             # Vocabulaire (feature names)
```

---

### `user_profile.py` 🔄 MIS À JOUR

**Construction profils utilisateurs (Tâches 1 & 2).**

#### Nouveautés:

- ✅ **Support modes SVD multi-dimensions**:
  - `mode="svd_50d"`: Profils SVD 50D
  - `mode="svd_100d"`: Profils SVD 100D
  - `mode="svd_200d"`: Profils SVD 200D
  - `mode="svd_300d"`: Profils SVD 300D
  - `mode="svd_auto"`: 🌟 Dimension recommandée (depuis `dimension_comparison.json`)

- ✅ **Chargement automatique dimension optimale**

#### Utilisation:

```python
from scripts.user_profile import (
    DatasetManager,
    ItemRepresentationLoader,
    UserProfileBuilder
)

# 1. Charger dataset train
ds = DatasetManager("data/joining/active_pre_split/train_interactions.parquet")

# 2. Charger représentations items
loader = ItemRepresentationLoader("data/joining/active_pre_split")
loader.load()

# 3. Construire profils avec dimension recommandée
builder = UserProfileBuilder(ds, loader, mode="svd_auto")
profiles, user_ids, report = builder.build()

# OU: dimension spécifique
builder_100d = UserProfileBuilder(ds, loader, mode="svd_100d")
profiles_100d, _, _ = builder_100d.build()
```

---

### `similarity.py`

**Calcul similarité cosine profil-items.**

- Compatible TF-IDF **et** SVD (dense/sparse)
- Batch processing pour scalabilité
- Retourne matrice scores (n_users × n_items)

**Utilisation:**
```python
from scripts.similarity import compute_similarity

# Fonctionne avec TF-IDF (sparse) ou SVD (dense)
scores = compute_similarity(user_profiles, item_matrix, batch_size=500)
```

---

### `joining.py`

**Jointure interactions ↔ metadata (Tâche 0).**

- Fusionne reviews avec métadonnées livres
- Normalisation, imputation, diagnostics
- Produit `*_clean_joined.parquet`

---

### `precursor.py`

**Pipeline P1 (échantillonnage, filtrage, splits).**

- Conversion JSONL → Parquet
- Échantillonnage utilisateurs actifs / temporel
- Filtrage k-core itératif
- Split train/test

---

## Fichiers Temporaires/Prototypes

### `temp/item_representation_bis.py`

**Prototype initial de réduction SVD (600D fixe).**

⚠️ **Remplacé par `dimension_reduction.py`** qui offre:
- Comparaison multi-dimensions (50, 100, 200, 300)
- Analyse variance vs. coût
- Recommandation automatique
- Métriques détaillées

**Utiliser `dimension_reduction.py` à la place.**

---

## Workflow Complet (Tâches 1 & 2)

### Étape 1: Préparation (Tâche 0)

```bash
# Générer datasets joints
python scripts/joining.py
```

**Produit:** `data/joining/*_clean_joined.parquet`

---

### Étape 2: Représentation TF-IDF (Tâche 1)

```bash
# Générer matrices TF-IDF
python scripts/item_representation.py
```

**Produit:** `books_representation_sparse.npz`, `vocabulary_tfidf.txt`

---

### Étape 3: Réduction de Dimension (Tâche 2)

```bash
# Appliquer SVD multi-dimensions
python scripts/dimension_reduction.py
```

**Produit:**
- Matrices réduites (50D, 100D, 200D, 300D)
- Modèles SVD entraînés
- Rapport comparatif avec recommandation

---

### Étape 4: Profils Utilisateurs (Tâches 1 & 2)

```bash
# Construire profils TF-IDF et SVD
python scripts/user_profile.py
```

**Produit:**
- `user_profiles_tfidf.npz` (sparse)
- `user_profiles_svd.npy` (dense, dimension recommandée)
- `user_ids.npy`
- `user_profiles_report.json`

---

### Étape 5: Évaluation (Notebook)

Voir `notebook_recommandation_contenu.ipynb` pour:
- Comparaison TF-IDF vs. SVD (50D, 100D, 200D, 300D)
- Métriques: NDCG@10, Recall@5/10, MAP
- Analyse compromis variance/qualité/coût

---

## Documentation Complémentaire

### `docs/dimension_reduction_methodology.md`

Documentation complète de la méthodologie:

1. **Choix de la méthode** (SVD vs. NMF vs. PCA)
2. **Avantages et limites** de TruncatedSVD
3. **Application** (entraînement train, transformation test)
4. **Analyse du compromis** (variance, coût, qualité)
5. **Sélection dimension** (critères et recommandation)
6. **Intégration pipeline** (artéfacts, usage notebook)
7. **Résultats attendus** (courbes variance, temps, NDCG)
8. **Références** (papiers, scikit-learn docs)

---

## Dépendances

```bash
# requirements.txt
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3   # TruncatedSVD, TfidfVectorizer
pandas>=2.0
pyarrow
```

---

## Troubleshooting

### "Matrice TF-IDF introuvable"

```bash
# Vérifier que item_representation.py a été exécuté
ls -lh data/joining/*/books_representation_sparse.npz

# Si manquant, générer:
python scripts/item_representation.py
```

---

### "Matrice SVD XD introuvable"

```bash
# Vérifier que dimension_reduction.py a été exécuté
ls -lh data/joining/*/items_reduced_svd_*d.npy

# Si manquant, générer:
python scripts/dimension_reduction.py
```

---

### Temps d'exécution long (> 60s par dimension)

**Causes possibles:**
- Vocabulaire TF-IDF très large (> 50k features)
- Matrice très dense (densité > 0.01)
- Nombre d'items élevé (> 50k)

**Solutions:**
1. Réduire vocabulaire TF-IDF:
   ```python
   # Dans item_representation.py
   TFIDF_PARAMS = {
       "max_features": 20_000,  # Limiter vocabulaire
       ...
   }
   ```

2. Augmenter filtrage fréquence:
   ```python
   TFIDF_PARAMS = {
       "min_df": 10,   # Au lieu de 5
       "max_df": 0.90, # Au lieu de 0.95
   }
   ```

3. Réduire dimensions testées:
   ```python
   # Dans dimension_reduction.py
   LATENT_DIMENSIONS = [100, 200]  # Au lieu de [50, 100, 200, 300]
   ```

---

### MemoryError avec PCA

⚠️ **Ne PAS utiliser PCA pour matrices creuses!**

- PCA nécessite conversion dense → explosion mémoire
- Pour 10k items × 30k features = 1.2 GB (dense)
- Utiliser **exclusivement TruncatedSVD**

---

## Auteurs

Équipe ML - INF6083 Projet P2
Mars 2026
