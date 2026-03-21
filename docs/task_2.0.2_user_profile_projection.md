# Tâche 2.0.2 - Projection des profils utilisateurs dans l'espace latent

## Description

Cette tâche projette les profils utilisateurs construits en Tâche 0 (`user_profile.py`) dans l'espace latent appris par SVD en Tâche 2.0.1 (`dimension_reduction.py`).

## Contraintes expérimentales

✅ **Profils et items dans le même espace vectoriel**
- Les profils utilisateurs et les vecteurs items sont projetés avec la même transformation SVD
- Même dimension latente (k) pour les deux matrices
- Permet le calcul de similarité cosinus entre profils et items

✅ **Aucune donnée du test utilisée**
- Seules les interactions de `train_interactions.parquet` sont utilisées
- Les profils sont construits uniquement à partir des données d'entraînement
- Garantit l'absence de fuite de données

✅ **Projection cohérente avec celle des items**
- Le modèle SVD appris sur les items TF-IDF est réutilisé
- Les profils TF-IDF sont transformés avec `svd.transform()`
- Même variance expliquée pour profils et items

## Fichiers créés

### `scripts/user_profile_projection.py`

Script principal qui:
1. Charge les artéfacts SVD (modèles et matrices réduites des items)
2. Construit les profils utilisateurs dans l'espace TF-IDF
3. Projette les profils dans l'espace latent pour chaque dimension (50, 100, 200, 300)
4. Sauvegarde les matrices et rapports

**Utilisation:**
```bash
python scripts/user_profile_projection.py
```

### `scripts/validate_user_projection.py`

Script de validation qui vérifie:
- Existence des fichiers de sortie
- Compatibilité des dimensions entre profils et items
- Absence de valeurs NaN ou Inf
- Cohérence du nombre d'utilisateurs

**Utilisation:**
```bash
python scripts/validate_user_projection.py
```

## Sorties produites

Pour chaque variant (active_pre_split, temporal_pre_split):

### Matrices latentes
```
results/svd/<variant>/
├── user_profiles_latent_50d.npy      # (n_users × 50)
├── user_profiles_latent_100d.npy     # (n_users × 100)
├── user_profiles_latent_200d.npy     # (n_users × 200)
├── user_profiles_latent_300d.npy     # (n_users × 300)
├── user_ids_latent.npy               # (n_users,) - mapping des lignes
└── items_reduced_svd_{dim}d.npy      # (n_items × dim) - déjà créé par Tâche 2.0.1
```

### Rapports
```
results/svd/<variant>/
├── user_profile_projection_report.json       # Rapport complet
├── user_profile_projection_50d.json          # Métriques 50D
├── user_profile_projection_100d.json         # Métriques 100D
├── user_profile_projection_200d.json         # Métriques 200D
└── user_profile_projection_300d.json         # Métriques 300D
```

## Algorithme

### 1. Construction des profils TF-IDF

```python
# Étape 1: Filtrer les interactions pour ne garder que les items connus
valid_interactions = train_df[train_df["parent_asin"].isin(item_index)]

# Étape 2: Construire la matrice de pondération R (users × items)
# R[u, i] = rating de l'utilisateur u pour l'item i
R = csr_matrix((ratings, (user_rows, item_cols)), shape=(n_users, n_items))

# Étape 3: Calculer les profils par moyenne pondérée
# P_tfidf = (R @ TF-IDF) / sum(ratings)
profiles_tfidf = (R @ item_tfidf_matrix) / weight_sums
```

### 2. Projection dans l'espace latent

```python
# Étape 1: Charger le modèle SVD appris sur les items
with open(f"reducer_svd_{dim}d.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Étape 2: Projeter les profils TF-IDF
# Les profils sont dans le même espace que les items TF-IDF
# Donc on peut directement appliquer la transformation
latent_profiles = svd_model.transform(profiles_tfidf)

# Étape 3: Sauvegarder en float32 pour cohérence avec les items
latent_profiles = latent_profiles.astype(np.float32)
np.save(f"user_profiles_latent_{dim}d.npy", latent_profiles)
```

## Vérification des contraintes

Le script génère un rapport JSON avec:

```json
{
  "constraints_satisfied": {
    "same_vector_space": true,
    "no_test_data_used": "Only train_interactions.parquet used",
    "consistent_with_items": "SVD model applied to user profiles in TF-IDF space"
  },
  "projection_results": [
    {
      "dimension": 50,
      "n_users": 12345,
      "n_items": 22007,
      "profile_shape": [12345, 50],
      "item_shape": [22007, 50],
      "same_vector_space": true,
      "variance_explained_pct": 16.23,
      "transform_time_s": 0.0234
    }
  ]
}
```

## Utilisation ultérieure

Les matrices latentes sont utilisées pour:
1. **Recommandation**: calcul de similarité cosinus entre profils et items
2. **Évaluation**: prédiction des préférences sur le test set
3. **Analyse**: visualisation dans l'espace latent (t-SNE, UMAP)

**Exemple d'utilisation:**
```python
import numpy as np

# Charger les matrices latentes
user_profiles = np.load("results/svd/active_pre_split/user_profiles_latent_100d.npy")
item_vectors = np.load("results/svd/active_pre_split/items_reduced_svd_100d.npy")
user_ids = np.load("results/svd/active_pre_split/user_ids_latent.npy")

# Calculer les similarités pour un utilisateur
user_idx = 0
similarities = user_profiles[user_idx] @ item_vectors.T  # (n_items,)
top_k = np.argsort(similarities)[-10:][::-1]  # Top 10 items

print(f"Top 10 items pour {user_ids[user_idx]}:")
for item_idx in top_k:
    print(f"  Item {item_idx}: similarité = {similarities[item_idx]:.4f}")
```

## Dépendances

- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- pyarrow >= 12.0.0

## Références

- Tâche 0: Construction des profils utilisateurs (`user_profile.py`)
- Tâche 2.0.1: Réduction de dimension SVD (`dimension_reduction.py`)
- Rapport: `INF6083_projet_p2.pdf`, Section 3.2
