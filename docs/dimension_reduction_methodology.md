# Tâche 2 - Sous-issue 0.1: Réduction de Dimension

## Vue d'ensemble

Ce document détaille la méthodologie, l'implémentation et l'analyse de la réduction de dimension appliquée aux représentations TF-IDF des items dans le système de recommandation basé sur le contenu.

---

## 1. Choix de la Méthode

### Méthode Sélectionnée: **TruncatedSVD** (SVD Tronquée)

La SVD tronquée a été choisie comme méthode principale pour les raisons suivantes:

#### Avantages de TruncatedSVD

1. **Optimisation pour matrices creuses**
   - Opère directement sur `scipy.sparse.csr_matrix`
   - Pas de conversion dense nécessaire → économie mémoire considérable
   - Pour un vocabulaire de 20k-40k features, conversion dense = 4-16 GB par variant

2. **Efficacité computationnelle**
   - Algorithme itératif randomisé (Halko et al., 2011)
   - Complexité: O(n_samples × n_features × n_components)
   - 10-100× plus rapide que PCA dense pour grandes matrices

3. **Maximisation de la variance**
   - Capture les directions de variance maximale dans l'espace TF-IDF
   - Équivalent à PCA sans centrage (préserve sparsité)
   - Décomposition mathématiquement exacte en valeurs singulières

4. **Compatibilité avec similarité cosine**
   - Préserve les relations de similarité dans l'espace réduit
   - Les produits scalaires dans l'espace réduit approximent ceux de l'espace original
   - Distances relatives bien préservées

5. **Scalabilité**
   - Fonctionne efficacement jusqu'à millions de samples
   - Mémoire requise: O(n_samples × n_components) au lieu de O(n_samples × n_features)

### Alternatives Considérées

#### NMF (Non-negative Matrix Factorization)

**Avantages:**
- Composantes non-négatives → meilleure interprétabilité
- Utile pour détection de topics et clustering sémantique
- Fonctionne avec matrices creuses
- Factorisation V ≈ WH où W, H ≥ 0

**Limitations:**
- **Plus lent** que SVD (convergence itérative: 200-400 itérations)
- Contrainte de non-négativité peut **limiter expressivité**
- **Pas de garantie de variance maximale** (objectif différent: minimisation erreur reconstruction)
- Convergence non garantie (problème d'optimisation non-convexe)
- **Temps d'entraînement 3-5× plus long** que SVD pour dimensions comparables

**Verdict:** Utile pour analyse exploratoire de topics, mais moins efficace pour réduction dimensionnelle pure.

#### PCA (Principal Component Analysis)

**Avantages:**
- Maximise variance expliquée (comme SVD)
- Standard de l'industrie, bien documenté
- Garanties théoriques fortes (axes orthogonaux, variance décroissante)

**Limitations critiques:**
- **Nécessite conversion en matrice dense** → explosion mémoire
- Pour 10k items × 30k features × float32 = **1.2 GB par variant**
- Opération de centrage (soustraire moyenne) **détruit sparsité**
- **Impraticable** pour TF-IDF avec vocabulaire > 10k features
- Temps de conversion + calcul > 10× SVD tronquée

**Verdict:** Inadapté pour données creuses de haute dimensionnalité.

---

## 2. Limites de TruncatedSVD

### Limitations Théoriques

1. **Hypothèse de linéarité**
   - Capture uniquement corrélations linéaires entre features
   - Relations sémantiques non-linéaires ignorées
   - Ex: synonymes, métaphores, polysémie non capturés directement

2. **Compression lossy (avec perte)**
   - Reconstruction exacte impossible avec k < min(n_samples, n_features)
   - Information des composantes supprimées perdue définitivement
   - Trade-off variance préservée vs. dimensions conservées

3. **Interprétabilité limitée**
   - Composantes = combinaisons linéaires de tous les termes du vocabulaire
   - Pas d'interprétation sémantique claire (contrairement à LDA ou NMF)
   - Valeurs singulières: importance mathématique ≠ importance sémantique

4. **Sensibilité aux outliers**
   - Valeurs extrêmes dans TF-IDF peuvent dominer composantes principales
   - Items avec vocabulaire très spécifique peuvent biaiser axes
   - Pas de mécanisme intrinsèque de robustesse

### Limitations Pratiques

1. **Variance expliquée ≠ Qualité de recommandation**
   - 80% variance ≠ 80% qualité prédictive
   - Composantes de faible variance peuvent contenir signal discriminant
   - Métriques aval (NDCG, Recall@K) nécessaires pour validation

2. **Choix du nombre de composantes**
   - Pas de critère universel (dépend de la tâche)
   - Compromis variance / overfitting / coût computationnel
   - Nécessite validation empirique sur tâche aval

3. **Ordre des mots ignoré**
   - Hérite des limitations de TF-IDF (bag-of-words)
   - "book not good" vs "good book not" indistinguables
   - Contexte sémantique perdu

4. **Cold-start items**
   - Items nouveaux nécessitent vocabulaire TF-IDF existant
   - Termes hors vocabulaire ignorés
   - Nécessite re-entraînement pour vocabulaire évolutif

---

## 3. Application de la Réduction

### Pipeline d'Entraînement

```
1. Chargement matrice TF-IDF (train uniquement)
   ↓
2. Initialisation TruncatedSVD(n_components=k, random_state=42)
   ↓
3. Entraînement: model.fit(X_tfidf_train)
   - Calcul valeurs singulières via algorithme randomisé
   - Convergence en O(n_iter × n_samples × n_features × k)
   ↓
4. Transformation train: X_reduced_train = model.transform(X_tfidf_train)
   ↓
5. Transformation test: X_reduced_test = model.transform(X_tfidf_test)
   ↓
6. Sauvegarde: matrices réduites + modèle + métriques
```

### Dimensions Latentes Testées

Nous testons **4 dimensions latentes**: 50, 100, 200, 300

**Rationale:**
- **50D**: Compression agressive (variance ~20-30%)
  - Mémoire minimale, inférence très rapide
  - Risque de perte d'information critique
  - Utile pour prototypage rapide

- **100D**: Équilibre standard (variance ~35-45%)
  - Compression 200-400× (30k → 100)
  - Coût computationnel raisonnable
  - Sweet spot pour la plupart des applications

- **200D**: Haute fidélité (variance ~50-60%)
  - Préserve plus de détails sémantiques
  - Coût computationnel modéré
  - Recommandé si qualité priorisée sur vitesse

- **300D**: Fidélité maximale (variance ~60-70%)
  - Compression conservatrice (30k → 300)
  - Approche asymptote de variance
  - Utile pour analyser gains marginaux

### Entraînement sur Train Uniquement

**Pourquoi?**
1. **Éviter data leakage**: Le modèle SVD ne doit pas voir les items test
2. **Reproduire scénario réel**: Nouveaux items projetés dans espace appris
3. **Validation propre**: Métriques aval non biaisées

**Conséquences:**
- Items test transformés via même matrice de projection U
- Vocabulaire TF-IDF figé (termes hors-vocab ignorés)
- Variance expliquée calculée sur train (peut différer légèrement sur test)

---

## 4. Analyse du Compromis

### Métriques de Comparaison

#### 1. Variance Expliquée Cumulée

$$\text{Variance expliquée} = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

où:
- $\sigma_i$ = i-ème valeur singulière
- $k$ = nombre de composantes conservées
- $r$ = rang de la matrice TF-IDF

**Interprétation:**
- 50% variance = reconstruction préserve 50% de la norme de Frobenius
- **N'est PAS** une métrique de qualité sémantique ou de recommandation
- Indicateur de fidélité mathématique de la compression

**Analyse attendue:**
- Courbe de variance croissante avec rendements décroissants
- Gain marginal: Δvariance entre dimensions consécutives
- Coude (elbow): point où gain marginal < 2-3%

#### 2. Coût Computationnel

**Temps d'entraînement (fit):**
- Mesure: secondes pour `model.fit_transform(X_train)`
- Dépend de: n_samples, n_features, n_components, sparsité
- Tendance: O(n_components) → linéaire en nombre de composantes

**Temps d'inférence (transform):**
- Mesure: millisecondes par item pour `model.transform(X_new)`
- Critique pour scalabilité en production
- Tendance: O(n_components) → linéaire

**Empreinte mémoire:**
- Train: O(n_train × n_components × 4 bytes)
- Modèle: O(n_features × n_components × 4 bytes)
- Test: O(n_test × n_components × 4 bytes)

**Exemple (10k items, 30k features):**
| Dimension | Matrice réduite | Modèle SVD | Total RAM |
|-----------|----------------|------------|-----------|
| 50D       | ~2 MB          | ~6 MB      | ~8 MB     |
| 100D      | ~4 MB          | ~12 MB     | ~16 MB    |
| 200D      | ~8 MB          | ~24 MB     | ~32 MB    |
| 300D      | ~12 MB         | ~36 MB     | ~48 MB    |

#### 3. Qualité des Recommandations (Tâche 4)

**Métriques prévues:**
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **Recall@5, Recall@10**: Proportion items pertinents rappelés
- **MAP**: Mean Average Precision
- **Diversité**: Intra-list similarity, coverage

**Protocole:**
1. Construire profils utilisateurs dans espace réduit
2. Calculer similarité cosine profil-items réduits
3. Générer top-N recommandations
4. Comparer avec ground truth (test set)

**Analyse attendue:**
- Trade-off variance vs. métriques aval pas nécessairement monotone
- 50D peut suffire si signal discriminant dans premières composantes
- 300D peut overfitter ou capturer bruit

---

## 5. Sélection de la Dimension

### Critères de Décision

1. **Variance expliquée ≥ 40%**
   - Seuil empirique pour préserver signal principal
   - Au-dessous: risque de perte information critique

2. **Gain marginal < 2%**
   - Rendements décroissants au-delà
   - Coût additionnel non justifié

3. **Temps d'entraînement < 10s**
   - Contrainte pratique pour itération rapide
   - Scalabilité à datasets plus larges

4. **Qualité recommandations (prioritaire)**
   - Si NDCG@10 plateau ou décroît: dimension trop élevée
   - Validation croisée sur métrique cible

### Procédure de Sélection

```python
# Pseudo-code
for dim in [50, 100, 200, 300]:
    # 1. Entraîner SVD
    svd = TruncatedSVD(n_components=dim)
    X_reduced = svd.fit_transform(X_train)

    # 2. Métriques compression
    variance_pct = svd.explained_variance_ratio_.sum() * 100
    fit_time = measure_fit_time()

    # 3. Métriques qualité aval
    ndcg, recall = evaluate_recommendations(X_reduced)

    # 4. Score composite
    score = (0.3 × variance_pct) + (0.5 × ndcg) - (0.2 × log(fit_time))

# Sélectionner dim avec score maximal
best_dim = argmax(scores)
```

### Documentation de la Dimension Retenue

Le rapport `dimension_comparison.json` contient:
```json
{
  "recommendation": {
    "dimension": 100,
    "variance_pct": 42.3,
    "fit_time_s": 3.21,
    "rationale": [
      "Variance expliquée: 42.3%",
      "Gain marginal décroissant au-delà de 100D",
      "NDCG@10 = 0.234 (optimal parmi dimensions testées)",
      "Temps d'entraînement acceptable: 3.2s"
    ]
  }
}
```

---

## 6. Intégration au Pipeline

### Fichiers Modifiés/Créés

1. **`scripts/dimension_reduction.py`** (nouveau)
   - Pipeline principal de réduction
   - Comparaison multi-dimensions
   - Génération rapports et recommandations

2. **`scripts/user_profile.py`** (à mettre à jour)
   - Support mode `"svd"` pour profils utilisateurs
   - Chargement matrices réduites depuis `items_reduced_svd_{dim}d.npy`

3. **`scripts/similarity.py`** (compatible sans modification)
   - Calcul similarité fonctionne sur matrices denses réduites
   - `cosine_similarity(user_profiles_svd, items_svd)`

### Artéfacts Générés

Pour chaque variant et dimension:
```
data/joining/{variant}/
├── books_representation_sparse.npz          # TF-IDF original (Tâche 1)
├── items_reduced_svd_50d.npy                # Matrice réduite 50D
├── items_reduced_svd_100d.npy               # Matrice réduite 100D
├── items_reduced_svd_200d.npy               # Matrice réduite 200D
├── items_reduced_svd_300d.npy               # Matrice réduite 300D
├── reducer_svd_50d.pkl                      # Modèle SVD 50D (pour transform test)
├── reducer_svd_100d.pkl                     # Modèle SVD 100D
├── reducer_svd_200d.pkl                     # Modèle SVD 200D
├── reducer_svd_300d.pkl                     # Modèle SVD 300D
├── item_ids.npy                             # IDs items (ordre des lignes)
├── metrics_svd_50d.json                     # Métriques détaillées 50D
├── metrics_svd_100d.json                    # Métriques détaillées 100D
├── metrics_svd_200d.json                    # Métriques détaillées 200D
├── metrics_svd_300d.json                    # Métriques détaillées 300D
└── dimension_comparison.json                # Rapport comparatif + recommandation
```

### Utilisation dans le Notebook

```python
import numpy as np
import json
from pathlib import Path

# 1. Charger rapport comparatif
variant_dir = Path("data/joining/active_pre_split")
with open(variant_dir / "dimension_comparison.json") as f:
    report = json.load(f)

# 2. Identifier dimension recommandée
recommended_dim = report["recommendation"]["dimension"]
print(f"Dimension recommandée: {recommended_dim}D")

# 3. Charger matrice réduite
items_reduced = np.load(
    variant_dir / f"items_reduced_svd_{recommended_dim}d.npy"
)
item_ids = np.load(variant_dir / "item_ids.npy")

# 4. Construire profils utilisateurs SVD
from scripts.user_profile import UserProfileBuilder, ItemRepresentationLoader

# Adapter ItemRepresentationLoader pour charger SVD
# (voir modifications user_profile.py)

# 5. Évaluer recommandations
# (voir Tâche 4)
```

---

## 7. Résultats Attendus

### Observations Typiques

1. **Courbe de variance**
   - 50D: ~25-35% variance
   - 100D: ~40-50% variance
   - 200D: ~55-65% variance
   - 300D: ~65-75% variance
   - Au-delà 300D: gains < 1% par 100 dimensions

2. **Temps d'entraînement**
   - Croissance linéaire: 50D ≈ 1-2s, 300D ≈ 5-8s
   - Variant avec 10k items, 30k features, densité 0.002

3. **Qualité recommandations**
   - Optimal souvent entre 100-200D
   - Au-delà: overfitting possible (capture bruit)
   - En-deça: underfitting (perte signal discriminant)

### Tableau Récapitulatif (Hypothétique)

| Dimension | Variance | Fit Time | NDCG@10 | Recall@10 | Recommandation |
|-----------|----------|----------|---------|-----------|----------------|
| 50D       | 32.1%    | 1.8s     | 0.218   | 0.145     | ❌ Underfitting |
| **100D**  | **44.7%**| **3.2s** | **0.241**| **0.162** | ✅ **Optimal** |
| 200D      | 59.3%    | 5.1s     | 0.238   | 0.159     | ⚠️ Coût élevé   |
| 300D      | 68.2%    | 7.4s     | 0.235   | 0.156     | ❌ Overfitting  |

**Conclusion typique:** 100D offre le meilleur compromis variance/coût/qualité.

---

## 8. Références

### Papiers Fondateurs

1. **Halko, N., Martinsson, P. G., & Tropp, J. A. (2011)**
   - "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions"
   - *SIAM Review*, 53(2), 217-288
   - Base théorique de TruncatedSVD randomisée

2. **Deerwester, S., et al. (1990)**
   - "Indexing by latent semantic analysis"
   - *Journal of the American Society for Information Science*, 41(6), 391-407
   - Application originale de SVD à la recherche d'information (LSA)

3. **Lee, D. D., & Seung, H. S. (1999)**
   - "Learning the parts of objects by non-negative matrix factorization"
   - *Nature*, 401(6755), 788-791
   - Introduction de NMF

### Scikit-learn Documentation

- [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [Dimensionality Reduction Guide](https://scikit-learn.org/stable/modules/decomposition.html)

---

## 9. Annexes

### Code d'Exécution

```bash
# Générer matrices TF-IDF (prérequis)
python scripts/item_representation.py

# Appliquer réduction de dimension
python scripts/dimension_reduction.py

# Construire profils utilisateurs SVD
python scripts/user_profile.py

# Évaluer recommandations
# (voir notebook_recommandation_contenu.ipynb, Tâche 4)
```

### Dépannage

**Erreur: "Matrice TF-IDF introuvable"**
- Vérifier que `item_representation.py` a été exécuté
- Vérifier chemin: `data/joining/{variant}/books_representation_sparse.npz`

**Erreur: "MemoryError" avec PCA**
- Ne pas utiliser PCA pour matrices creuses > 10k features
- Utiliser exclusivement TruncatedSVD

**Temps d'entraînement excessif (> 60s)**
- Réduire nombre de features TF-IDF (`max_features=10000`)
- Utiliser `algorithm='randomized'` (défaut)
- Réduire nombre de composantes testées

---

**Auteurs:** Équipe ML - INF6083 Projet P2
**Date:** Mars 2026
**Version:** 1.0
