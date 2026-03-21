"""
Exemple d'utilisation des profils utilisateurs projetés dans l'espace latent.

Ce script démontre comment:
1. Charger les profils utilisateurs latents
2. Charger les vecteurs items latents
3. Calculer les similarités cosinus
4. Générer des recommandations

Usage: python example_latent_recommendation.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def load_latent_data(variant: str, dimension: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Charge les données latentes pour un variant et une dimension donnés.

    Args:
        variant: Nom du variant (ex: "active_pre_split", "temporal_pre_split")
        dimension: Dimension latente (50, 100, 200, 300)

    Returns:
        user_profiles: matrice (n_users × dimension)
        item_vectors: matrice (n_items × dimension)
        user_ids: array des user_ids correspondant aux lignes de user_profiles
    """
    results_dir = Path("results/svd") / variant

    # Charger les profils utilisateurs
    user_profiles_path = results_dir / f"user_profiles_latent_{dimension}d.npy"
    if not user_profiles_path.exists():
        raise FileNotFoundError(
            f"Profils utilisateurs introuvables: {user_profiles_path}\n"
            f"Exécutez d'abord: python scripts/user_profile_projection.py"
        )
    user_profiles = np.load(user_profiles_path)

    # Charger les vecteurs items
    item_vectors_path = results_dir / f"items_reduced_svd_{dimension}d.npy"
    if not item_vectors_path.exists():
        raise FileNotFoundError(
            f"Vecteurs items introuvables: {item_vectors_path}\n"
            f"Exécutez d'abord: python scripts/dimension_reduction.py"
        )
    item_vectors = np.load(item_vectors_path)

    # Charger les user_ids
    user_ids_path = results_dir / "user_ids_latent.npy"
    if not user_ids_path.exists():
        raise FileNotFoundError(f"user_ids introuvables: {user_ids_path}")
    user_ids = np.load(user_ids_path)

    return user_profiles, item_vectors, user_ids


def compute_similarities(
    user_profile: np.ndarray,
    item_vectors: np.ndarray
) -> np.ndarray:
    """Calcule les similarités cosinus entre un profil utilisateur et tous les items.

    Args:
        user_profile: vecteur (dimension,)
        item_vectors: matrice (n_items × dimension)

    Returns:
        similarities: array (n_items,) de scores de similarité
    """
    # Normaliser le profil utilisateur
    user_norm = np.linalg.norm(user_profile)
    if user_norm > 0:
        user_profile = user_profile / user_norm

    # Normaliser les vecteurs items
    item_norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
    item_vectors_normed = item_vectors / np.maximum(item_norms, 1e-8)

    # Calculer les similarités cosinus
    similarities = item_vectors_normed @ user_profile

    return similarities


def get_top_k_recommendations(
    user_idx: int,
    user_profiles: np.ndarray,
    item_vectors: np.ndarray,
    k: int = 10
) -> List[Tuple[int, float]]:
    """Génère les top-k recommandations pour un utilisateur.

    Args:
        user_idx: index de l'utilisateur
        user_profiles: matrice (n_users × dimension)
        item_vectors: matrice (n_items × dimension)
        k: nombre de recommandations

    Returns:
        Liste de tuples (item_idx, similarity_score)
    """
    user_profile = user_profiles[user_idx]
    similarities = compute_similarities(user_profile, item_vectors)

    # Trier par similarité décroissante et prendre les top-k
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    recommendations = [
        (int(idx), float(similarities[idx]))
        for idx in top_k_indices
    ]

    return recommendations


def print_recommendations(
    user_id: str,
    recommendations: List[Tuple[int, float]],
    variant: str
) -> None:
    """Affiche les recommandations de manière lisible."""
    print(f"\n{'='*70}")
    print(f"  RECOMMANDATIONS POUR {user_id} ({variant})")
    print(f"{'='*70}")

    for rank, (item_idx, score) in enumerate(recommendations, 1):
        print(f"  {rank:2d}. Item #{item_idx:6d} - Similarité: {score:.4f}")

    print(f"{'='*70}\n")


def main() -> None:
    """Exemple d'utilisation."""
    # Configuration
    variant = "active_pre_split"
    dimension = 100  # ou 50, 200, 300
    k = 10  # nombre de recommandations

    print(f"\nChargement des données latentes ({variant}, {dimension}D)...")

    try:
        user_profiles, item_vectors, user_ids = load_latent_data(variant, dimension)
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        return

    print(f"  Profils utilisateurs: {user_profiles.shape}")
    print(f"  Vecteurs items: {item_vectors.shape}")
    print(f"  Nombre d'utilisateurs: {len(user_ids):,}")

    # Charger le rapport pour afficher les métriques
    report_path = Path("results/svd") / variant / f"user_profile_projection_{dimension}d.json"
    if report_path.exists():
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        print(f"  Variance expliquée: {report['variance_explained_pct']:.2f}%")

    # Exemple 1: Recommandations pour le premier utilisateur
    user_idx = 0
    user_id = user_ids[user_idx]

    print(f"\nGénération de {k} recommandations pour l'utilisateur {user_id}...")
    recommendations = get_top_k_recommendations(
        user_idx=user_idx,
        user_profiles=user_profiles,
        item_vectors=item_vectors,
        k=k
    )

    print_recommendations(user_id, recommendations, variant)

    # Exemple 2: Calculer les statistiques sur toutes les similarités
    print("Calcul des statistiques de similarité pour tous les utilisateurs...")

    # Pour éviter de saturer la mémoire, on traite par batch
    batch_size = 1000
    n_users = user_profiles.shape[0]
    all_max_sims = []

    for start_idx in range(0, n_users, batch_size):
        end_idx = min(start_idx + batch_size, n_users)
        batch_profiles = user_profiles[start_idx:end_idx]

        # Normaliser
        batch_norms = np.linalg.norm(batch_profiles, axis=1, keepdims=True)
        batch_profiles_normed = batch_profiles / np.maximum(batch_norms, 1e-8)

        item_norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        item_vectors_normed = item_vectors / np.maximum(item_norms, 1e-8)

        # Similarités
        sims = batch_profiles_normed @ item_vectors_normed.T
        max_sims = sims.max(axis=1)
        all_max_sims.extend(max_sims)

    all_max_sims = np.array(all_max_sims)

    print(f"\n{'='*70}")
    print(f"  STATISTIQUES DE SIMILARITÉ")
    print(f"{'='*70}")
    print(f"  Similarité maximale moyenne: {all_max_sims.mean():.4f}")
    print(f"  Similarité maximale médiane: {np.median(all_max_sims):.4f}")
    print(f"  Similarité maximale min: {all_max_sims.min():.4f}")
    print(f"  Similarité maximale max: {all_max_sims.max():.4f}")
    print(f"  Écart-type: {all_max_sims.std():.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
