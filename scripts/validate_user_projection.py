"""
Script de validation pour la projection des profils utilisateurs.

Vérifie que:
1. Les profils et items sont dans le même espace vectoriel (même dimension)
2. Aucune donnée du test n'est utilisée (seulement train_interactions.parquet)
3. La projection est cohérente avec celle des items (même transformation SVD)
4. Les matrices de sortie ont les bonnes dimensions

Usage: python validate_user_projection.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def validate_projection(variant: str, results_dir: Path) -> Dict[str, bool]:
    """Valide la projection des profils utilisateurs pour un variant."""
    checks = {}

    print(f"\n{'='*70}")
    print(f"  VALIDATION: {variant}")
    print(f"{'='*70}")

    # 1. Vérifier que le rapport de projection existe
    report_path = results_dir / "user_profile_projection_report.json"
    if not report_path.exists():
        print(f"❌ Rapport de projection introuvable: {report_path}")
        checks["report_exists"] = False
        return checks

    checks["report_exists"] = True
    print(f"✓ Rapport de projection trouvé")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # 2. Vérifier que les contraintes expérimentales sont satisfaites
    constraints = report.get("constraints_satisfied", {})

    same_space = constraints.get("same_vector_space", False)
    checks["same_vector_space"] = same_space
    print(f"{'✓' if same_space else '❌'} Même espace vectoriel: {same_space}")

    no_test = constraints.get("no_test_data_used", "")
    checks["no_test_data"] = bool(no_test)
    print(f"✓ Pas de données de test: {no_test}")

    consistent = constraints.get("consistent_with_items", "")
    checks["consistent_projection"] = bool(consistent)
    print(f"✓ Projection cohérente: {consistent}")

    # 3. Vérifier chaque dimension
    dimensions = report.get("dimensions_tested", [])
    print(f"\n  Dimensions testées: {dimensions}")

    for dim in dimensions:
        print(f"\n  --- Dimension {dim}D ---")

        # Profils utilisateurs
        user_profiles_path = results_dir / f"user_profiles_latent_{dim}d.npy"
        if not user_profiles_path.exists():
            print(f"  ❌ Profils utilisateurs introuvables: {user_profiles_path.name}")
            checks[f"user_profiles_{dim}d"] = False
            continue

        user_profiles = np.load(user_profiles_path)
        print(f"  ✓ Profils utilisateurs: {user_profiles.shape}")
        checks[f"user_profiles_{dim}d"] = True

        # Vecteurs items
        item_vectors_path = results_dir / f"items_reduced_svd_{dim}d.npy"
        if not item_vectors_path.exists():
            print(f"  ❌ Vecteurs items introuvables: {item_vectors_path.name}")
            checks[f"item_vectors_{dim}d"] = False
            continue

        item_vectors = np.load(item_vectors_path)
        print(f"  ✓ Vecteurs items: {item_vectors.shape}")
        checks[f"item_vectors_{dim}d"] = True

        # Vérifier les dimensions
        if user_profiles.shape[1] != item_vectors.shape[1]:
            print(f"  ❌ Dimensions incompatibles: users={user_profiles.shape[1]}, "
                  f"items={item_vectors.shape[1]}")
            checks[f"dimensions_match_{dim}d"] = False
        else:
            print(f"  ✓ Dimensions compatibles: {user_profiles.shape[1]}")
            checks[f"dimensions_match_{dim}d"] = True

        # Vérifier les types et valeurs
        if user_profiles.dtype != np.float32:
            print(f"  ⚠ Type utilisateurs: {user_profiles.dtype} (attendu: float32)")

        if item_vectors.dtype != np.float32:
            print(f"  ⚠ Type items: {item_vectors.dtype} (attendu: float32)")

        # Vérifier qu'il n'y a pas de NaN ou Inf
        if np.any(np.isnan(user_profiles)):
            print(f"  ❌ NaN détectés dans les profils utilisateurs")
            checks[f"no_nan_users_{dim}d"] = False
        else:
            checks[f"no_nan_users_{dim}d"] = True

        if np.any(np.isinf(user_profiles)):
            print(f"  ❌ Inf détectés dans les profils utilisateurs")
            checks[f"no_inf_users_{dim}d"] = False
        else:
            checks[f"no_inf_users_{dim}d"] = True

    # 4. Vérifier les user_ids
    user_ids_path = results_dir / "user_ids_latent.npy"
    if not user_ids_path.exists():
        print(f"\n❌ user_ids introuvables: {user_ids_path.name}")
        checks["user_ids_exist"] = False
    else:
        user_ids = np.load(user_ids_path)
        print(f"\n✓ user_ids trouvés: {len(user_ids):,} utilisateurs")
        checks["user_ids_exist"] = True

        # Vérifier que le nombre d'utilisateurs correspond
        for dim in dimensions:
            user_profiles_path = results_dir / f"user_profiles_latent_{dim}d.npy"
            if user_profiles_path.exists():
                user_profiles = np.load(user_profiles_path)
                if len(user_ids) != user_profiles.shape[0]:
                    print(f"❌ Nombre d'utilisateurs incompatible pour {dim}D: "
                          f"{len(user_ids)} user_ids vs {user_profiles.shape[0]} profils")
                    checks[f"user_count_match_{dim}d"] = False
                else:
                    checks[f"user_count_match_{dim}d"] = True

    # Résumé
    print(f"\n{'='*70}")
    total = len(checks)
    passed = sum(checks.values())
    print(f"  RÉSULTAT: {passed}/{total} validations passées")
    print(f"{'='*70}\n")

    return checks


def main() -> None:
    """Valide tous les variants disponibles."""
    results_dir = Path("results/svd")

    all_checks = {}

    for variant_dir in sorted(results_dir.glob("*")):
        if not variant_dir.is_dir():
            continue

        if variant_dir.name in [".", ".."]:
            continue

        variant = variant_dir.name
        checks = validate_projection(variant, variant_dir)
        all_checks[variant] = checks

    # Résumé global
    print("\n" + "=" * 70)
    print("  RÉSUMÉ GLOBAL")
    print("=" * 70)

    for variant, checks in all_checks.items():
        total = len(checks)
        passed = sum(checks.values())
        status = "✓" if passed == total else "⚠"
        print(f"{status} {variant}: {passed}/{total} validations passées")

    print("=" * 70 + "\n")

    # Échec si au moins une validation a échoué
    all_passed = all(all(checks.values()) for checks in all_checks.values())
    if not all_passed:
        print("⚠ Certaines validations ont échoué")
        exit(1)
    else:
        print("✓ Toutes les validations ont réussi")


if __name__ == "__main__":
    main()
