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
from typing import Dict

import numpy as np


DATA_DIR = Path("data/joining")
RESULTS_DIR = Path("results/svd")


def validate_projection(variant: str, data_dir: Path, results_dir: Path) -> Dict[str, bool]:
    """Valide la projection des profils utilisateurs pour un variant."""
    checks = {}

    print(f"\n{'='*70}")
    print(f"VALIDATION: {variant}")
    print(f"{'='*70}")

    # 1. Vérifier que le rapport de projection existe
    report_path = results_dir / "user_profile_projection_report.json"
    if not report_path.exists():
        print(f"Rapport de projection introuvable: {report_path}")
        checks["report_exists"] = False
        return checks

    checks["report_exists"] = True
    print("Rapport de projection trouvé : OK.")

    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    # 2. Vérifier que les contraintes expérimentales sont satisfaites
    constraints = report.get("constraints_satisfied", {})

    # Interpréter explicitement les contraintes comme des booléens
    same_space = bool(constraints.get("same_vector_space", False))
    checks["same_vector_space"] = same_space
    print(f"Même espace vectoriel: {'OK' if same_space else 'ERREUR'}")

    # Validation plus stricte: pas de données de test utilisées
    no_test_raw = constraints.get("no_test_data_used")
    if isinstance(no_test_raw, bool):
        no_test_flag = no_test_raw
    else:
        no_test_flag = str(no_test_raw).strip().lower() in {"true", "yes", "1", "ok", "passed"}

    train_path = report.get("train_path") or ""
    expected_train_filename = "train_interactions.parquet"
    train_path_obj = Path(train_path) if train_path else None
    train_path_exists = bool(train_path_obj and train_path_obj.exists())
    uses_expected_train = bool(train_path_obj) and train_path_obj.name == expected_train_filename and train_path_exists
    # On considère que la contrainte "pas de données de test" est satisfaite
    # seulement si le flag est à True, que le chemin pointe vers le bon fichier
    # *et* que ce fichier existe réellement.
    checks["no_test_data"] = no_test_flag and uses_expected_train
    print(
        "Pas de données de test: "
        f"{'Oui' if checks['no_test_data'] else 'Non'} "
        f"(no_test_data_used={no_test_raw!r}, train_path={train_path!r}, "
        f"train_exists={train_path_exists})"
    )

    # Vérifier que la projection utilisateur est cohérente avec celle des items
    consistent_raw = constraints.get("consistent_with_items")
    if isinstance(consistent_raw, bool):
        consistent_flag = consistent_raw
    else:
        consistent_flag = str(consistent_raw).strip().lower() in {"true", "yes", "1", "ok", "passed"}

    checks["consistent_projection"] = consistent_flag
    print(
        f"Projection cohérente avec les items: {consistent_flag} "
        f"(valeur brute: {consistent_raw!r})"
    )
    # 3. Vérifier chaque dimension
    dimensions = report.get("dimensions_tested", [])
    print(f"\n  Dimensions testées: {dimensions}")

    for dim in dimensions:
        print(f"\n  --- Dimension {dim}D ---")

        # Profils utilisateurs
        user_profiles_path = data_dir / f"user_profiles_latent_{dim}d.npy"
        if not user_profiles_path.exists():
            print(f"Profils utilisateurs introuvables: {user_profiles_path.name}")
            checks[f"user_profiles_{dim}d"] = False
            continue

        user_profiles = np.load(user_profiles_path)
        print(f"Profils utilisateurs: {user_profiles.shape}")
        checks[f"user_profiles_{dim}d"] = True

        # Vecteurs items
        item_vectors_path = data_dir / f"items_reduced_svd_{dim}d.npy"
        if not item_vectors_path.exists():
            print(f"Vecteurs items introuvables: {item_vectors_path.name}")
            checks[f"item_vectors_{dim}d"] = False
            continue

        item_vectors = np.load(item_vectors_path)
        print(f"Vecteurs items: {item_vectors.shape}")
        checks[f"item_vectors_{dim}d"] = True

        # Vérifier les dimensions
        if user_profiles.shape[1] != item_vectors.shape[1]:
            print(f"Dimensions incompatibles: users={user_profiles.shape[1]}, "
                  f"items={item_vectors.shape[1]}")
            checks[f"dimensions_match_{dim}d"] = False
        else:
            print(f"Dimensions compatibles: {user_profiles.shape[1]}")
            checks[f"dimensions_match_{dim}d"] = True

        # Vérifier les types et valeurs
        if user_profiles.dtype != np.float32:
            print(f"Type utilisateurs: {user_profiles.dtype} (attendu: float32)")

        if item_vectors.dtype != np.float32:
            print(f"Type items: {item_vectors.dtype} (attendu: float32)")

        # Vérifier qu'il n'y a pas de NaN ou Inf
        if np.any(np.isnan(user_profiles)):
            print("NaN détectés dans les profils utilisateurs")
            checks[f"no_nan_users_{dim}d"] = False
        else:
            checks[f"no_nan_users_{dim}d"] = True

        if np.any(np.isinf(user_profiles)):
            print("Inf détectés dans les profils utilisateurs")
            checks[f"no_inf_users_{dim}d"] = False
        else:
            checks[f"no_inf_users_{dim}d"] = True

    # 4. Vérifier les user_ids
    user_ids_path = data_dir / "user_ids_latent.npy"
    if not user_ids_path.exists():
        print(f"\nuser_ids introuvables: {user_ids_path.name}")
        checks["user_ids_exist"] = False
    else:
        user_ids = np.load(user_ids_path)
        print(f"\nuser_ids trouvés: {len(user_ids):,} utilisateurs")
        checks["user_ids_exist"] = True

        # Vérifier que le nombre d'utilisateurs correspond
        for dim in dimensions:
            user_profiles_path = data_dir / f"user_profiles_latent_{dim}d.npy"
            if user_profiles_path.exists():
                user_profiles = np.load(user_profiles_path)
                if len(user_ids) != user_profiles.shape[0]:
                    print(f"Nombre d'utilisateurs incompatible pour {dim}D: "
                          f"{len(user_ids)} user_ids vs {user_profiles.shape[0]} profils")
                    checks[f"user_count_match_{dim}d"] = False
                else:
                    checks[f"user_count_match_{dim}d"] = True

    # Résumé
    print(f"\n{'='*70}")
    total = len(checks)
    passed = sum(checks.values())
    print(f"RÉSULTAT: {passed}/{total} validations passées")
    print(f"{'='*70}\n")

    return checks


def main() -> None:
    """Valide tous les variants disponibles."""
    all_checks = {}

    for variant_results_dir in sorted(RESULTS_DIR.glob("*")):
        if not variant_results_dir.is_dir():
            continue

        if variant_results_dir.name in [".", ".."]:
            continue

        variant = variant_results_dir.name
        variant_data_dir = DATA_DIR / variant
        checks = validate_projection(variant, variant_data_dir, variant_results_dir)
        all_checks[variant] = checks

    # Résumé global
    print("\n" + "=" * 70)
    print("RÉSUMÉ GLOBAL")
    print("=" * 70)

    for variant, checks in all_checks.items():
        total = len(checks)
        passed = sum(checks.values())
        status = "Valide" if passed == total else "Non valide!"
        print(f"{status} {variant}: {passed}/{total} validations passées")

    print("=" * 70 + "\n")

    # Échec si au moins une validation a échoué
    all_passed = all(all(checks.values()) for checks in all_checks.values())
    if not all_passed:
        print("Certaines validations ont échoué")
        raise SystemExit(1)
    else:
        print("Toutes les validations ont réussi")


if __name__ == "__main__":
    main()
