"""
Tâche 2 - Réduction de dimension avec PCA.

PCA exige une matrice dense. Ce script vérifie d'abord si la conversion dense
complète est raisonnable en mémoire; sinon il documente explicitement que
l'exécution doit être remplacée par le benchmark contrôlé.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dimension_reduction import (
    CLEAN_DATASETS_PATHS,
    LATENT_DIMENSIONS,
    MAX_PCA_DENSE_MEMORY_GB,
    estimate_dense_memory_gb,
    run_dimension_reduction_pipeline,
)


def save_pca_feasibility_report(
    variant_dir: Path,
    estimated_dense_memory_gb: float,
    memory_budget_gb: float,
    executed: bool,
    reason: str,
) -> None:
    report = {
        "method": "pca",
        "executed_full_pipeline": executed,
        "estimated_dense_memory_gb": round(estimated_dense_memory_gb, 4),
        "memory_budget_gb": memory_budget_gb,
        "reason": reason,
        "recommended_alternative": "python scripts/compare_reduction_methods.py",
        "dimensions_requested": LATENT_DIMENSIONS,
    }
    out_path = variant_dir / "pca_feasibility_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def get_variant_shape(variant_dir: Path) -> tuple[int, int]:
    """Récupère la shape depuis le rapport SVD existant, sinon charge la matrice."""
    comparison_path = variant_dir / "dimension_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        results = report.get("comparison_results", [])
        if results and "input_shape" in results[0]:
            shape = results[0]["input_shape"]
            return int(shape[0]), int(shape[1])

    import scipy.sparse as sp

    sparse_path = variant_dir / "books_representation_sparse.npz"
    matrix = sp.load_npz(sparse_path)
    return int(matrix.shape[0]), int(matrix.shape[1])


def main() -> None:
    t0 = time.perf_counter()

    print("\n" + "=" * 70)
    print("  TÂCHE 2 - RÉDUCTION DE DIMENSION AVEC PCA")
    print("=" * 70)
    print(
        f"\nBudget mémoire maximal pour PCA dense: {MAX_PCA_DENSE_MEMORY_GB:.1f} GB"
    )

    if not CLEAN_DATASETS_PATHS:
        print("\n[ERROR] Aucun dataset trouvé dans data/joining/")
        print("Exécutez d'abord item_representation.py pour générer les matrices TF-IDF")
        return

    reports = []
    for clean_path in CLEAN_DATASETS_PATHS:
        variant_name = clean_path.stem.replace("_clean_joined", "")
        variant_dir = clean_path.parent / variant_name
        sparse_path = variant_dir / "books_representation_sparse.npz"

        if not variant_dir.exists() or not sparse_path.exists():
            print(f"\n[SKIP] {variant_name} - Matrice TF-IDF manquante")
            continue

        matrix_shape = get_variant_shape(variant_dir)
        estimated_dense_memory_gb = estimate_dense_memory_gb(matrix_shape, dtype_bytes=4)

        if estimated_dense_memory_gb > MAX_PCA_DENSE_MEMORY_GB:
            reason = (
                "Conversion dense complète trop coûteuse pour ce variant; "
                "utiliser le benchmark comparatif sur sous-ensemble."
            )
            print(
                f"\n[SKIP] {variant_name} - PCA dense estimée à "
                f"{estimated_dense_memory_gb:.2f} GB"
            )
            save_pca_feasibility_report(
                variant_dir=variant_dir,
                estimated_dense_memory_gb=estimated_dense_memory_gb,
                memory_budget_gb=MAX_PCA_DENSE_MEMORY_GB,
                executed=False,
                reason=reason,
            )
            continue

        try:
            report = run_dimension_reduction_pipeline(
                variant_dir=variant_dir,
                method="pca",
                dimensions=LATENT_DIMENSIONS,
                force=True,
                verbose=True,
            )
            save_pca_feasibility_report(
                variant_dir=variant_dir,
                estimated_dense_memory_gb=estimated_dense_memory_gb,
                memory_budget_gb=MAX_PCA_DENSE_MEMORY_GB,
                executed=True,
                reason="PCA complet exécuté sur la matrice dense.",
            )
            reports.append(report)
        except Exception as exc:
            reason = f"Échec pendant l'exécution PCA: {exc}"
            print(f"\n[ERROR] {variant_name}: {exc}")
            save_pca_feasibility_report(
                variant_dir=variant_dir,
                estimated_dense_memory_gb=estimated_dense_memory_gb,
                memory_budget_gb=MAX_PCA_DENSE_MEMORY_GB,
                executed=False,
                reason=reason,
            )

    print(f"\n{'=' * 70}")
    print(f"  Pipeline PCA terminé en {time.perf_counter() - t0:.1f}s")
    print(f"  {len(reports)} variant(s) traité(s)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
