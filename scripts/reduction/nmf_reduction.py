"""
Tâche 2 - Réduction de dimension avec NMF.

Script dédié à l'entraînement de NMF sur les matrices TF-IDF déjà générées.
Il réutilise le pipeline commun de `dimension_reduction.py` afin de produire
les mêmes artéfacts que la SVD, mais avec la méthode `nmf`.
"""

from __future__ import annotations

import time
from pathlib import Path

from dimension_reduction import (
    CLEAN_DATASETS_PATHS,
    LATENT_DIMENSIONS,
    run_dimension_reduction_pipeline,
)


def main() -> None:
    t0 = time.perf_counter()

    print("\n" + "=" * 70)
    print("  TÂCHE 2 - RÉDUCTION DE DIMENSION AVEC NMF")
    print("=" * 70)

    if not CLEAN_DATASETS_PATHS:
        print("\n[ERROR] Aucun dataset trouvé dans data/joining/")
        print("Exécutez d'abord item_representation.py pour générer les matrices TF-IDF")
        return

    reports = []
    for clean_path in CLEAN_DATASETS_PATHS:
        variant_name = clean_path.stem.replace("_clean_joined", "")
        variant_dir = clean_path.parent / variant_name

        if not variant_dir.exists() or not (variant_dir / "books_representation_sparse.npz").exists():
            print(f"\n[SKIP] {variant_name} - Matrice TF-IDF manquante")
            continue

        try:
            report = run_dimension_reduction_pipeline(
                variant_dir=variant_dir,
                method="nmf",
                dimensions=LATENT_DIMENSIONS,
                force=True,
                verbose=True,
            )
            reports.append(report)
        except Exception as exc:
            print(f"\n[ERROR] {variant_name}: {exc}")

    print(f"\n{'=' * 70}")
    print(f"  Pipeline NMF terminé en {time.perf_counter() - t0:.1f}s")
    print(f"  {len(reports)} variant(s) traité(s)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
