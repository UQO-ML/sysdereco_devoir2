"""
main.py - Orchestrateur complet du pipeline de préparation des données.

Exécute séquentiellement :
  1. Échantillonnage des utilisateurs actifs  (GPU si disponible, sinon CPU)
  2. Échantillonnage temporel                 (GPU si disponible, sinon CPU)
  3. Nettoyage des échantillons               (CPU / PyArrow)
  4. Filtrage itératif par seuils d'activité  (CPU / PyArrow)
  5. Split train/test + matrices CSR + sauvegarde  (CPU / PyArrow)

Usage : python main.py
"""

import gc
import os
import time
import sys

from queue import Empty

from scripts.precursor import (
    RAPIDS_AVAILABLE,
    RAW_BOOKS_PATH,
    SAMPLE_ACTIVE_DIR,
    SAMPLE_TEMPORAL_DIR,
    SAMPLE_GLOB_FILTERED,
    TARGET_YEARS,
    # Dataset preparation (JSONL  to Parquet)
    jsonl_to_parquet_conversion,
    resolve_glob,
    # GPU sampling
    sample_active_users_gpu,
    sample_temporal_gpu,
    # CPU sampling (fallback)
    sample_active_users_cpu,
    sample_temporal_cpu,
    # Post-processing (CPU)
    clean_samples,
    filter_samples,
    split_and_save,
    # Memory helpers
    flush_ram,
    flush_gpu,
)


def _final_files_checker() -> bool:

    result = True
    filtered_data_paths = resolve_glob(SAMPLE_GLOB_FILTERED)
    active_splits_dir_path = f"{SAMPLE_ACTIVE_DIR}/splits/"
    temporal_splits_dir_path = f"{SAMPLE_TEMPORAL_DIR}/splits/"

    if len(filtered_data_paths) == 0:
        result = False
    for path in filtered_data_paths:
        if os.path.getsize(path) < 1024:
            result = False
    
    active_splits_dir = os.listdir(active_splits_dir_path) if os.path.isdir(active_splits_dir_path) else []
    if len(active_splits_dir) == 0:
        result = False

    temporal_splits_dir = os.listdir(temporal_splits_dir_path) if os.path.isdir(temporal_splits_dir_path) else []
    if len(temporal_splits_dir) == 0:
        result = False

    return result


def precursor():
    t_start = time.time()

    use_gpu = RAPIDS_AVAILABLE
    backend = "GPU (RAPIDS)" if use_gpu else "CPU (PyArrow)"
    print(f"Pipeline de préparation - backend : {backend}\n")

    # -- 0. Conversion Dataset ----------------------------------------
    result = False
    try:
        result = jsonl_to_parquet_conversion()
    except Exception as e:
        print(f"  ⚠ Conversion Dataset jsonl_to_parquet_conversion a échoué : {e}")
        sys.exit(1) 
               
    if result:

        # -- 1. Échantillonnage : utilisateurs actifs ---------------------

        print("=" * 70)
        print("  ÉTAPE 1/5 : Échantillonnage des utilisateurs actifs")
        print("=" * 70)

        active_out = f"{SAMPLE_ACTIVE_DIR}/active_users_original.parquet"
        if use_gpu:
            sample_active_users_gpu(RAW_BOOKS_PATH, active_out)
        else:
            sample_active_users_cpu(RAW_BOOKS_PATH, active_out)

        flush_ram()
        flush_gpu()
        gc.collect()

        # -- 2. Échantillonnage : temporel --------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 2/5 : Échantillonnage temporel")
        print("=" * 70)

        temporal_out = f"{SAMPLE_TEMPORAL_DIR}/temporal_original.parquet"
        if use_gpu:
            sample_temporal_gpu(
                RAW_BOOKS_PATH, temporal_out, target_years=TARGET_YEARS,
            )
        else:
            sample_temporal_cpu(
                RAW_BOOKS_PATH, temporal_out, target_years=TARGET_YEARS,
            )

        flush_ram()
        flush_gpu()
        gc.collect()

        # -- 3. Nettoyage -------------------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 3/5 : Nettoyage des échantillons")
        print("=" * 70)

        clean_samples()
        flush_ram()

        # -- 4. Filtrage --------------------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 4/5 : Filtrage par seuils d'activité")
        print("=" * 70)

        filter_samples()
        flush_ram()

        # -- 5. Split + sauvegarde ----------------------------------------

        print("\n" + "=" * 70)
        print("  ÉTAPE 5/5 : Split train/test + matrices CSR + sauvegarde")
        print("=" * 70)

        split_and_save()
        flush_ram()

        # -- Résumé -------------------------------------------------------

        elapsed = time.time() - t_start
        print(f"\n{'=' * 70}")
        print(f"  ✓ Pipeline complet en {elapsed:.1f}s")
        print(f"{'=' * 70}")

    else:
        print(f"  ⚠ Conversion Dataset jsonl_to_parquet_conversion ({e}),  : {result}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  ✓ Pipeline complet en {elapsed:.1f}s")
    print(f"{'=' * 70}")

def main():
    final_files_checker = False
    final_files_checker = _final_files_checker()
    print(f"final_files_checker : {final_files_checker}")
    if final_files_checker:
        print("Echantillon present")
    else:
        precursor()


if __name__ == "__main__":
    main()
