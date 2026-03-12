"""
precursor.py — Échantillonnage GPU (RAPIDS) pour le pipeline de recommandation.

Ce module fournit deux stratégies d'échantillonnage accélérées GPU :
  - sample_active_users_gpu : utilisateurs actifs (≥ N reviews), puis échantillon aléatoire.
  - sample_temporal_gpu     : fenêtre temporelle (ex. 2020–2023), puis utilisateurs actifs.

Nécessite cudf, cupy, rmm (RAPIDS). Si absents, les fonctions retournent None et ne font rien.
Usage en script : python precursor.py
Usage en module : from precursor import sample_active_users_gpu, sample_temporal_gpu
"""

from __future__ import annotations

import gc
import glob
import os
import time

import pandas as pd
import pynvml

# RAPIDS (optionnel) — cudf, cupy, rmm pour l'échantillonnage GPU
RAPIDS_AVAILABLE = False
cp = None
try:
    import cudf
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    managed_mr = rmm.mr.ManagedMemoryResource()
    limited_mr = rmm.mr.LimitingResourceAdaptor(
        managed_mr,
        allocation_limit=50 * (1024**3),
    )
    pool_mr = rmm.mr.PoolMemoryResource(
        limited_mr,
        initial_pool_size=2 * (1024**3),    # start with 2 GB
        maximum_pool_size=50 * (1024**3),    # grow up to 80 GB
    )
    rmm.mr.set_current_device_resource(pool_mr)

    cp.cuda.set_allocator(rmm_cupy_allocator)

    RAPIDS_AVAILABLE = True
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Configuration (chemins et constantes)
# -----------------------------------------------------------------------------

# Répertoires de données
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
RAW_JSONL_GLOB = f"{RAW_DATA_DIR}jsonl/*.jsonl"
RAW_PARQUET_GLOB = f"{RAW_DATA_DIR}parquet/*.parquet"
RAW_BOOKS_PATH = f"{RAW_DATA_DIR}parquet/Books.parquet"
RAW_JSONL_PATHS = sorted(glob.glob(RAW_JSONL_GLOB))
RAW_PARQUET_PATHS = sorted(glob.glob(RAW_PARQUET_GLOB))

# Conversion JSONL → Parquet (colonnes à traiter comme numériques malgré valeurs texte)
CANDIDATE_NUMERIC_COLUMNS = [
    "rating", "helpful_vote",
    "average_rating", "rating_number", "price",
]
N_SPOT_CHECK = 100_000

# Échantillonnage (stratégie et tailles)
MIN_REVIEWS = 20
NUM_USERS = 50_000
SEED = 42
CHUNK_SIZE = 2_000_000
TARGET_YEARS = [2020, 2021, 2022, 2023]
TARGET_TOTAL = 2_000_000

# Noms de fichiers et sous-dossiers pour les échantillons et splits
SAMPLE_GLOB_ORIGINAL = f"{PROCESSED_DATA_DIR}sample-*/*_original.parquet"
SAMPLE_GLOB_CLEANED = f"{PROCESSED_DATA_DIR}sample-*/*_cleaned.parquet"
SAMPLE_GLOB_FILTERED = f"{PROCESSED_DATA_DIR}sample-*/*_filtered.parquet"
SPLIT_SUBDIR = "splits"
TRAIN_FILENAME = "train.parquet"
TEST_FILENAME = "test.parquet"

# Nettoyage et filtrage (plages de dates, seuils utilisateur/item)
MIN_DATE = pd.Timestamp("1995-07-01")
MAX_DATE = pd.Timestamp("2025-12-31")
MIN_RATINGS_USER = 20
MIN_RATINGS_BOOK = 5
MAX_ITER = 20

# Split train / test
TRAIN_RATIO = 0.80
TEST_RATIO = 1.0 - TRAIN_RATIO


# -----------------------------------------------------------------------------
# Mémoire : helpers publics (utilisables par le notebook ou d'autres scripts)
# -----------------------------------------------------------------------------

def flush_ram() -> None:
    """Lance un passage du garbage collector pour libérer les objets non référencés (RAM)."""
    gc.collect()


def flush_gpu() -> None:
    """Libère les blocs GPU en cache (CuPy) et synchronise. À appeler après de grosses opérations GPU."""
    gc.collect()
    if RAPIDS_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Mémoire et diagnostic GPU (usage interne aux fonctions d'échantillonnage)
# -----------------------------------------------------------------------------

def _flush_memory() -> None:
    """Libération agressive RAM + pools CuPy (usage interne)."""
    gc.collect()
    if RAPIDS_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


def _print_gpu_status(label: str = "") -> None:
    """Affiche l'utilisation GPU (NVIDIA) via pynvml (usage interne)."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"  [{label}] GPU: {info.used/1e9:.2f}/{info.total/1e9:.2f} GB (free: {info.free/1e9:.2f} GB)")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Échantillonnage : utilisateurs actifs (≥ min_reviews), puis N utilisateurs
# -----------------------------------------------------------------------------

def sample_active_users_gpu(
    parquet_path: str,
    output_path: str,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    seed: int = SEED,
    verbose: bool = True,
) -> int | None:
    """
    Échantillonne des utilisateurs « actifs » (≥ min_reviews) puis écrit un sous-ensemble
    de leurs reviews en Parquet. Tout le travail est fait sur GPU ; l'écriture se fait
    GPU → disque (pas de grosse copie en RAM).

    Args:
        parquet_path: Fichier Parquet source (ex. Books.parquet).
        output_path: Fichier Parquet de sortie.
        min_reviews: Seuil minimum de reviews par utilisateur pour être considéré actif.
        num_users: Nombre d'utilisateurs à échantillonner.
        seed: Graine aléatoire pour la reproductibilité.
        verbose: Afficher les phases et le statut GPU.

    Returns:
        Nombre de reviews écrites, ou None si RAPIDS n'est pas disponible.
    """

    flush_ram()
    flush_gpu()


    if verbose:
        _print_gpu_status("Before start")
    start = time.time()

    # Phase 1 : chargement et identification des utilisateurs actifs
    if verbose:
        print("Phase 1 : Chargement en mémoire GPU...")
    gdf = cudf.read_parquet(parquet_path)
    gdf["rating"] = gdf["rating"].astype("int8")
    if verbose:
        _print_gpu_status("After load")
        print(f"  GPU DataFrame: {gdf.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    user_counts = gdf["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_reviews].index   

    cp.random.seed(seed)
    n_to_sample = min(num_users, len(active_users))
    indices = cp.random.choice(len(active_users), size=n_to_sample, replace=False)
    active_users_series = active_users.to_series().reset_index(drop=True)
    selected_users = active_users_series.iloc[cp.asnumpy(indices)]
    del user_counts, active_users

    if verbose:
        print(f"  Active users: {len(selected_users):,}")

    _flush_memory()
    if verbose:
        _print_gpu_status("After count flush")

    # Phase 2 : filtrage des reviews pour les utilisateurs échantillonnés
    if verbose:
        print("\nPhase 2 : Filtrage...")
    selected_series = cudf.Series(selected_users)
    del selected_users
    
    mask = gdf["user_id"].isin(selected_series)
    sample_gdf = gdf[mask]
    if verbose:
        print(f"  Reviews correspondantes : {len(sample_gdf):,}")
    del gdf, mask, selected_series
    _flush_memory()
    if verbose:
        _print_gpu_status("After filter flush")

    # Phase 3 : écriture directe GPU → Parquet (sans passer par un gros DataFrame en RAM)
    if verbose:
        print("\nPhase 3 : Sauvegarde Parquet (GPU → disque)...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_gdf.to_parquet(output_path, compression="snappy")
    n_reviews = len(sample_gdf)
    n_users = sample_gdf["user_id"].nunique()
    del sample_gdf

    _flush_memory()

    if verbose:
        elapsed = time.time() - start
        print(f"  Temps: {elapsed:.2f}s — Reviews: {n_reviews:,} — Utilisateurs: {n_users:,}")
        _print_gpu_status("Final cleanup")
    return n_reviews


# -----------------------------------------------------------------------------
# Échantillonnage : temporel (fenêtre d'années, puis utilisateurs actifs)
# -----------------------------------------------------------------------------

def sample_temporal_gpu(
    parquet_path: str,
    output_path: str,
    target_years: list[int] | None = None,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    seed: int = SEED,
    verbose: bool = True,
) -> int | None:
    """
    Échantillonne dans une fenêtre temporelle (ex. 2020–2023) : on garde les reviews
    dans ces années, on compte les reviews par utilisateur dans la période, on garde
    les utilisateurs avec ≥ min_reviews, on en tire num_users au hasard, puis on
    écrit toutes leurs reviews de la période en Parquet (GPU → disque).

    Args:
        parquet_path: Fichier Parquet source.
        output_path: Fichier Parquet de sortie.
        target_years: Liste des années à garder (défaut : TARGET_YEARS).
        min_reviews: Seuil minimum de reviews dans la période.
        num_users: Nombre d'utilisateurs à échantillonner.
        seed: Graine aléatoire.
        verbose: Afficher les étapes et le statut GPU.

    Returns:
        Nombre de reviews écrites, ou None si RAPIDS n'est pas disponible.
    """

    if target_years is None:
        target_years = TARGET_YEARS

    flush_ram()
    flush_gpu()



    start = time.time()
    if verbose:
        print("Chargement en mémoire GPU...")
    _flush_memory()
    if verbose:
        _print_gpu_status("Load start")

    gdf = cudf.read_parquet(parquet_path)
    gdf["rating"] = gdf["rating"].astype("int8")
    gdf["timestamp"] = cudf.to_datetime(gdf["timestamp"], unit="ms")
    gdf["year"] = gdf["timestamp"].dt.year
    if verbose:
        print(f"  Mémoire GPU : {gdf.memory_usage(deep=True).sum() / 1e9:.2f} Go")

    # Statistiques par année (affichage)
    gdf_period = gdf[gdf["year"].isin(target_years)]
    del gdf

    year_stats = (
        gdf_period.groupby("year")
        .agg({"user_id": ["count", "nunique"], "rating": "mean"})
    )
    ys = year_stats.to_pandas()
    del year_stats

    ys.columns = ["review_count", "unique_users", "avg_rating"]
    ys = ys.sort_index()
    if verbose:
        print(f"\n── Répartition {target_years[0]}–{target_years[-1]} ──")
        print(ys.to_string())
        print(f"\nTotal reviews : {ys['review_count'].sum():,}")

    del ys

    if verbose:
        print(f"  Reviews dans période : {len(gdf_period):,}")

    user_counts = gdf_period["user_id"].value_counts().reset_index()
    user_counts.columns = ["user_id", "review_count"]
    active_in_period = user_counts[user_counts["review_count"] >= min_reviews]
    del user_counts
    if verbose:
        print(f"Active users (>= {min_reviews} reviews): {len(active_in_period):,}")

    active_user_ids = active_in_period["user_id"].reset_index(drop=True)
    del active_in_period

    cp.random.seed(seed)
    n_to_sample = min(num_users, len(active_user_ids))
    indices = cp.random.choice(len(active_user_ids), size=n_to_sample, replace=False)
    sampled_series = active_user_ids.iloc[cp.asnumpy(indices)]
    del active_user_ids
    
    sample_gdf = gdf_period[gdf_period["user_id"].isin(sampled_series)]
    if verbose:
        print(f"  Reviews : {len(sample_gdf):,} — Utilisateurs : {sample_gdf['user_id'].nunique():,}")
    del gdf_period, sampled_series

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_gdf.to_parquet(output_path, compression="snappy")
    n_reviews = len(sample_gdf)
    del sample_gdf
    _flush_memory()

    if verbose:
        print(f"\n⚡ Temps: {time.time() - start:.2f}s — Reviews écrites: {n_reviews:,}")
    return n_reviews


