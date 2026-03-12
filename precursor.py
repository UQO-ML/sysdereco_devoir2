"""
precursor.py — Échantillonnage et prétraitement pour le pipeline de recommandation.

Ce module fournit le pipeline complet de préparation des données :

  Échantillonnage (GPU ou CPU) :
    - sample_active_users_gpu / sample_active_users_cpu
    - sample_temporal_gpu     / sample_temporal_cpu

  Post-traitement (CPU, pandas) :
    - clean_samples       : nettoyage des ratings et timestamps invalides
    - filter_samples      : filtrage itératif par seuils d'activité (user/item)
    - split_and_save      : split train/test stratifié + matrices CSR + sauvegarde

Les variantes GPU nécessitent cudf, cupy, rmm (RAPIDS).
Les variantes CPU et le post-traitement fonctionnent avec pandas seul.

Usage en script : python precursor.py
Usage en module : from precursor import sample_active_users_gpu, clean_samples, ...
"""

from __future__ import annotations

import gc
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

try:
    import pynvml
except ImportError:
    pynvml = None

# ---------------------------------------------------------------------------
# RAPIDS (optionnel) — cudf, cupy, rmm pour l'échantillonnage GPU
# Le bloc try/except garantit que l'import ne casse rien sur une machine CPU.
# Le pool RMM est initialisé une seule fois au chargement du module avec un
# plafond de 50 Go : CUDA UVM spille en RAM quand la VRAM est pleine, et
# LimitingResourceAdaptor empêche de consommer toute la RAM système.
# ---------------------------------------------------------------------------

RAPIDS_AVAILABLE = False
cp = None
try:
    import cudf
    import cupy as cp
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    _MANAGED_MEMORY_CAP = 50 * (1024**3)

    managed_mr = rmm.mr.ManagedMemoryResource()
    limited_mr = rmm.mr.LimitingResourceAdaptor(
        managed_mr,
        allocation_limit=_MANAGED_MEMORY_CAP,
    )
    pool_mr = rmm.mr.PoolMemoryResource(
        limited_mr,
        initial_pool_size=2 * (1024**3),
        maximum_pool_size=_MANAGED_MEMORY_CAP,
    )
    rmm.mr.set_current_device_resource(pool_mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    RAPIDS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# Configuration (chemins et constantes)
# =============================================================================

# Répertoires de données
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
RAW_JSONL_GLOB = f"{RAW_DATA_DIR}jsonl/*.jsonl"
RAW_PARQUET_GLOB = f"{RAW_DATA_DIR}parquet/*.parquet"
RAW_BOOKS_PATH = f"{RAW_DATA_DIR}parquet/Books.parquet"
RAW_JSONL_PATHS = sorted(glob.glob(RAW_JSONL_GLOB))
RAW_PARQUET_PATHS = sorted(glob.glob(RAW_PARQUET_GLOB))

# Colonnes pouvant contenir des valeurs non numériques dans le JSONL Amazon
CANDIDATE_NUMERIC_COLUMNS = [
    "rating", "helpful_vote",
    "average_rating", "rating_number", "price",
]
N_SPOT_CHECK = 100_000

# Échantillonnage
MIN_REVIEWS = 20
NUM_USERS = 50_000
SEED = 42
CHUNK_SIZE = 2_000_000
TARGET_YEARS = [2020, 2021, 2022, 2023]
TARGET_TOTAL = 2_000_000

# Répertoires de sortie pour chaque stratégie d'échantillonnage.
# La convention sample-*/ permet aux globs de post-traitement de trouver
# automatiquement tous les échantillons.
SAMPLE_ACTIVE_DIR = f"{PROCESSED_DATA_DIR}sample-active-users"
SAMPLE_TEMPORAL_DIR = f"{PROCESSED_DATA_DIR}sample-temporal"

# Globs pour le post-traitement (résolus à l'exécution via _resolve_glob())
SAMPLE_GLOB_ORIGINAL = f"{PROCESSED_DATA_DIR}sample-*/*_original.parquet"
SAMPLE_GLOB_CLEANED = f"{PROCESSED_DATA_DIR}sample-*/*_cleaned.parquet"
SAMPLE_GLOB_FILTERED = f"{PROCESSED_DATA_DIR}sample-*/*_filtered.parquet"
SPLIT_SUBDIR = "splits"
TRAIN_FILENAME = "train.parquet"
TEST_FILENAME = "test.parquet"

# Nettoyage et filtrage
MIN_DATE = pd.Timestamp("1995-07-01")   # fondation d'Amazon
MAX_DATE = pd.Timestamp("2025-12-31")
MIN_RATINGS_USER = 20
MIN_RATINGS_BOOK = 5
MAX_ITER = 20

# Split train / test
TRAIN_RATIO = 0.80
TEST_RATIO = 1.0 - TRAIN_RATIO


# =============================================================================
# Utilitaires internes
# =============================================================================

def _resolve_glob(pattern: str) -> list[str]:
    """Résout un glob pattern en une liste triée de chemins existants."""
    return sorted(glob.glob(pattern))


# =============================================================================
# Mémoire : helpers publics
# =============================================================================

def flush_ram() -> None:
    """Lance le garbage collector pour libérer les objets non référencés."""
    gc.collect()


def flush_gpu() -> None:
    """Libère les blocs GPU en cache (CuPy) et synchronise."""
    gc.collect()
    if RAPIDS_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
        except Exception:
            pass


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
    """Affiche l'utilisation GPU (NVIDIA) via pynvml."""
    if pynvml is None:
        return
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(
            f"  [{label}] GPU: {info.used / 1e9:.2f}/{info.total / 1e9:.2f} GB "
            f"(free: {info.free / 1e9:.2f} GB)"
        )
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _print_ram_status(label: str = "") -> None:
    """Affiche la mémoire RAM consommée par le processus courant (optionnel, via psutil)."""
    try:
        import psutil
        mem = psutil.Process().memory_info()
        print(f"  [{label}] RAM process: {mem.rss / 1e9:.2f} GB")
    except ImportError:
        pass


# =============================================================================
#  ÉCHANTILLONNAGE GPU
# =============================================================================

def sample_active_users_gpu(
    parquet_path: str,
    output_path: str,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    seed: int = SEED,
    verbose: bool = True,
) -> int | None:
    """
    Échantillonne les utilisateurs « actifs » (≥ min_reviews reviews) via GPU.

    Charge le parquet complet en VRAM (cuDF), compte les reviews par user,
    en tire num_users au hasard via CuPy, filtre leurs reviews et écrit
    le résultat en Parquet (GPU → disque, sans grosse copie en RAM).

    Returns:
        Nombre de reviews écrites, ou None si RAPIDS n'est pas disponible.
    """
    if not RAPIDS_AVAILABLE:
        if verbose:
            print("ℹ RAPIDS non disponible — sample_active_users_gpu ignoré.")
        return None

    flush_ram()
    flush_gpu()

    if verbose:
        _print_gpu_status("Before start")
    start = time.time()

    if verbose:
        print("Phase 1 : Chargement en mémoire GPU...")
    gdf = cudf.read_parquet(parquet_path)
    gdf["rating"] = gdf["rating"].astype("int8")
    if verbose:
        _print_gpu_status("After load")
        print(f"  GPU DataFrame: {gdf.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # value_counts() retourne une Series indexée par user_id, valeurs = nb de reviews
    user_counts = gdf["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_reviews].index

    # Sélection aléatoire sur GPU — seuls les 50k indices sont copiés vers le CPU
    cp.random.seed(seed)
    n_to_sample = min(num_users, len(active_users))
    indices = cp.random.choice(len(active_users), size=n_to_sample, replace=False)
    active_users_series = active_users.to_series().reset_index(drop=True)
    selected_users = active_users_series.iloc[cp.asnumpy(indices)]
    del user_counts, active_users, active_users_series, indices

    if verbose:
        print(f"  Utilisateurs actifs sélectionnés: {len(selected_users):,}")

    _flush_memory()
    if verbose:
        _print_gpu_status("After count flush")

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

    if verbose:
        print("\nPhase 3 : Sauvegarde Parquet (GPU → disque)...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_gdf.to_parquet(output_path, compression="snappy")
    n_reviews = len(sample_gdf)
    n_users_out = sample_gdf["user_id"].nunique()
    del sample_gdf
    _flush_memory()

    if verbose:
        elapsed = time.time() - start
        print(f"  Temps: {elapsed:.2f}s — Reviews: {n_reviews:,} — Utilisateurs: {n_users_out:,}")
        _print_gpu_status("Final cleanup")
    return n_reviews


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
    Échantillonnage temporel via GPU : filtre par années cibles, puis
    ne garde que les utilisateurs ayant ≥ min_reviews dans la période,
    en tire num_users au hasard et écrit toutes leurs reviews.

    Returns:
        Nombre de reviews écrites, ou None si RAPIDS n'est pas disponible.
    """
    if not RAPIDS_AVAILABLE:
        if verbose:
            print("ℹ RAPIDS non disponible — sample_temporal_gpu ignoré.")
        return None

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

    # Filtrer à la période cible et libérer immédiatement le DF complet
    gdf_period = gdf[gdf["year"].isin(target_years)]
    del gdf
    _flush_memory()

    # Statistiques par année (petit transfert vers pandas pour affichage)
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

    # Compter les reviews par utilisateur dans la période
    user_counts = gdf_period["user_id"].value_counts().reset_index()
    user_counts.columns = ["user_id", "review_count"]
    active_in_period = user_counts[user_counts["review_count"] >= min_reviews]
    del user_counts
    if verbose:
        print(f"Active users (>= {min_reviews} reviews): {len(active_in_period):,}")

    # Sélection aléatoire sur GPU
    active_user_ids = active_in_period["user_id"].reset_index(drop=True)
    del active_in_period

    cp.random.seed(seed)
    n_to_sample = min(num_users, len(active_user_ids))
    indices = cp.random.choice(len(active_user_ids), size=n_to_sample, replace=False)
    sampled_series = active_user_ids.iloc[cp.asnumpy(indices)]
    del active_user_ids, indices

    if verbose:
        print(f"  Utilisateurs échantillonnés : {n_to_sample:,}")

    sample_gdf = gdf_period[gdf_period["user_id"].isin(sampled_series)]
    if verbose:
        print(f"  Reviews : {len(sample_gdf):,} — Utilisateurs : {sample_gdf['user_id'].nunique():,}")
    del gdf_period, sampled_series
    _flush_memory()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_gdf.to_parquet(output_path, compression="snappy")
    n_reviews = len(sample_gdf)
    del sample_gdf
    _flush_memory()

    if verbose:
        print(f"\n⚡ Temps: {time.time() - start:.2f}s — Reviews écrites: {n_reviews:,}")
    return n_reviews


# =============================================================================
#  ÉCHANTILLONNAGE CPU (alternatives pandas pour machines sans GPU)
# =============================================================================

def sample_active_users_cpu(
    parquet_path: str,
    output_path: str,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    seed: int = SEED,
    verbose: bool = True,
) -> int:
    """
    Équivalent CPU (pandas) de sample_active_users_gpu.

    Même logique : charge le parquet, identifie les utilisateurs ayant
    ≥ min_reviews reviews, en tire num_users au hasard, filtre leurs
    reviews et écrit le résultat en Parquet.

    Plus lent que la variante GPU mais ne nécessite ni GPU ni RAPIDS.

    Returns:
        Nombre de reviews écrites.
    """
    flush_ram()
    start = time.time()

    if verbose:
        print("Phase 1 : Chargement en mémoire (pandas)...")
    df = pd.read_parquet(parquet_path)
    df["rating"] = df["rating"].astype("int8")
    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"  DataFrame: {mem_gb:.2f} GB — {len(df):,} lignes")
        _print_ram_status("After load")

    # Identifier les utilisateurs ayant >= min_reviews reviews
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= min_reviews].index.tolist()
    del user_counts
    if verbose:
        print(f"  Utilisateurs actifs (>= {min_reviews} reviews): {len(active_users):,}")

    # Sélection aléatoire
    random.seed(seed)
    n_to_sample = min(num_users, len(active_users))
    selected_users = set(random.sample(active_users, n_to_sample))
    del active_users
    gc.collect()

    if verbose:
        print(f"  Utilisateurs échantillonnés: {n_to_sample:,}")

    # Filtrage — .copy() pour libérer le DF original via GC
    if verbose:
        print("\nPhase 2 : Filtrage...")
    sample_df = df.loc[df["user_id"].isin(selected_users)].copy()
    del df, selected_users
    gc.collect()

    if verbose:
        print(f"  Reviews correspondantes : {len(sample_df):,}")
        _print_ram_status("After filter")

    # Écriture
    if verbose:
        print("\nPhase 3 : Sauvegarde Parquet...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_df.to_parquet(output_path, compression="snappy", index=False)
    n_reviews = len(sample_df)
    n_users_out = sample_df["user_id"].nunique()
    del sample_df
    gc.collect()

    if verbose:
        elapsed = time.time() - start
        print(f"  Temps: {elapsed:.2f}s — Reviews: {n_reviews:,} — Utilisateurs: {n_users_out:,}")
    return n_reviews


def sample_temporal_cpu(
    parquet_path: str,
    output_path: str,
    target_years: list[int] | None = None,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    seed: int = SEED,
    verbose: bool = True,
) -> int:
    """
    Équivalent CPU (pandas) de sample_temporal_gpu.

    Même logique : filtre par années cibles, ne garde que les utilisateurs
    ayant ≥ min_reviews dans la période, en tire num_users au hasard, et
    écrit toutes leurs reviews de la période.

    Returns:
        Nombre de reviews écrites.
    """
    if target_years is None:
        target_years = TARGET_YEARS

    flush_ram()
    start = time.time()

    if verbose:
        print("Chargement en mémoire (pandas)...")
    df = pd.read_parquet(parquet_path)
    df["rating"] = df["rating"].astype("int8")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["year"] = df["timestamp"].dt.year
    if verbose:
        mem_gb = df.memory_usage(deep=True).sum() / 1e9
        print(f"  DataFrame: {mem_gb:.2f} GB — {len(df):,} lignes")

    # Filtrer à la période cible, libérer le DF complet
    df_period = df.loc[df["year"].isin(target_years)].copy()
    del df
    gc.collect()

    # Statistiques par année
    year_stats = df_period.groupby("year").agg(
        review_count=("user_id", "count"),
        unique_users=("user_id", "nunique"),
        avg_rating=("rating", "mean"),
    )
    if verbose:
        print(f"\n── Répartition {target_years[0]}–{target_years[-1]} ──")
        print(year_stats.to_string())
        print(f"\nTotal reviews : {year_stats['review_count'].sum():,}")
    del year_stats

    if verbose:
        print(f"  Reviews dans période : {len(df_period):,}")

    # Compter les reviews par utilisateur dans la période
    user_counts = df_period["user_id"].value_counts()
    active_user_ids = user_counts[user_counts >= min_reviews].index.tolist()
    del user_counts
    if verbose:
        print(f"Active users (>= {min_reviews} reviews): {len(active_user_ids):,}")

    # Sélection aléatoire
    random.seed(seed)
    n_to_sample = min(num_users, len(active_user_ids))
    sampled_users = set(random.sample(active_user_ids, n_to_sample))
    del active_user_ids
    gc.collect()

    if verbose:
        print(f"  Utilisateurs échantillonnés : {n_to_sample:,}")

    sample_df = df_period.loc[df_period["user_id"].isin(sampled_users)].copy()
    del df_period, sampled_users
    gc.collect()

    if verbose:
        print(f"  Reviews : {len(sample_df):,} — Utilisateurs : {sample_df['user_id'].nunique():,}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    sample_df.to_parquet(output_path, compression="snappy", index=False)
    n_reviews = len(sample_df)
    del sample_df
    gc.collect()

    if verbose:
        print(f"\n⚡ Temps: {time.time() - start:.2f}s — Reviews écrites: {n_reviews:,}")
    return n_reviews


# =============================================================================
#  POST-TRAITEMENT — Nettoyage
# =============================================================================

def clean_samples(
    glob_pattern: str = SAMPLE_GLOB_ORIGINAL,
    verbose: bool = True,
) -> list[str]:
    """
    Nettoyage des fichiers Parquet d'échantillons (*_original.parquet).

    Trois étapes appliquées à chaque fichier :
      1. Suppression des reviews sans note valide (rating NaN ou 0)
      2. Filtrage des timestamps hors plage [1995-07-01, 2025-12-31]
         (avant Amazon = erreur de donnée ; dans le futur = aberrant)
      3. Conversion du rating en float64 pour cohérence numérique

    Toutes les conditions sont combinées en un seul masque booléen pour
    éviter de créer plusieurs copies intermédiaires du DataFrame.

    Produit : *_cleaned.parquet dans le même répertoire.

    Returns:
        Liste des chemins des fichiers nettoyés créés.
    """
    paths = _resolve_glob(glob_pattern)
    if not paths:
        print(f"  Aucun fichier trouvé pour : {glob_pattern}")
        return []

    cleaned_paths: list[str] = []

    for path in paths:
        if os.path.getsize(path) < 1024:
            if verbose:
                print(f"  ⚠ Fichier ignoré (trop petit) : {path}")
            continue

        if verbose:
            print(f"\n{'═' * 60}")
            print(f"  Nettoyage : {path}")
            print(f"{'═' * 60}")

        df = pd.read_parquet(path)
        n_original = len(df)

        if n_original == 0:
            if verbose:
                print("  ⚠ Fichier vide, passage au suivant.")
            continue

        # Construire un masque unique au lieu de filtrer en chaîne
        # (évite ~4 copies simultanées du DataFrame)
        rating_ok = df["rating"].notna() & (df["rating"] > 0)
        n_dropped_rating = int((~rating_ok).sum())

        _date = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        ts_ok = _date.between(MIN_DATE, MAX_DATE)
        n_dropped_ts = int((rating_ok & ~ts_ok).sum())

        df = df.loc[rating_ok & ts_ok].copy()
        del rating_ok, ts_ok, _date
        gc.collect()

        df["rating"] = df["rating"].astype(float)

        n_final = len(df)
        if verbose:
            print(f"  Reviews initiales     : {n_original:,}")
            print(f"  Supprimées (rating)   : {n_dropped_rating:,}")
            print(f"  Supprimées (timestamp): {n_dropped_ts:,}")
            print(f"  Reviews finales       : {n_final:,}  "
                  f"({n_final / n_original * 100:.1f}% rétention)")
            print(f"  Utilisateurs restants : {df['user_id'].nunique():,}")
            print(f"  Livres restants       : {df['parent_asin'].nunique():,}")

        out_path = path.replace("_original.parquet", "_cleaned.parquet")
        df.to_parquet(out_path, index=False)
        cleaned_paths.append(out_path)

        if verbose:
            print(f"  ✓ Sauvegardé : {out_path}")

        del df
        gc.collect()

    flush_ram()
    return cleaned_paths


# =============================================================================
#  POST-TRAITEMENT — Filtrage itératif par seuils d'activité
# =============================================================================

def filter_samples(
    glob_pattern: str = SAMPLE_GLOB_CLEANED,
    min_ratings_user: int = MIN_RATINGS_USER,
    min_ratings_book: int = MIN_RATINGS_BOOK,
    max_iter: int = MAX_ITER,
    verbose: bool = True,
) -> list[str]:
    """
    Filtrage itératif des fichiers *_cleaned.parquet.

    Élimine les utilisateurs et les livres ayant trop peu d'interactions.
    En recommandation, un utilisateur avec 1–2 notes ne fournit aucun signal
    exploitable (cold start), et un livre noté par un seul lecteur ne peut
    pas être recommandé par similarité.

    Le filtrage boucle car supprimer des livres rares peut faire descendre
    des utilisateurs sous le seuil, et inversement. On itère jusqu'à ce que
    plus aucune ligne ne soit supprimée (convergence en 2–4 itérations
    en pratique).

    .copy() est appelé après chaque filtre pour que le GC puisse libérer
    l'ancien DataFrame au lieu de le garder comme backing store d'une vue.

    Produit : *_filtered.parquet dans le même répertoire.

    Returns:
        Liste des chemins des fichiers filtrés créés.
    """
    paths = _resolve_glob(glob_pattern)
    if not paths:
        print(f"  Aucun fichier trouvé pour : {glob_pattern}")
        return []

    filtered_paths: list[str] = []

    for path in paths:
        if os.path.getsize(path) < 1024:
            if verbose:
                print(f"  ⚠ Fichier ignoré (trop petit) : {path}")
            continue

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"  Filtrage : {path}")
            print(f"{'═' * 70}")

        df = pd.read_parquet(path)
        n_before = len(df)
        u_before = df["user_id"].nunique()
        b_before = df["parent_asin"].nunique()

        if n_before == 0:
            if verbose:
                print("  ⚠ Fichier vide, passage au suivant.")
            continue

        sparsity_before = 1 - n_before / (u_before * b_before)

        if verbose:
            print(f"  AVANT : {n_before:,} reviews, "
                  f"{u_before:,} utilisateurs, {b_before:,} livres")
            print(f"  Sparsité : {sparsity_before * 100:.4f}%")
            print(f"  Seuils : utilisateur ≥ {min_ratings_user}, "
                  f"livre ≥ {min_ratings_book}")

        for iteration in range(1, max_iter + 1):
            n_start = len(df)

            book_counts = df.groupby("parent_asin").size()
            books_ok = book_counts[book_counts >= min_ratings_book].index
            df = df.loc[df["parent_asin"].isin(books_ok)].copy()
            del book_counts, books_ok

            user_counts = df.groupby("user_id").size()
            users_ok = user_counts[user_counts >= min_ratings_user].index
            df = df.loc[df["user_id"].isin(users_ok)].copy()
            del user_counts, users_ok

            gc.collect()

            total_dropped = n_start - len(df)
            if verbose:
                print(f"  Itération {iteration:>2d} : "
                      f"−{total_dropped:,} reviews → {len(df):,} restantes")

            if total_dropped == 0:
                if verbose:
                    print(f"  ✓ Convergence atteinte à l'itération {iteration}.")
                break
        else:
            if verbose:
                print(f"  ⚠ Limite de {max_iter} itérations atteinte.")

        n_after = len(df)
        u_after = df["user_id"].nunique()
        b_after = df["parent_asin"].nunique()
        sparsity_after = (
            1 - n_after / (u_after * b_after)
            if u_after > 0 and b_after > 0
            else 1.0
        )

        if verbose:
            print(f"\n  APRÈS : {n_after:,} reviews, "
                  f"{u_after:,} utilisateurs, {b_after:,} livres")
            print(f"  Sparsité : {sparsity_after * 100:.4f}%")
            pct_r = n_after / n_before * 100 if n_before else 0
            pct_u = u_after / u_before * 100 if u_before else 0
            pct_b = b_after / b_before * 100 if b_before else 0
            print(f"  Rétention : {pct_r:.1f}% reviews, "
                  f"{pct_u:.1f}% users, {pct_b:.1f}% livres")

        out_path = path.replace("_cleaned.parquet", "_filtered.parquet")
        df.to_parquet(out_path, index=False)
        filtered_paths.append(out_path)

        if verbose:
            print(f"  ✓ Sauvegardé : {out_path}")

        del df
        gc.collect()

    flush_ram()
    return filtered_paths


# =============================================================================
#  POST-TRAITEMENT — Split train/test stratifié + matrices CSR + sauvegarde
# =============================================================================

def split_and_save(
    glob_pattern: str = SAMPLE_GLOB_FILTERED,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    verbose: bool = True,
) -> list[str]:
    """
    Split train/test stratifié par utilisateur, construction des matrices CSR
    et sauvegarde complète sur disque.

    Le split est vectorisé (pas de boucle Python sur les utilisateurs) :
      1. Attribuer un nombre aléatoire à chaque rating
      2. Trier par (user_id, random) pour mélanger intra-utilisateur
      3. Calculer la position cumulative de chaque rating via cumcount()
      4. Les derniers n_test ratings de chaque user vont dans le test

    Pour chaque utilisateur on garantit au moins 1 rating dans le train
    ET dans le test (contrainte essentielle en recommandation).

    Produit dans <sample_dir>/splits/ :
      - train.parquet, test.parquet         (DataFrames)
      - R_train.npz, R_test.npz             (matrices CSR scipy)
      - user_ids.npy, item_ids.npy           (mappings indice ↔ identifiant)
      - metadata.json                        (paramètres et statistiques)

    Chaque échantillon est traité et sauvé indépendamment pour ne jamais
    accumuler plusieurs splits en mémoire simultanément.

    Returns:
        Liste des répertoires de splits créés.
    """
    test_ratio = 1.0 - train_ratio
    paths = _resolve_glob(glob_pattern)
    if not paths:
        print(f"  Aucun fichier trouvé pour : {glob_pattern}")
        return []

    split_dirs: list[str] = []

    for path in paths:
        if os.path.getsize(path) < 1024:
            continue

        if verbose:
            print(f"\n{'═' * 70}")
            print(f"  Split Train/Test : {path}")
            print(f"{'═' * 70}")

        df = pd.read_parquet(path)
        if len(df) == 0:
            if verbose:
                print("  ⚠ Fichier vide.")
            continue

        # Dédoublonner : un user ne doit noter un livre qu'une seule fois
        n_dup = df.duplicated(subset=["user_id", "parent_asin"]).sum()
        if n_dup > 0:
            df = df.drop_duplicates(
                subset=["user_id", "parent_asin"], keep="last"
            )
            if verbose:
                print(f"  {n_dup:,} doublons supprimés")

        n_total = len(df)
        n_users = df["user_id"].nunique()
        n_items = df["parent_asin"].nunique()

        if verbose:
            print(f"  Ratings : {n_total:,}  |  "
                  f"Utilisateurs : {n_users:,}  |  Livres : {n_items:,}")

        # ── Split vectorisé ───────────────────────────────────────────
        t0 = time.perf_counter()

        rng = np.random.default_rng(seed)
        df["_rand"] = rng.random(len(df))
        df = df.sort_values(["user_id", "_rand"]).reset_index(drop=True)

        df["_pos"] = df.groupby("user_id").cumcount()
        df["_total"] = df.groupby("user_id")["_pos"].transform("count")

        # n_test = max(1, floor(total × test_ratio)), borné à total − 1
        n_test_arr = np.floor(df["_total"].values * test_ratio).astype(int)
        n_test_arr = np.clip(n_test_arr, 1, df["_total"].values - 1)

        df["is_test"] = df["_pos"] >= (df["_total"] - n_test_arr)

        aux_cols = ["_rand", "_pos", "_total", "is_test"]
        train_df = df.loc[~df["is_test"]].drop(columns=aux_cols)
        test_df = df.loc[df["is_test"]].drop(columns=aux_cols)
        del df, n_test_arr
        gc.collect()

        t_split = time.perf_counter() - t0

        if verbose:
            actual_train = len(train_df) / n_total
            print(f"  Split vectorisé en {t_split * 1000:.1f} ms")
            print(f"  Ratio effectif : {actual_train:.2%} train / "
                  f"{1 - actual_train:.2%} test")

        # Vérification de la stratification
        users_train = set(train_df["user_id"].unique())
        users_test = set(test_df["user_id"].unique())
        violations = (users_train - users_test) | (users_test - users_train)
        if violations:
            if verbose:
                print(f"  ⚠ {len(violations)} utilisateur(s) absent(s) "
                      f"d'un ensemble")
        elif verbose:
            print("  ✓ Chaque utilisateur présent dans train ET test")
        del users_train, users_test, violations

        # ── Matrices CSR ──────────────────────────────────────────────
        # factorize() sur l'union train+test pour des indices cohérents
        # entre R_train et R_test (même utilisateur = même ligne).
        t1 = time.perf_counter()

        all_users = pd.concat([train_df["user_id"], test_df["user_id"]])
        user_codes, user_ids = pd.factorize(all_users, sort=False)
        del all_users

        all_items = pd.concat(
            [train_df["parent_asin"], test_df["parent_asin"]]
        )
        item_codes, item_ids = pd.factorize(all_items, sort=False)
        del all_items
        gc.collect()

        n_u, n_i = len(user_ids), len(item_ids)
        n_train = len(train_df)

        r_train = csr_matrix(
            (
                train_df["rating"].values.astype(np.float32),
                (user_codes[:n_train], item_codes[:n_train]),
            ),
            shape=(n_u, n_i),
            dtype=np.float32,
        )
        r_test = csr_matrix(
            (
                test_df["rating"].values.astype(np.float32),
                (user_codes[n_train:], item_codes[n_train:]),
            ),
            shape=(n_u, n_i),
            dtype=np.float32,
        )
        del user_codes, item_codes

        t_build = time.perf_counter() - t1
        if verbose:
            print(f"  Matrices CSR ({n_u:,} × {n_i:,}) "
                  f"construites en {t_build * 1000:.1f} ms")
            print(f"  R_train : {r_train.nnz:,} entrées  |  "
                  f"R_test : {r_test.nnz:,} entrées")

        # ── Sauvegarde sur disque ─────────────────────────────────────
        sample_dir = Path(path).parent
        split_dir = sample_dir / SPLIT_SUBDIR
        split_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_parquet(split_dir / TRAIN_FILENAME, index=False)
        test_df.to_parquet(split_dir / TEST_FILENAME, index=False)
        save_npz(split_dir / "R_train.npz", r_train)
        save_npz(split_dir / "R_test.npz", r_test)
        np.save(split_dir / "user_ids.npy", user_ids)
        np.save(split_dir / "item_ids.npy", item_ids)

        metadata = {
            "source_file": str(path),
            "split_seed": seed,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "n_users": int(n_u),
            "n_items": int(n_i),
            "train": {
                "n_ratings": int(r_train.nnz),
                "n_users": int(train_df["user_id"].nunique()),
                "n_items": int(train_df["parent_asin"].nunique()),
            },
            "test": {
                "n_ratings": int(r_test.nnz),
                "n_users": int(test_df["user_id"].nunique()),
                "n_items": int(test_df["parent_asin"].nunique()),
            },
        }
        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        split_dirs.append(str(split_dir))

        if verbose:
            total_size = sum(
                os.path.getsize(split_dir / fn)
                for fn in os.listdir(split_dir)
            )
            print(f"  ✓ Sauvegardé dans {split_dir}/ "
                  f"({total_size / 1024**2:.1f} Mo)")

        del train_df, test_df, r_train, r_test, user_ids, item_ids
        gc.collect()

    flush_ram()
    return split_dirs
