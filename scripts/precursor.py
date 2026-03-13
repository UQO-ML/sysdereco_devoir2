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
import time
from pathlib import Path
from collections import Counter, deque
from collections.abc import Callable

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import hashlib

import numpy as np
import pandas as pd
import polars as pl
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

    def _align_down_256(n: int) -> int:
        return n - (n % 256)

    def _compute_managed_memory_cap(
        ram_reserve_gb: float = 4.0,
        vram_fraction: float = 0.90,
    ) -> int:
        """
        Compute a safe RMM managed-memory cap based on actual hardware.
        
        With managed_memory=True, CUDA UVM can spill from VRAM to RAM.
        The cap must account for both:
        - VRAM: use most of it (vram_fraction)
        - RAM spill: leave ram_reserve_gb free for the OS, Python, pandas, etc.
        
        Returns the cap in bytes.
        """
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        vram_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total
        pynvml.nvmlShutdown()

        # Total system RAM (requires psutil, or read from /proc/meminfo)
        try:
            import psutil
            ram_total = psutil.virtual_memory().total
        except ImportError:
            # Fallback: read from /proc/meminfo (Linux only)
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_total = int(line.split()[1]) * 1024  # kB → bytes
                        break

        usable_vram = int(vram_total * vram_fraction)
        usable_ram_spill = ram_total - int(ram_reserve_gb * 1024**3)
        
        # The cap is VRAM + how much RAM we're willing to let UVM spill into
        cap = usable_vram + max(usable_ram_spill, 0)
        
        return _align_down_256(cap)

    try:
        _MANAGED_MEMORY_CAP = _compute_managed_memory_cap()
    except Exception:
        _MANAGED_MEMORY_CAP = 50 * (1024**3)  # safe fallback
    
    print(f"Managed_memory_cap: {_MANAGED_MEMORY_CAP/(1024**3)} GB")
    
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


def jsonl_to_parquet_conversion() -> bool:

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  CONVERSION JSONL → PARQUET                                         ║")
    print("║  Lecture des fichiers JSONL bruts et conversion en Parquet (Polars). ║")
    print("║  Le format Parquet offre une compression columnar 3-5× plus         ║")
    print("║  compacte que JSONL et permet la lecture paresseuse (lazy scanning). ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  Fichiers JSONL trouvés : {len(RAW_JSONL_PATHS)}")
    print(f"  Colonnes numériques surveillées : {CANDIDATE_NUMERIC_COLUMNS}")

    result = True

    for jsonl_path in RAW_JSONL_PATHS:
        parquet_path = jsonl_path.replace("jsonl", "parquet")
        
        # ── 1. Passer si déjà fait (avec vérification d'intégrité) ───────────
        if os.path.exists(parquet_path):
            try:
                n_parquet = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
                cols_parquet = set(pl.scan_parquet(parquet_path).collect_schema().names())
                
                with open(jsonl_path, "rb") as f:
                    n_jsonl = sum(1 for line in f if line.strip())
                
                if n_parquet != n_jsonl:
                    print(f"  ⚠ Nombre de lignes incohérent, reconversion : {jsonl_path}")
                else:
                    schema_jsonl = pl.scan_ndjson(jsonl_path, infer_schema_length=1).collect_schema()
                    cols_jsonl = set(schema_jsonl.names())
                    if cols_jsonl == cols_parquet:
                        print(f"  ✓ Déjà converti (vérifié) : {parquet_path}")
                        continue
                    else:
                        print(f"  ⚠ Schéma incohérent, reconversion : {jsonl_path}")
            except Exception as e:
                print(f"  ⚠ Fichier existant invalide ({e}), reconversion : {parquet_path}")
        
        # ── 2. Détection des colonnes à surcharger ───────────────────────────
        schema_initial = pl.scan_ndjson(jsonl_path, infer_schema_length=1).collect_schema()
        cols_to_override = [c for c in schema_initial.names() if c in CANDIDATE_NUMERIC_COLUMNS]
        schema_overrides = {c: pl.Utf8 for c in cols_to_override} if cols_to_override else None
        
        # ── 3. Conversion ────────────────────────────────────────────────────
        print(f"  Conversion {jsonl_path} → {parquet_path}...")
        kwargs = {"schema_overrides": schema_overrides} if schema_overrides else {}
        lf = pl.scan_ndjson(jsonl_path, **kwargs)
        if cols_to_override:
            lf = lf.with_columns([
                pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_override
            ])
        lf.sink_parquet(parquet_path)
        del lf
        gc.collect()
        
        # ── 4. Intégrité : nombre de lignes (sans chargement complet) ────────
        with open(jsonl_path, "rb") as f:
            n_jsonl = sum(1 for line in f if line.strip())
        n_parquet = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
        
        if n_jsonl != n_parquet:
            raise ValueError(
                f"Incohérence de données ! JSONL : {n_jsonl:,} lignes, Parquet : {n_parquet:,} lignes. "
                f"Fichier : {jsonl_path}"
            )
        print(f"  ✓ Lignes vérifiées : {n_parquet:,}")
        
        # ── 5. Intégrité : schéma (noms de colonnes) ─────────────────────────
        cols_jsonl = set(pl.scan_ndjson(jsonl_path, infer_schema_length=1).collect_schema().names())
        cols_parquet = set(pl.scan_parquet(parquet_path).collect_schema().names())
        
        if cols_jsonl != cols_parquet:
            only_jsonl = cols_jsonl - cols_parquet
            only_parquet = cols_parquet - cols_jsonl
            raise ValueError(
                f"Schéma incohérent ! {jsonl_path}\n"
                f"  Uniquement dans JSONL : {only_jsonl or 'aucun'}\n"
                f"  Uniquement dans Parquet : {only_parquet or 'aucun'}"
            )
        print(f"  ✓ Schéma vérifié : {list(cols_parquet)}")
        
        # ── 6. Vérification par échantillon ──────────────────────────────────
        n_sample = min(N_SPOT_CHECK, n_parquet)
        if n_sample == 0:
            print(f"  ✓ Vérification ignorée (fichier vide)")
        else:
            read_kwargs = {"schema_overrides": schema_overrides} if schema_overrides else {}
            df_jsonl_sample = pl.read_ndjson(jsonl_path, n_rows=n_sample, **read_kwargs)
            if cols_to_override:
                df_jsonl_sample = df_jsonl_sample.with_columns([
                    pl.col(c).cast(pl.Float64, strict=False) for c in cols_to_override
                ])
            df_parquet_sample = pl.read_parquet(parquet_path, n_rows=n_sample)
            df_jsonl_sample = df_jsonl_sample.select(df_parquet_sample.columns)
            
            for col in df_parquet_sample.columns:
                s_jsonl = df_jsonl_sample[col]
                s_parquet = df_parquet_sample[col]
                if not s_jsonl.eq_missing(s_parquet).all():
                    diff_mask = ~s_jsonl.eq_missing(s_parquet)
                    idx = diff_mask.arg_true()[0]
                    result = False
                    raise ValueError(
                        f"Vérification échouée : colonne '{col}' diffère à la ligne {idx}\n"
                        f"  JSONL :   {s_jsonl[idx]}\n"
                        f"  Parquet : {s_parquet[idx]}"
                    )
            print(f"  ✓ Vérification réussie ({n_sample:,} premières lignes)")
            
            del df_jsonl_sample, df_parquet_sample
            gc.collect()
        
        # ── 7. Nettoyage par fichier ─────────────────────────────────────────
        gc.collect()

    # ── 8. Nettoyage final et résumé ────────────────────────────────────
    gc.collect()
    print("\n✓ Conversion terminée. Tous les fichiers JSONL ont été convertis en Parquet.")
    print("  Les fichiers Parquet sont prêts pour l'analyse dans les cellules suivantes.")  
    return result


def deterministic_sample_users(
    user_ids, 
    num_users: int = NUM_USERS, 
    seed: int = SEED,
) -> list[str]:
    """
    Deterministic backend-independent sampling by hash ranking.
    user_ids: iterable of user_id (any backend, convertable to str)
    """
    seen = set()
    scored = []

    for uid in user_ids:
        s = str(uid)
        if s in seen:
            continue
        seen.add(s)

        h = hashlib.blake2b(f"{seed}:{s}".encode("utf-8"), digest_size=8).digest()
        score = int.from_bytes(h, "big")
        scored.append((score, s))

    scored.sort(key=lambda x: x[0])  # smallest hashes first
    return [uid for _, uid in scored[:min(num_users, len(scored))]]


def _get_free_vram_bytes(
    device_index: int = 0, 
    fallback_bytes: int = 2 * 1024**3
) -> int:
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"Free Vram : {int(info.free)/1e9:.2f}")
            return max(1, int(info.free))
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    if RAPIDS_AVAILABLE and cp is not None:
        try:
            free_bytes, _ = cp.cuda.runtime.memGetInfo()
            return max(1, int(free_bytes))
            
        except Exception:
            pass
    return max(1, int(fallback_bytes))


def _droppable_columns(table: pa.Table) -> list[str]:
    dominated = []
    for field in table.schema:
        if pa.types.is_large_list(field.type) or pa.types.is_nested(field.type):
            dominated.append(field.name)
    return dominated


def _estimates_bytes_per_row(
    parquet_path: str,
    columns: list[str],
    sample_rows: int = CHUNK_SIZE,
)-> int:
    pf = pq.ParquetFile(parquet_path)

    batches = pf.iter_batches(
        batch_size=sample_rows,
        columns=columns if columns else None,
    )
    first_batch = next(batches, None)
    if first_batch is None or first_batch.num_rows == 0:
        return 1
    table = pa.Table.from_batches([first_batch])
    return max(1, table.nbytes // table.num_rows)


def _compute_adaptive_chunk_rows(
    parquet_path: str,
    probe_columns: list[str],
    safety_ratio: float = 0.9,
    min_rows: int = 50_000,
    max_rows: int = 500_000_000
) -> int:
    free_vram = _get_free_vram_bytes()
    bytes_per_row = _estimates_bytes_per_row(
        parquet_path,
        probe_columns
        )
    bytes_per_row = max(1, int(bytes_per_row))

    raw_rows = int((free_vram * float(safety_ratio)) / bytes_per_row)

    clamped_rows = max(min_rows, raw_rows)

    print(f"Clamped_rows : {clamped_rows}")
    return int(clamped_rows)


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True

    msg = str(exc).lower()
    patterns = (
        "out of memory",
        "cudaerrormemoryallocation",
        "std::bad_alloc",
        "rmm",
        "memory pool",
    )
    if any(p in msg for p in patterns):
        return True

    # Optionnel: checks de types RAPIDS si disponibles
    try:
        import rmm  # noqa: F401
        # selon versions RAPIDS, les classes peuvent varier
    except Exception:
        pass

    return False


def _process_table_with_oom_retry(
    table: pa.Table,
    process_fn: Callable[[pa.Table], None],
    min_rows: int,
    verbose: bool = False,
) -> tuple[int, int]:
    """
    Traite une table Arrow avec retry OOM via split binaire.
    Returns: (oom_retries, split_count)
    """
    queue = deque([table])
    oom_retries = 0
    split_count = 0

    while queue:
        current = queue.popleft()
        try:
            process_fn(current)
            _flush_memory()
        except Exception as exc:
            if not _is_oom_error(exc):
                raise

            oom_retries += 1
            _flush_memory()

            if current.num_rows <= min_rows:
                # trop petit pour split, on remonte l'erreur
                raise

            mid = current.num_rows // 2
            left = current.slice(0, mid)
            right = current.slice(mid)

            split_count += 1
            # traiter left puis right
            queue.appendleft(right)
            queue.appendleft(left)

            if verbose:
                print(f"  OOM retry: split {current.num_rows:,} -> {left.num_rows:,} + {right.num_rows:,}")

    return oom_retries, split_count

    
# =============================================================================
#   ÉCHANTILLONNAGE GPU
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

    start = time.time()

    if not RAPIDS_AVAILABLE:
        if verbose:
            print("ℹ RAPIDS non disponible — sample_active_users_gpu ignoré.")
        return None

    flush_ram()
    flush_gpu()

    if verbose:
        _print_gpu_status("Before start")

    if verbose:
        print("Phase 1 : Chargement en mémoire GPU...")

    batch_size_a = _compute_adaptive_chunk_rows(
        parquet_path,
        ["user_id"],
    )

    if verbose:
        free = _get_free_vram_bytes()
        bpr = _estimates_bytes_per_row(parquet_path, ["user_id"])
        print(f"  Pass A: batch_size={batch_size_a:,} (free_vram={free/1e9:.2f} GB, bpr={bpr} B)")

    user_counts_cpu = Counter()

    oom_retries_total = 0

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size_a, columns=["user_id"]):
        table = pa.Table.from_batches([batch])

        def _count_chunk(t: pa.Table) -> None:
            gdf = cudf.DataFrame.from_arrow(t)
            vc = gdf["user_id"].value_counts()
            # Small transfer: only (user_id, count) pairs -> CPU
            pdf = vc.to_pandas()
            for uid, cnt in zip(pdf.index, pdf.values):
                user_counts_cpu[str(uid)] += int(cnt)
            if verbose:
                _print_gpu_status("After load")
                print(f"  GPU DataFrame: {gdf.memory_usage(deep=True).sum() / 1e9:.2f} GB")

            del gdf, vc, pdf

        retries, splits = _process_table_with_oom_retry(
            table, _count_chunk, min_rows=50_000, verbose=verbose,
        )
        oom_retries_total += retries

    del pf

    active_users = [uid for uid, cnt in user_counts_cpu.items() if cnt >= min_reviews]
    del user_counts_cpu

    if verbose:
        print(f"  Utilisateurs actifs (>= {min_reviews}): {len(active_users):,}")

    selected_users = deterministic_sample_users(active_users, num_users=num_users, seed=seed)
    del active_users
    
    if verbose:
        print(f"  Utilisateurs échantillonnés: {len(selected_users):,}")

    _flush_memory()

    # # Sélection aléatoire sur GPU — seuls les 50k indices sont copiés vers le CPU
    # cp.random.seed(seed)
    # n_to_sample = min(num_users, len(active_users))
    # indices = cp.random.choice(len(active_users), size=n_to_sample, replace=False)
    # active_users_series = active_users.to_series().reset_index(drop=True)
    # selected_users = active_users_series.iloc[cp.asnumpy(indices)]
    # del user_counts, active_users, active_users_series, indices

    if verbose:
        print(f"  Utilisateurs actifs sélectionnés: {len(selected_users):,}")

    _flush_memory()
    if verbose:
        _print_gpu_status("After count flush")

    if verbose:
        print("\nPhase 2 : Filtrage...")
    
    batch_size_b = _compute_adaptive_chunk_rows(parquet_path, [])  # all columns

    if verbose:
        free = _get_free_vram_bytes()
        bpr = _estimates_bytes_per_row(parquet_path, [])
        print(f"  Pass B: batch_size={batch_size_b:,} (free_vram={free/1e9:.2f} GB, bpr={bpr} B)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected_set_gpu = cudf.Series(selected_users)
    del selected_users

    writer = None
    n_reviews = 0

    try:
        pf2 = pq.ParquetFile(parquet_path)
        for batch in pf2.iter_batches(batch_size=batch_size_b):
            table = pa.Table.from_batches([batch])

            def _filter_chunk(t: pa.Table) -> None:
                nonlocal writer, n_reviews
                t_slim = t.drop(_droppable_columns(t))
                gdf = cudf.DataFrame.from_arrow(t_slim)
                mask = gdf["user_id"].isin(selected_set_gpu)
                indices = mask[mask].index
                    
                if len(indices) == 0:
                    del gdf, mask
                    return
                # Filter the *original* Arrow table to preserve all columns
                out_arrow = t.filter(mask.to_arrow())
                del gdf, mask

                if writer is None:
                    writer = pq.ParquetWriter(output_path, out_arrow.schema, compression="snappy")
                writer.write_table(out_arrow)
                n_reviews += out_arrow.num_rows

            retries, splits = _process_table_with_oom_retry(
                table, _filter_chunk, min_rows=50_000, verbose=verbose,
            )
            oom_retries_total += retries

        del pf2, selected_set_gpu
    finally:
        if writer is not None:
            writer.close()

    _flush_memory()

    if verbose:
        elapsed = time.time() - start
        print(f"  Temps: {elapsed:.2f}s — Reviews: {n_reviews:,}")
        _print_gpu_status("Final cleanup")
        if oom_retries_total > 0:
            print(f"  OOM retries: {oom_retries_total}")
        _print_gpu_status("Final")
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
    start = time.time()

    if not RAPIDS_AVAILABLE:
        if verbose:
            print("ℹ RAPIDS non disponible — sample_temporal_gpu ignoré.")
        return None

    if target_years is None:
        target_years = TARGET_YEARS

    target_years_set = target_years

    flush_ram()
    flush_gpu()

    if verbose:
        print("Chargement en mémoire GPU...")
    _flush_memory()
    if verbose:
        _print_gpu_status("Load start")

    batch_size_a = _compute_adaptive_chunk_rows(parquet_path, ["user_id", "timestamp"])
    if verbose:
        print(f"  Pass A: batch_size={batch_size_a:,}")



    user_counts_cpu = Counter()
    oom_retries_total = 0

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size_a, columns=["user_id", "timestamp"]):
        table = pa.Table.from_batches([batch])

        def _count_temporal_chunk(t: pa.Table) -> None:
            t_slim = t.drop(_droppable_columns(t))
            gdf = cudf.DataFrame.from_arrow(t_slim)

            # Normalize timestamp -> year
            ts = gdf["timestamp"]
            if not str(ts.dtype).startswith("datetime64"):
                gdf["timestamp"] = cudf.to_datetime(ts, unit="ms")
            gdf["year"] = gdf["timestamp"].dt.year

            # Keep only target years
            gdf_period = gdf[gdf["year"].isin(target_years_set)]

            if len(gdf_period) == 0:
                del gdf, gdf_period
                return

            vc = gdf_period["user_id"].value_counts()
            pdf = vc.to_pandas()
            for uid, cnt in zip(pdf.index, pdf.values):
                user_counts_cpu[str(uid)] += int(cnt)

            # Statistiques par année (petit transfert vers pandas pour affichage)
            year_stats = (
                gdf_period.groupby("year")
                .agg({"user_id": ["count", "nunique"]})
            )
            ys = year_stats.to_pandas()
            del year_stats
            ys.columns = ["review_count", "unique_users"]
            ys = ys.sort_index()
            if verbose:
                print(f"\n── Répartition {target_years[0]}–{target_years[-1]} ──")
                print(ys.to_string())
                print(f"\nTotal reviews : {ys['review_count'].sum():,}")
            del ys

            if verbose:
                print(f"  Reviews dans période : {len(gdf_period):,}")

            del gdf, gdf_period, vc, pdf

        retries, _ = _process_table_with_oom_retry(
            table, _count_temporal_chunk, min_rows=50_000, verbose=verbose,
        )
        oom_retries_total += retries

    del pf
    _flush_memory()


    active_users = [uid for uid, cnt in user_counts_cpu.items() if cnt >= min_reviews]
    del user_counts_cpu

    if verbose:
        print(f"  Utilisateurs actifs dans période (>= {min_reviews}): {len(active_users):,}")

    selected_users = deterministic_sample_users(active_users, num_users=num_users, seed=seed)
    del active_users

    if verbose:
        print(f"  Utilisateurs échantillonnés: {len(selected_users):,}")

    _flush_memory()


    # # Sélection aléatoire sur GPU
    # active_user_ids = active_in_period["user_id"].reset_index(drop=True)
    # del active_in_period
    # cp.random.seed(seed)
    # n_to_sample = min(num_users, len(active_user_ids))
    # indices = cp.random.choice(len(active_user_ids), size=n_to_sample, replace=False)
    # sampled_series = active_user_ids.iloc[cp.asnumpy(indices)]
    # del active_user_ids, indices

    batch_size_b = _compute_adaptive_chunk_rows(parquet_path, [])

    if verbose:
        print(f"  Pass B: batch_size={batch_size_b:,}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected_set_gpu = cudf.Series(selected_users)
    del selected_users

    writer = None
    n_reviews = 0

    try:
        pf2 = pq.ParquetFile(parquet_path)
        for batch in pf2.iter_batches(batch_size=batch_size_b):
            table = pa.Table.from_batches([batch])

            def _filter_temporal_chunk(t: pa.Table) -> None:
                nonlocal writer, n_reviews
                t_slim = t.drop(_droppable_columns(t))
                gdf = cudf.DataFrame.from_arrow(t_slim)

                # Year filter
                ts = gdf["timestamp"]

                if not str(ts.dtype).startswith("datetime64"):
                    gdf["timestamp"] = cudf.to_datetime(ts, unit="ms")

                gdf["year"] = gdf["timestamp"].dt.year
                gdf_period = gdf[gdf["year"].isin(target_years_set)]
                del gdf

                if len(gdf_period) == 0:
                    del gdf_period
                    return

                # User filter
                filtered = gdf_period[gdf_period["user_id"].isin(selected_set_gpu)]
                del gdf_period

                if len(filtered) == 0:
                    del filtered
                    return

                # Drop helper column before writing
                if "year" in filtered.columns:
                    filtered = filtered.drop(columns=["year"])

                out_arrow = filtered.to_arrow()
                del filtered

                if writer is None:
                    writer = pq.ParquetWriter(output_path, out_arrow.schema, compression="snappy")
                writer.write_table(out_arrow)
                n_reviews += out_arrow.num_rows

            retries, _ = _process_table_with_oom_retry(
                table, _filter_temporal_chunk, min_rows=50_000, verbose=verbose,
            )
            oom_retries_total += retries

        del pf2, selected_set_gpu
    finally:
        if writer is not None:
            writer.close()

    _flush_memory()

    if verbose:
        elapsed = time.time() - start
        print(f"\n⚡ Temps: {elapsed:.2f}s — Reviews écrites: {n_reviews:,}")
        if oom_retries_total > 0:
            print(f"  OOM retries: {oom_retries_total}")
        _print_gpu_status("Final")

    return n_reviews


# =============================================================================
#  ÉCHANTILLONNAGE CPU (alternatives PyArrow pour machines sans GPU)
# =============================================================================

def sample_active_users_cpu(
    parquet_path: str,
    output_path: str,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    batch_size: int = CHUNK_SIZE,
    seed: int = SEED,
    verbose: bool = True,
) -> int:
    flush_ram()
    start = time.time()

    # Pass 1: stream only user_id to count reviews/user
    pf = pq.ParquetFile(parquet_path)
    user_chunks = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=["user_id"]):
        user_chunks.append(batch.column("user_id"))
    del pf
    
    all_user_ids = pa.chunked_array(user_chunks)
    del user_chunks
    gc.collect()

    vc = pc.value_counts(all_user_ids)  # struct array with values + counts
    del all_user_ids

    values = vc.field("values")
    counts = vc.field("counts")
    active_mask = pc.greater_equal(counts, min_reviews)
    active_users = pc.filter(values, active_mask).to_pylist()
    del vc, values, counts, active_mask
    gc.collect()

    if verbose:
        print(f"  Utilisateurs actifs (>= {min_reviews} reviews): {len(active_users):,}")


    # random.seed(seed)
    # n_to_sample = min(num_users, len(active_users))
    # if n_to_sample == 0:
    #     print(f"Pas d\'utilisateur a echantilloner n_to_sample = {n_to_sample}")
    # selected_users = random.sample(active_users, n_to_sample) if n_to_sample > 0 else []
    # del active_users


    selected_users = deterministic_sample_users(
        active_users,
        num_users=num_users,
        seed=seed,
    )

    n_to_sample = len(selected_users)
    if n_to_sample == 0:
        print(f"Pas d\'utilisateur a echantilloner n_to_sample = {n_to_sample}")

    del active_users

    
    gc.collect()

    if verbose:
        print(f"  Utilisateurs échantillonnés: {n_to_sample:,}")

    selected_arr = pa.array(selected_users)
    del selected_users
    gc.collect()

    # Pass 2: stream all columns, filter rows by selected users, write incrementally
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = None
    n_reviews = 0

    # Incremental write keeps memory bounded to ~one batch
    pf2 = pq.ParquetFile(parquet_path)
    for batch in pf2.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])
        mask = pc.is_in(table.column("user_id"), value_set=selected_arr)
        filtered = table.filter(mask)

        if filtered.num_rows > 0:
            if writer is None:
                writer = pq.ParquetWriter(output_path, filtered.schema, compression="snappy")
            writer.write_table(filtered)
            n_reviews += filtered.num_rows

    if writer is not None:
        writer.close()

    if verbose:
        elapsed = time.time() - start
        print(f"  Temps: {elapsed:.2f}s — Reviews: {n_reviews:,}")

    return n_reviews


def sample_temporal_cpu(
    parquet_path: str,
    output_path: str,
    target_years: list[int] | None = None,
    min_reviews: int = MIN_REVIEWS,
    num_users: int = NUM_USERS,
    batch_size: int = CHUNK_SIZE,
    seed: int = SEED,
    verbose: bool = True,
) -> int:
    """
    CPU-safe temporal sampling using PyArrow streaming (no full DataFrame load).

    Pass 1:
      - Stream only user_id + timestamp
      - Keep rows in target years
      - Count reviews/user in period
      - Sample users with >= min_reviews

    Pass 2:
      - Stream full rows
      - Keep rows in target years
      - Keep sampled users
      - Write incrementally with ParquetWriter
    """
    if target_years is None:
        target_years = TARGET_YEARS

    flush_ram()
    start = time.time()

    target_years_set = set(target_years)
    target_years_arr = pa.array(sorted(target_years_set), type=pa.int64())

    # ---------------------------
    # Pass 1: count active users in period
    # ---------------------------
    if verbose:
        print("Pass 1/2 : comptage des utilisateurs actifs dans la période...")

    # Streaming read to avoid loading full parquet in RAM
    pf = pq.ParquetFile(parquet_path)

    user_chunks = []
    year_counter = {y: 0 for y in sorted(target_years_set)}

    for batch in pf.iter_batches(batch_size=batch_size, columns=["user_id", "timestamp"]):
        table = pa.Table.from_batches([batch])

        ts_col = table.column("timestamp")
        ts_type = ts_col.type

        # Handle both int(ms) and timestamp parquet schemas
        if pa.types.is_integer(ts_type):
            # ms -> timestamp[ms]
            ts = pc.cast(ts_col, pa.timestamp("ms"))
        elif pa.types.is_timestamp(ts_type):
            ts = ts_col
        else:
            # Fallback: try cast anyway
            ts = pc.cast(ts_col, pa.timestamp("ms"))

        years = pc.year(ts)
        year_mask = pc.is_in(years, value_set=target_years_arr)

        # Optional yearly diagnostics
        if verbose:
            for y in year_counter:
                year_counter[y] += int(pc.sum(pc.equal(years, y).cast(pa.int64())).as_py() or 0)
            
            print(f"  Reviews dans période {min(target_years_set)}-{max(target_years_set)} : {sum(year_counter.values()):,}")

        filtered_users = pc.filter(table.column("user_id"), year_mask)
        if len(filtered_users) > 0:
            user_chunks.append(filtered_users)

    del pf

    if not user_chunks:
        if verbose:
            print("Aucune review trouvée dans la période cible.")
        return 0

    all_period_users = pa.chunked_array(user_chunks)
    del user_chunks
    gc.collect()

    vc = pc.value_counts(all_period_users)
    del all_period_users
    gc.collect()

    values = vc.field("values")
    counts = vc.field("counts")
    active_mask = pc.greater_equal(counts, min_reviews)
    active_user_ids = pc.filter(values, active_mask).to_pylist()
    if verbose:
        print(f"  Utilisateurs actifs (>= {min_reviews}) : {len(active_user_ids):,}")

    del vc, values, counts, active_mask
    gc.collect()

    # random.seed(seed)
    # n_to_sample = min(num_users, len(active_user_ids))
    # if verbose: 
    #     print(f"  Utilisateurs échantillonnés : {n_to_sample:,}")
    # sampled_users = random.sample(active_user_ids, n_to_sample) if n_to_sample > 0 else []
    # sampled_set = set(sampled_users)
    # sampled_arr = pa.array(sampled_users)
    # del active_user_ids, sampled_users

    sampled_users = deterministic_sample_users(
        active_user_ids,
        num_users=num_users,
        seed=seed,
    )
    n_to_sample = len(sampled_users)
    if verbose: 
        print(f"  Utilisateurs échantillonnés : {n_to_sample:,}")
    sampled_set = set(sampled_users)
    sampled_arr = pa.array(sampled_users)

    del active_user_ids, sampled_users

    gc.collect()

    if verbose:
        total_period_reviews = sum(year_counter.values())
        print(f"  Reviews dans période {min(target_years_set)}-{max(target_years_set)} : {total_period_reviews:,}")
        print(f"  Utilisateurs actifs (>= {min_reviews}) : {len(sampled_set):,}")
        print(f"  Utilisateurs échantillonnés : {n_to_sample:,}")

    # ---------------------------
    # Pass 2: stream full rows and write filtered output
    # ---------------------------
    if verbose:
        print("Pass 2/2 : filtrage final et écriture incrémentale...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = None
    n_reviews = 0

    # Streaming read to avoid loading full parquet in RAM
    pf2 = pq.ParquetFile(parquet_path)
    for batch in pf2.iter_batches(batch_size=batch_size):
        table = pa.Table.from_batches([batch])

        ts_col = table.column("timestamp")
        ts_type = ts_col.type
        if pa.types.is_integer(ts_type):
            ts = pc.cast(ts_col, pa.timestamp("ms"))
        elif pa.types.is_timestamp(ts_type):
            ts = ts_col
        else:
            ts = pc.cast(ts_col, pa.timestamp("ms"))

        years = pc.year(ts)
        year_mask = pc.is_in(years, value_set=target_years_arr)
        table_period = table.filter(year_mask)

        if table_period.num_rows == 0:
            continue

        user_mask = pc.is_in(table_period.column("user_id"), value_set=sampled_arr)
        filtered = table_period.filter(user_mask)

        if filtered.num_rows > 0:
            if writer is None:
                writer = pq.ParquetWriter(output_path, filtered.schema, compression="snappy")
            writer.write_table(filtered)
            n_reviews += filtered.num_rows

    if writer is not None:
        writer.close()

    if verbose:
        elapsed = time.time() - start
        print(f"\n⚡ Temps: {elapsed:.2f}s — Reviews écrites: {n_reviews:,}")

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
