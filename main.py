"""main.py — Orchestrateur pour l'échantillonnage GPU."""

import gc
from precursor import (
    RAPIDS_AVAILABLE, 
    RAW_BOOKS_PATH, 
    PROCESSED_DATA_DIR, 
    TARGET_YEARS,
    sample_active_users_gpu, 
    sample_temporal_gpu,
    flush_ram, 
    flush_gpu,
)

MAX_MANAGED_MEMORY = 50 * (1024 ** 3)  # 50 GB safety cap


def init_rmm():
    """Initialize RMM once with a capped managed-memory pool."""
    import rmm
    import cupy as cp
    from rmm.allocators.cupy import rmm_cupy_allocator

    managed_mr = rmm.mr.ManagedMemoryResource()
    limited_mr = rmm.mr.LimitingResourceAdaptor(managed_mr, allocation_limit=MAX_MANAGED_MEMORY)
    pool_mr = rmm.mr.PoolMemoryResource(
        limited_mr,
        initial_pool_size=2 * (1024 ** 3),
        maximum_pool_size=MAX_MANAGED_MEMORY,
    )
    rmm.mr.set_current_device_resource(pool_mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    print(f"RMM initialized: {MAX_MANAGED_MEMORY / (1024**3):.0f} GB cap")


def main():
    if not RAPIDS_AVAILABLE:
        print("RAPIDS non disponible.")
        return

    init_rmm()

    print("\n=== Active Users Sampling ===")
    sample_active_users_gpu(
        RAW_BOOKS_PATH,
        f"{PROCESSED_DATA_DIR}sample_gpu_active_users_original.parquet",
    )

    # Full cleanup between runs
    flush_ram()
    flush_gpu()
    gc.collect()

    print("\n=== Temporal Sampling ===")
    sample_temporal_gpu(
        RAW_BOOKS_PATH,
        f"{PROCESSED_DATA_DIR}sample_gpu_temporal_original.parquet",
        target_years=TARGET_YEARS,
    )


if __name__ == "__main__":
    main()