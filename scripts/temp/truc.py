from __future__ import annotations
from pydoc import resolve

import pandas as pd
import pyarrow.parquet as pq
import gc
import os
import time
import glob


def resolve_glob(pattern: str) -> list[str]:
    """Résout un glob pattern en une liste triée de chemins existants."""
    return sorted(glob.glob(pattern))


PROCESSED_DATA_DIR = "data/processed/"
RAW_DATA_DIR = "data/raw/"
RAW_PARQUET_GLOB = f"{RAW_DATA_DIR}parquet/*.parquet"
RAW_PARQUET_PATHS = resolve_glob(RAW_PARQUET_GLOB)

SAMPLE_GLOB_ORIGINAL = f"{PROCESSED_DATA_DIR}sample-*/*_original.parquet"
SAMPLE_GLOB_CLEANED = f"{PROCESSED_DATA_DIR}sample-*/*_cleaned.parquet"
SAMPLE_GLOB_FILTERED = f"{PROCESSED_DATA_DIR}sample-*/*_filtered.parquet"
            

JOINED_DATA_DIR = "data/joining"
CLEANED_JOINED_DATA = resolve_glob(f"{JOINED_DATA_DIR}/*_clean_joined.parquet")
TRAIN_SPLITED_JOINED_DATA = resolve_glob(f"{JOINED_DATA_DIR}/*/train_interactions.parquet")
TEST_SPLITED_JOINED_DATA = resolve_glob(f"{JOINED_DATA_DIR}/*/test_interactions.parquet")

SAMPLE_GLOB_FILTERED_LIST = resolve_glob(SAMPLE_GLOB_FILTERED)
SAMPLE_GLOB_ORIGINAL_LIST = resolve_glob(SAMPLE_GLOB_ORIGINAL)


def _fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"


def _disk_size(path: str) -> str:
    try:
        return _fmt_size(os.path.getsize(path))
    except OSError:
        return "N/A"


def _df_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def stuff():



    for raw_parquet_files in RAW_PARQUET_PATHS:
        parquet_data = pd.read_parquet(raw_parquet_files)
        print(f"\n{raw_parquet_files},\ndisk: {_disk_size(raw_parquet_files)}, \nmemory (loaded): {_df_memory_mb(parquet_data):.1f} MiB, \n{parquet_data.shape}, \n{parquet_data.columns.tolist()}\n")
        del parquet_data
        gc.collect()
        pf = pq.ParquetFile(raw_parquet_files)
        print(pf.schema_arrow)
        print(pf.metadata.num_rows)
        del pf
        gc.collect()


    # for original_sample_path in SAMPLE_GLOB_ORIGINAL_LIST:
    #     if os.path.exists(original_sample_path):
    #         original_sample_data = pd.read_parquet(original_sample_path)
    #         print(f"\n{original_sample_path}, \ndisk: {_disk_size(original_sample_path)}, \nmemory (loaded): {_df_memory_mb(original_sample_data):.1f} MiB, \n{original_sample_data.shape},\n{original_sample_data.columns.tolist()}\n, \n{original_sample_data.schema_arrow}\n, \n{original_sample_data}\n")
    #         del original_sample_data
    #         gc.collect()
    #         pf = pq.ParquetFile(original_sample_path)
    #         print(pf.schema_arrow)
    #         print(pf.metadata.num_rows)
    #         del pf
    #         gc.collect()
            
    # for filtered_sample_path in SAMPLE_GLOB_FILTERED_LIST:
    #     if os.path.exists(filtered_sample_path):
    #         filtered_sample_data = pd.read_parquet(filtered_sample_path)
    #         print(f"\n{filtered_sample_path}, \ndisk: {_disk_size(filtered_sample_path)}, \nmemory (loaded): {_df_memory_mb(filtered_sample_data):.1f} MiB, \n{filtered_sample_data.shape},\n{filtered_sample_data.columns.tolist()}\n, \n{filtered_sample_data.schema_arrow}\n, \n{filtered_sample_data}\n")
    #         del filtered_sample_data
    #         gc.collect()
    #         pf = pq.ParquetFile(filtered_sample_path)
    #         print(pf.schema_arrow)
    #         print(pf.metadata.num_rows)
    #         del pf
    #         gc.collect()

    # for clean_joined_path in CLEANED_JOINED_DATA:
    #     if os.path.exists(clean_joined_path):
    #         clean_joined_data = pd.read_parquet(clean_joined_path)
    #         print(f"\n{clean_joined_path}, \ndisk: {_disk_size(clean_joined_path)}, \nmemory (loaded): {_df_memory_mb(clean_joined_data):.1f} MiB, \n{clean_joined_data.shape},\n{clean_joined_data.columns.tolist()}\n, \n{clean_joined_data.schema_arrow}\n, \n\n")
    #         del clean_joined_data
    #         gc.collect()
    #         pf = pq.ParquetFile(clean_joined_path)
    #         print(pf.schema_arrow)
    #         print(pf.metadata.num_rows)
    #         del pf
    #         gc.collect()

    # for train_joined_path in TRAIN_SPLITED_JOINED_DATA:
    #     if os.path.exists(train_joined_path):
    #         train_joined_data = pd.read_parquet(train_joined_path)
    #         print(f"\n{train_joined_path}, \ndisk: {_disk_size(train_joined_path)}, \nmemory (loaded): {_df_memory_mb(train_joined_data):.1f} MiB, \n{train_joined_data.shape},\n{train_joined_data.columns.tolist()}\n, \n{train_joined_data.schema_arrow}\n, \n\n")
    #         del train_joined_data
    #         gc.collect()
    #         pf = pq.ParquetFile(train_joined_path)
    #         print(pf.schema_arrow)
    #         print(pf.metadata.num_rows)
    #         del pf
    #         gc.collect()

    # for test_joined_path in TEST_SPLITED_JOINED_DATA:
    #     if os.path.exists(test_joined_path):
    #         test_joined_data = pd.read_parquet(test_joined_path)
    #         print(f"\n{test_joined_path}, \ndisk: {_disk_size(test_joined_path)}, \nmemory (loaded): {_df_memory_mb(test_joined_data):.1f} MiB, \n{test_joined_data.shape},\n{test_joined_data.columns.tolist()}\n, \n{test_joined_data.schema_arrow}\n, \n\n")
    #         del test_joined_data
    #         gc.collect()
    #         pf = pq.ParquetFile(test_joined_path)
    #         print(pf.schema_arrow)
    #         print(pf.metadata.num_rows)


if __name__ == "__main__":
    t_start = time.time()
    stuff()
    print(f"Elapsed: {(time.time() - t_start):.1f}s")