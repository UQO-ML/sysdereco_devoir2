from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import gc
import os
import pathlib as Path

try:
    from precursor import(
        RAW_PARQUET_PATHS,
        SAMPLE_GLOB_FILTERED,
        SAMPLE_GLOB_ORIGINAL,
        resolve_glob
    )
except:
    from scripts.precursor import(
        RAW_PARQUET_PATHS,
        SAMPLE_GLOB_FILTERED,
        SAMPLE_GLOB_ORIGINAL,
        resolve_glob
    )

SAMPLE_GLOB_FILTERED_LIST = resolve_glob(SAMPLE_GLOB_FILTERED)
SAMPLE_GLOB_ORIGINAL_LIST = resolve_glob(SAMPLE_GLOB_ORIGINAL)


def stuff():
    pd.set_option("display.max_columns", None)
    for raw_parquet_files in RAW_PARQUET_PATHS:
        parquet_data = pd.read_parquet(raw_parquet_files)
        print(f"\n{raw_parquet_files},\n{parquet_data.shape}, \n{parquet_data.columns.tolist()}\n, \n{parquet_data}\n")
        del parquet_data

    for original_sample_path in SAMPLE_GLOB_ORIGINAL_LIST:
        if os.path.exists(original_sample_path):
            original_sample_data = pd.read_parquet(original_sample_path)
            print(f"\n{original_sample_path},\n{original_sample_data.shape},\n{original_sample_data.columns.tolist()}\n, \n{original_sample_data}\n")
            del original_sample_data
        
    for filtered_sample_path in SAMPLE_GLOB_FILTERED_LIST:
        if os.path.exists(filtered_sample_path):
            filtered_sample_data = pd.read_parquet(filtered_sample_path)
            print(f"\n{filtered_sample_path},\n{filtered_sample_data.shape},\n{filtered_sample_data.columns.tolist()}\n, \n{filtered_sample_data}\n")
            del filtered_sample_data


if __name__ == "__main__":
    stuff()