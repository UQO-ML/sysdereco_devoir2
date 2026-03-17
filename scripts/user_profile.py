from __future__ import annotations

import pandas as pd
import gc
import glob
import os
import time

TRAIN_DATA_PATHS = sorted(glob.glob("data/joining/*/train_interactions.parquet"))
print(f"{TRAIN_DATA_PATHS}")
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


class user_profile:
    
    def load_dataset(
        self, 
        verbose: bool = True,
        t_start: float = time.time(),
        path: str = TRAIN_DATA_PATHS
    ) -> pd.DataFrame:
        if verbose:
            print("\nload_dataset()")

        df = pd.read_parquet(path)

        print(f"\n{df},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(df):.1f} MiB, \n{df.shape}, \n{df.columns.tolist()}\n, \n{df}\n")

        print(f"load_dataset end: {(time.time() - t_start):.1f}\n")

        return df





    def group_by_user_id(
        self,
        verbose: bool = True,
        t_start: float = time.time(),
        train_dataset_paths: [str] = TRAIN_DATA_PATHS,
    ):
        if verbose:
            print("group_by_user_id()")

        for path in train_dataset_paths:
            print(f"{path}")
            if os.path.isfile(path) and os.path.exists(path):
                train_df = user_profile.load_dataset(self, verbose=verbose, t_start=t_start, path=path)
                grouped_train_df = train_df.groupby("user_id")
                if verbose:
                    print("user_profile.group_by_user().for.if - grouped_train_df : \n"
                        f"{grouped_train_df}\n")
                print(f"grouped_train_df end: {(time.time() - t_start):.1f}")

                del grouped_train_df
                gc.collect()
        print(f"group_by_user_id end: {(time.time() - t_start):.1f}")












def main() -> None:
    t_start = time.time()
    user_profile.group_by_user_id(
        self=user_profile,
        verbose=True,
        t_start=t_start,
        train_dataset_paths=TRAIN_DATA_PATHS,

    )



if __name__ == "__main__":
    main()