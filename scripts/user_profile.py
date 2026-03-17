from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gc
import os
import time
import glob

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

TRAIN_PATHS = sorted(Path("data/joining").glob("*/train_interactions.parquet"))
TRAIN_PATHS2 = sorted(glob.glob("data/joining/*/train_interactions.parquet"))


TFIDF_ARTIFACTS = {
    "matrix": "tfidf_matrix.npz",
    "item_ids": "tfidf_item_ids.npy",
}




class UserProfile:
    def __init__(
        self,
        verbose: bool | True,
        t_start: float | None,
        path: str | TRAIN_PATHS,
    ):
        self.verbose = verbose
        self.t_start = t_start
        self.path = path
        print(f"{path}")
        self._df: Optional[pd.DataFrame] = None

    def group_by_user_id(self):
        if self.verbose:
            print("group_by_user_id()")
            print(f"{self.path}")
            if os.path.isfile(self.path) and os.path.exists(self.path):
                train_df = self._load_dataset()
                grouped_train_df = train_df.groupby("user_id")
                if self.verbose:
                    print("user_profile.group_by_user().for.if - grouped_train_df : \n"
                        f"{grouped_train_df}\n")
                print(f"grouped_train_df end: {(time.time() - self.t_start):.1f}")
                del grouped_train_df
                gc.collect()
        print(f"group_by_user_id end: {(time.time() - self.t_start):.1f}")
    
    def _fmt_size(self, n_bytes: float) -> str:
        for unit in ("B", "KiB", "MiB", "GiB"):
            if abs(n_bytes) < 1024:
                return f"{n_bytes:.1f} {unit}"
            n_bytes /= 1024
        return f"{n_bytes:.1f} TiB"

    def _disk_size(self) -> str:
        try:
            return self._fmt_size(os.path.getsize(self.path))
        except OSError:
            return "N/A"

    def _df_memory_mb(self) -> float:
        return self._df.memory_usage(deep=True).sum() / (1024 * 1024)

    def _load_dataset(self) -> pd.DataFrame:
        if self.verbose:
            print("\nload_dataset()")
        self._df = pd.read_parquet(self.path)
        print(f"\n{self._df},\ndisk: {self._disk_size()}, \nmemory (loaded): {self._df_memory_mb():.1f} MiB, \n{self._df.shape}, \n{self._df.columns.tolist()}\n, \n{self._df}\n")
        print(f"load_dataset end: {(time.time() - self.t_start):.1f}\n")        
        return self._df





class DatasetManager:
    """Charge et expose un dataset d'interactions jointes."""
    def __init__(
        self,
        path: str | Path,
        verbose: bool | True,
    ):
        self.path = Path(path)
        self.verbose = verbose
        self._df: Optional[pd.DataFrame] = None

    
    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self._load()
        return self._df

    def _load(self) -> pd.DataFrame:
        t0 = time.perf_counter()
        df = pd.read_parquet(self.path)
        if self.verbose:
            mem = df.memory_usage(deep=True).sum() / 1024**2
            print(f"[DatasetManager] {self.path.name}: "
                  f"{len(df):,} rows, {mem:.0f} MiB, {time.perf_counter()-t0:.2f}s")
        return df
    
    def interactions_by_user(self) -> pd.core.groupby.DataFrameGroupBy:
        return self.df.groupby("user_id")
    
    def release(self):
        del self._df
        self._df = None
        gc.collect()





class ContentRepresenter:
    """Charge une matrice TF-IDF pré-calculée depuis le disque."""

    def __init__(self, artifacts_dir: str | Path, verbose: bool = True):
        self.artifacts_dir = Path(artifacts_dir)
        self.verbose = verbose
        self._matrix: Optional[csr_matrix] = None
        self._item_ids: Optional[np.ndarray] = None
        self._item_index: Optional[Dict[str, int]] = None

    @property
    def available(self) -> bool:
        """Vérifie si les artéfacts TF-IDF existent sur disque."""
        return all(
            (self.artifacts_dir / fname).exists()
            for fname in TFIDF_ARTIFACTS.values()
        )
    

    


class UserProfileBuilder:
    """Construit les profils utilisateurs à partir des interactions + TF-IDF chargé."""
    def __init__(
        self,
        dataset: DatasetManager,
        representer: ContentRepresenter,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.representer = representer
        self.verbose = verbose
    
    def build_profiles(self) -> Dict[str, np.ndarray]:
        """ Profil Utilisateur = moyenne pondérée (par rating)  """




# def main() -> None:
    # for train_path in TRAIN_PATHS:
    #     variant_dir = train_path.parent
    #     print(f"\n{'='*70}\n  {variant_dir.name}\n{'='*70}")

    #     # 1) Dataset
    #     ds = DatasetManager(train_path, verbose=True)


    #     # 2) TF-IDF pré-calculé
    #     rep = ContentRepresenter(artifacts_dir=variant_dir)
    #     if not rep.available:
    #         print(f"  [SKIP] TF-IDF introuvable dans {variant_dir}")
    #         ds.release()
    #         continue
    #     rep.load()


    #     # 3) Profils
    #     builder = UserProfileBuilder(ds, rep)
    #     profiles, user_ids = builder.build_profiles_sparse()

    #     # ... exploitation (sauvegarde, évaluation, etc.)

    #     ds.release()

    #     del rep, builder, profiles
    #     gc.collect()



def main() -> None:
    for path in TRAIN_PATHS:
        profile = UserProfile(verbose=True,
        t_start=time.time(),
        path=path)
        profile.group_by_user_id()
    return None



if __name__ == "__main__":
    main()    
