from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gc
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz


TRAIN_PATHS = sorted(Path("data/joining").glob("*/train_interactions.parquet"))

TFIDF_ARTIFACTS = {
    "matrix": "tfidf_matrix.npz",
    "item_ids": "tfidf_item_ids.npy",
}


class DatasetManager:
    """Charge et expose un dataset d'interactions jointes."""

    def __init__(self, path: str | Path, verbose: bool = True):
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

    def load(self) -> ContentRepresenter:
        """Charge la matrice et le mapping depuis le disque."""
        if not self.available:
            raise FileNotFoundError(
                f"TF-IDF artifacts missing in {self.artifacts_dir}. "
                f"Expected: {list(TFIDF_ARTIFACTS.values())}"
            )

        t0 = time.perf_counter()

        mat_path = self.artifacts_dir / TFIDF_ARTIFACTS["matrix"]
        ids_path = self.artifacts_dir / TFIDF_ARTIFACTS["item_ids"]

        self._matrix = load_npz(mat_path)
        self._item_ids = np.load(ids_path, allow_pickle=True)
        self._item_index = {asin: i for i, asin in enumerate(self._item_ids)}

        if self.verbose:
            print(f"[ContentRepresenter] loaded {mat_path.name}: "
                  f"{self._matrix.shape}, {time.perf_counter()-t0:.2f}s")
        return self

    @property
    def tfidf_matrix(self) -> csr_matrix:
        if self._matrix is None:
            raise RuntimeError("Call load() first.")
        return self._matrix

    @property
    def item_index(self) -> Dict[str, int]:
        if self._item_index is None:
            raise RuntimeError("Call load() first.")
        return self._item_index

    @property
    def n_features(self) -> int:
        return self.tfidf_matrix.shape[1]


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

    def build_profiles_sparse(self) -> Tuple[np.ndarray, List[str]]:
        """Profils via multiplication matricielle sparse.

        Returns (profiles_matrix, user_ids_list)
        où profiles_matrix[i] = profil pondéré normalisé du user user_ids_list[i].
        """
        t0 = time.perf_counter()
        df = self.dataset.df
        item_idx = self.representer.item_index
        tfidf = self.representer.tfidf_matrix

        valid_mask = df["parent_asin"].isin(item_idx)
        df_valid = df[valid_mask]

        user_ids_unique = sorted(df_valid["user_id"].unique())
        user_to_idx = {u: i for i, u in enumerate(user_ids_unique)}

        row_indices = df_valid["user_id"].map(user_to_idx).values
        col_indices = df_valid["parent_asin"].map(item_idx).values
        weights = df_valid["rating"].values.astype(np.float32)

        R = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(len(user_ids_unique), tfidf.shape[0]),
        )

        weight_sums = np.array(R.sum(axis=1)).ravel()
        weight_sums[weight_sums == 0] = 1.0

        profiles = (R @ tfidf).toarray() / weight_sums[:, np.newaxis]

        if self.verbose:
            print(f"[UserProfileBuilder] {len(user_ids_unique):,} profiles, "
                  f"shape={profiles.shape}, {time.perf_counter()-t0:.2f}s")

        return profiles, user_ids_unique


def main() -> None:
    for train_path in TRAIN_PATHS:
        variant_dir = train_path.parent
        print(f"\n{'='*70}\n  {variant_dir.name}\n{'='*70}")

        # 1) Dataset
        ds = DatasetManager(train_path)

        # 2) TF-IDF pré-calculé
        rep = ContentRepresenter(artifacts_dir=variant_dir)
        if not rep.available:
            print(f"  [SKIP] TF-IDF introuvable dans {variant_dir}")
            ds.release()
            continue
        rep.load()

        # 3) Profils
        builder = UserProfileBuilder(ds, rep)
        profiles, user_ids = builder.build_profiles_sparse()


        ds.release()
        del rep, builder, profiles
        gc.collect()


if __name__ == "__main__":
    main()