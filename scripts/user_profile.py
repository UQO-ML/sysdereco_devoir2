from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gc
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, hstack, issparse, save_npz


# -- Configuration --------------------------------------------------

TRAIN_PATHS = sorted(Path("data/joining").glob("*/train_interactions.parquet"))

INTERACTION_COLS = ["user_id", "parent_asin", "rating", "text"]

NOTEBOOK_ARTIFACTS_DIR = Path("data/tfidf")
NOTEBOOK_NPZ = "books_representation_sparse.npz"
NOTEBOOK_SOURCE_PARQUET = Path("data/joining/active_pre_split_clean_joined.parquet")
NOTEBOOK_ARTIFACTS_IN = {"tfidf_matrix": NOTEBOOK_NPZ}

ARTIFACTS_IN = {
    "tfidf_matrix": "tfidf_matrix.npz",
    "svd_matrix": "tfidf_svd_matrix.npy",
    "item_ids": "tfidf_item_ids.npy",
    "numeric_features": "numeric_features.npy",
}

ARTIFACTS_OUT = {
    "profiles_tfidf_npy": "user_profiles_tfidf.npy",
    "profiles_tfidf_npz": "user_profiles_tfidf.npz",
    "profiles_svd": "user_profiles_svd.npy",
    "user_ids": "user_ids.npy",
    "report": "user_profiles_report.json",
}

"""
MIN_INTERACTIONS_FOR_PROFILE = 1 rend le mécanisme cold-start (users avec < min_interactions) inopérant dans la pratique, 
car tout user présent dans le train a au moins 1 interaction. 
Si l’objectif est de gérer les historiques trop courts, 
augmenter ce seuil (ex. 3 comme le split temporel) ou ajuster la logique/description pour refléter le comportement réel.
"""
MIN_INTERACTIONS_FOR_PROFILE = 1


# -- DatasetManager -------------------------------------------------

class DatasetManager:
    """Charge le train parquet (colonnes d'interaction uniquement)."""

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
        df = pd.read_parquet(self.path, columns=INTERACTION_COLS)
        if self.verbose:
            mem = df.memory_usage(deep=True).sum() / 1024**2
            print(f"[DatasetManager] {self.path.name}: "
                  f"{len(df):,} rows, {mem:.0f} MiB, {time.perf_counter()-t0:.2f}s")
        return df

    def release(self):
        del self._df
        self._df = None
        gc.collect()


# -- ItemRepresentationLoader --------------------------------------

class ItemRepresentationLoader:
    """Charge les artéfacts produits par item_representation.py."""

    def __init__(self, artifacts_dir: str | Path, verbose: bool = True, notebook_mode: bool = True):
        self.dir = Path(artifacts_dir)
        self.verbose = verbose
        self.notebook_mode = notebook_mode
        self._tfidf: Optional[csr_matrix] = None
        self._svd: Optional[np.ndarray] = None
        self._numeric: Optional[np.ndarray] = None
        self._item_ids: Optional[np.ndarray] = None
        self._item_index: Optional[Dict[str, int]] = None

    @property
    def available(self) -> bool:
        if self.notebook_mode:
            return (self.dir / NOTEBOOK_ARTIFACTS_IN["tfidf_matrix"]).exists() and \
                NOTEBOOK_SOURCE_PARQUET.exists()
        else:
            return (self.dir / ARTIFACTS_IN["tfidf_matrix"]).exists() and \
               (self.dir / ARTIFACTS_IN["item_ids"]).exists()

    def load(self) -> ItemRepresentationLoader:
        if not self.available:
            raise FileNotFoundError(f"Artéfacts manquants dans {self.dir}")

        t0 = time.perf_counter()

        if self.notebook_mode:
            self._tfidf = load_npz(self.dir / NOTEBOOK_ARTIFACTS_IN["tfidf_matrix"])

            df_src = pd.read_parquet(NOTEBOOK_SOURCE_PARQUET, columns=["parent_asin"])

            if (len(df_src) != self._tfidf.shape[0]):
                print("La matrice ne correspond pas au Dataset")
                raise FileNotFoundError

            df_src["_row"] = np.arange(len(df_src))
            dedup = df_src.drop_duplicates(subset=["parent_asin"], keep="first")
            self._item_index = dict(zip(dedup["parent_asin"].values, dedup["_row"].values))
            self._item_ids = dedup["parent_asin"].values
            self._svd = None
            self._numeric = None

        else:
            self._item_ids = np.load(self.dir / ARTIFACTS_IN["item_ids"], allow_pickle=True)
            self._item_index = {asin: i for i, asin in enumerate(self._item_ids)}

            self._tfidf = load_npz(self.dir / ARTIFACTS_IN["tfidf_matrix"])

            svd_path = self.dir / ARTIFACTS_IN["svd_matrix"]
            if svd_path.exists():
                self._svd = np.load(svd_path)

            num_path = self.dir / ARTIFACTS_IN["numeric_features"]
            if num_path.exists():
                self._numeric = np.load(num_path)

        if self.verbose:
            parts = [f"tfidf={self._tfidf.shape}"]
            if self._svd is not None:
                parts.append(f"svd={self._svd.shape}")
            if self._numeric is not None:
                parts.append(f"numeric={self._numeric.shape}")
            print(f"[ItemRepLoader] {', '.join(parts)}, {time.perf_counter()-t0:.2f}s")

        return self

    @property
    def item_index(self) -> Dict[str, int]:
        return self._item_index

    @property
    def n_items(self) -> int:
        return len(self._item_ids)

    def get_matrix(self, mode: str = "tfidf") -> np.ndarray | csr_matrix:
        """Retourne la matrice items selon le mode demandé.

        Modes :
          - "tfidf"          : sparse TF-IDF (Tâche 1)
          - "svd"            : dense SVD (Tâche 2)
          - "tfidf+numeric"  : TF-IDF concaténé avec features numériques
          - "svd+numeric"    : SVD concaténé avec features numériques
        """
        if mode == "tfidf":
            return self._tfidf
        elif mode == "svd":
            if self._svd is None:
                raise RuntimeError("SVD matrix not available.")
            return self._svd
        elif mode == "tfidf+numeric":
            if self._numeric is None:
                return self._tfidf
            return hstack([self._tfidf, csr_matrix(self._numeric)], format="csr")
        elif mode == "svd+numeric":
            base = self._svd if self._svd is not None else self._tfidf.toarray()
            if self._numeric is None:
                return base
            return np.hstack([base, self._numeric])
        else:
            raise ValueError(f"Mode inconnu: {mode}")


# -- UserProfileBuilder --------------------------------------------

class UserProfileBuilder:
    """Construit les profils utilisateurs par moyenne pondérée des représentations items.

    Choix documentés (section 3.1.4 du PDF) :
      1. Items utilisés : tous les items notés dans le train (note ≥ 1)
      2. Pondération : rating utilisé comme poids (un livre noté 5 pèse 5× plus qu'un livre noté 1)
      3. Historique limité : les users avec < MIN_INTERACTIONS interactions reçoivent un profil
         "moyen global" (centroïde de tous les items) — cold-start fallback
      4. Hypothèses : le profil est une combinaison linéaire des items ; les goûts sont stables
         dans le temps (pas de décroissance temporelle) ; tous les ratings > 0 sont considérés
         comme indicateurs de préférence (y compris les notes basses)
    """

    def __init__(
        self,
        dataset: DatasetManager,
        item_loader: ItemRepresentationLoader,
        mode: str = "tfidf",
        min_interactions: int = MIN_INTERACTIONS_FOR_PROFILE,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.item_loader = item_loader
        self.mode = mode
        self.min_interactions = min_interactions
        self.verbose = verbose

    def build(self) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Construit les profils et retourne (profiles_matrix, user_ids, report)."""
        t0 = time.perf_counter()
        df = self.dataset.df
        item_idx = self.item_loader.item_index
        item_mat = self.item_loader.get_matrix(self.mode)

        # is_sparse = hasattr(item_mat, "toarray")
        is_sparse = issparse(item_mat)
        # n_features = item_mat.shape[1]

        valid_mask = df["parent_asin"].isin(item_idx)
        df_valid = df[valid_mask].copy()

        user_counts = df_valid.groupby("user_id").size()
        active_users = set(user_counts[user_counts >= self.min_interactions].index)
        cold_users = set(user_counts[user_counts < self.min_interactions].index)

        all_users = sorted(active_users | cold_users)
        user_to_idx = {u: i for i, u in enumerate(all_users)}

        # Profils actifs via multiplication matricielle
        df_active = df_valid[df_valid["user_id"].isin(active_users)]

        row = df_active["user_id"].map(user_to_idx).values
        col = df_active["parent_asin"].map(item_idx).values
        weights = df_active["rating"].values.astype(np.float32)

        R = csr_matrix(
            (weights, (row, col)),
            shape=(len(all_users), item_mat.shape[0]),
        )

        weight_sums = np.array(R.sum(axis=1)).ravel()
        weight_sums[weight_sums == 0] = 1.0

        if is_sparse:
            # profiles = (R @ item_mat).toarray() / weight_sums[:, np.newaxis]
            profiles = (R @ item_mat).tocsr()
            profiles.data = profiles.data.astype(np.float32, copy=False)
            inv = (1.0 / weight_sums).astype(np.float32)
            row_ids = np.repeat(
                np.arange(profiles.shape[0], dtype=np.int32),
                np.diff(profiles.indptr)
            )
            profiles.data *= inv[row_ids]

        else:
            profiles = (R @ item_mat) / weight_sums[:, np.newaxis]
            profiles = profiles.astype(np.float32, copy=False)


        # Cold-start : remplacer par le centroïde global
        if cold_users:
            if is_sparse:
                centroid = np.array(item_mat.mean(axis=0)).ravel().astype(np.float32)
            else:
                centroid = item_mat.mean(axis=0).astype(np.float32)
            for uid in cold_users:
                profiles[user_to_idx[uid]] = centroid

        elapsed = time.perf_counter() - t0

        report = {
            "mode": self.mode,
            "n_users_total": len(all_users),
            "n_users_active": len(active_users),
            "n_users_cold_start": len(cold_users),
            "min_interactions_threshold": self.min_interactions,
            "cold_start_strategy": "centroïde global (moyenne de tous les items)",
            "weighting": "rating comme poids (profil = Σ(rating_i × item_vec_i) / Σ(rating_i))",
            "profile_shape": list(profiles.shape),
            "items_used": "tous les items notés dans le train",
            "hypotheses": [
                "Combinaison linéaire des vecteurs items pondérée par rating",
                "Goûts stables dans le temps (pas de décroissance temporelle)",
                "Tous les ratings ≥ 1 comptent comme préférence positive",
            ],
            "build_time_s": round(elapsed, 2),
            "profile_storage": "sparse_csr_npz" if is_sparse else "dense_npy",
            "profile_nnz": int(profiles.nnz) if is_sparse else None,
            "profile_density": round(float(profiles.nnz) / (profiles.shape[0] * profiles.shape[1]), 8) if is_sparse else 1.0,
        }

        if self.verbose:
            print(f"[UserProfileBuilder:{self.mode}] "
                  f"{len(active_users):,} actifs + {len(cold_users):,} cold-start, "
                  f"shape={profiles.shape}, {elapsed:.2f}s\n")

        return profiles, all_users, report


# -- Persistance ---------------------------------------------------

def save_profiles(
    out_dir: Path,
    profiles_tfidf: np.ndarray,
    profiles_svd: Optional[np.ndarray],
    user_ids: List[str],
    report: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    if issparse(profiles_tfidf):
        save_npz(out_dir / ARTIFACTS_OUT["profiles_tfidf_npz"], profiles_tfidf.tocsr())
        paths["profiles_tfidf_npz"] = str(out_dir / ARTIFACTS_OUT["profiles_tfidf_npz"])
    else:
        np.save(out_dir / ARTIFACTS_OUT["profiles_tfidf_npy"], profiles_tfidf)
        paths["profiles_tfidf_npy"] = str(out_dir / ARTIFACTS_OUT["profiles_tfidf_npy"])

    if profiles_svd is not None:
        np.save(out_dir / ARTIFACTS_OUT["profiles_svd"], profiles_svd)
        paths["profiles_svd"] = str(out_dir / ARTIFACTS_OUT["profiles_svd"])

    np.save(out_dir / ARTIFACTS_OUT["user_ids"], np.array(user_ids))
    paths["user_ids"] = str(out_dir / ARTIFACTS_OUT["user_ids"])

    with open(out_dir / ARTIFACTS_OUT["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    paths["report"] = str(out_dir / ARTIFACTS_OUT["report"])

    if verbose:
        for name, p in paths.items():
            size = Path(p).stat().st_size / 1024 / 1024
            print(f"  saved {name}: {p} ({size:.1f} MiB)")

    return paths


def profiles_exist(variant_dir: Path) -> bool:
    return all((variant_dir / v).exists() for v in ARTIFACTS_OUT.values())


# -- Pipeline principal --------------------------------------------

def build_all_profiles(
    train_path: Path,
    force: bool = False,
    verbose: bool = True,
    notebook_mode: bool = True
) -> Dict[str, Any]:
    variant_dir = train_path.parent
    variant = variant_dir.name

    if not force and profiles_exist(variant_dir):
        if verbose:
            print(f"[{variant}] Profils déjà présents — skip")
        with open(variant_dir / ARTIFACTS_OUT["report"], encoding="utf-8") as f:
            return json.load(f)

    if verbose:
        print(f"\n{'='*70}\n  {variant} — Construction des profils utilisateurs\n{'='*70}")

    ds = DatasetManager(train_path, verbose=verbose)

    if notebook_mode:
        loader = ItemRepresentationLoader(artifacts_dir=NOTEBOOK_ARTIFACTS_DIR, verbose=verbose, notebook_mode=notebook_mode)
    else:
        loader = ItemRepresentationLoader(artifacts_dir=variant_dir, verbose=verbose, notebook_mode=notebook_mode)

    if not loader.available:
        print(f"  [SKIP] Artéfacts manquants dans {loader.dir}")
        return {}

    loader.load()

    # Profils TF-IDF (Tâche 1)
    builder_tfidf = UserProfileBuilder(ds, loader, mode="tfidf", verbose=verbose)
    profiles_tfidf, user_ids, report_tfidf = builder_tfidf.build()

    # Profils SVD (Tâche 2)
    profiles_svd = None
    report_svd = {}
    try:
        builder_svd = UserProfileBuilder(ds, loader, mode="svd", verbose=verbose)
        profiles_svd, _, report_svd = builder_svd.build()
    except RuntimeError:
        if verbose:
            print("  [INFO] SVD non disponible, profils SVD ignorés")

    combined_report = {
        "variant": variant,
        "tfidf_profiles": report_tfidf,
        "svd_profiles": report_svd,
    }

    save_profiles(variant_dir, profiles_tfidf, profiles_svd, user_ids,
                  combined_report, verbose=verbose)

    ds.release()
    del loader, profiles_tfidf, profiles_svd
    gc.collect()

    return combined_report


def main() -> None:
    t0 = time.perf_counter()
    for train_path in TRAIN_PATHS:
        t1 = time.perf_counter()
        report = build_all_profiles(train_path, force=True, verbose=True, notebook_mode=True)
        if report:
            tfidf_r = report.get("tfidf_profiles", {})
            svd_r = report.get("svd_profiles", {})
            print(f"\n--- Résumé {report.get('variant', '?')} ---")
            print(f"  TF-IDF: {tfidf_r.get('n_users_active', '?')} actifs, "
                  f"{tfidf_r.get('n_users_cold_start', '?')} cold-start, "
                  f"shape={tfidf_r.get('profile_shape')}"
                  f"\n Elapsed: {(time.perf_counter() - t1):.1f}s")
            if svd_r:
                print(f"  SVD:    {svd_r.get('n_users_active', '?')} actifs, "
                      f"shape={svd_r.get('profile_shape')}")
    print(f"\n Total elapse: {(time.perf_counter() - t0):.1f}s")

if __name__ == "__main__":
    main()