"""
Tâche 2.0.2 - Projection des profils utilisateurs dans l'espace latent.

Ce script projette les profils utilisateurs construits en Tâche 0 (user_profile.py)
dans l'espace latent appris par SVD en Tâche 2.0.1 (dimension_reduction.py).

Contraintes expérimentales respectées:
- Profils et items dans le même espace vectoriel (même transformation SVD)
- Aucune donnée du test utilisée (seulement train_interactions.parquet)
- Projection cohérente avec celle des items

Sorties:
- latent_user_profiles : matrice (n_users × k) pour chaque dimension
- latent_item_vectors : matrice (n_items × k) déjà produite par dimension_reduction.py

Usage en script : python user_profile_projection.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, issparse

# Configuration
LATENT_DIMENSIONS = [50, 100, 200, 300]
RESULTS_DIR = Path("results/svd")
GLOB_PATTERN = "*_clean_joined.parquet"
GLOB_SUFFIX = GLOB_PATTERN.replace("*", "")

# Artéfacts d'entrée
ARTIFACTS_IN = {
    "tfidf_matrix": "books_representation_sparse.npz",
    "dimension_comparison": "dimension_comparison.json",
}

# Artéfacts de sortie
ARTIFACTS_OUT = {
    "latent_user_profiles": "user_profiles_latent_{dim}d.npy",
    "latent_item_vectors": "items_reduced_svd_{dim}d.npy",  # Référence aux items déjà créés
    "user_ids": "user_ids_latent.npy",
    "projection_report": "user_profile_projection_report.json",
}


class LatentUserProfileProjector:
    """Projette les profils utilisateurs TF-IDF dans l'espace latent SVD."""

    def __init__(
        self,
        data_dir: Path,
        results_dir: Path,
        train_path: Path,
        verbose: bool = True
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.train_path = train_path
        self.variant = data_dir.name
        self.verbose = verbose

        # Données chargées
        self._tfidf_matrix: Optional[csr_matrix] = None
        self._item_ids: Optional[np.ndarray] = None
        self._item_index: Optional[Dict[str, int]] = None
        self._train_df: Optional[pd.DataFrame] = None

    def load_artifacts(self) -> LatentUserProfileProjector:
        """Charge la matrice TF-IDF des items et les IDs."""
        if self.verbose:
            print(f"\n[Load Artifacts {self.variant}]")

        t0 = time.perf_counter()

        # Charger matrice TF-IDF des items
        tfidf_path = self.data_dir / ARTIFACTS_IN["tfidf_matrix"]
        if not tfidf_path.exists():
            raise FileNotFoundError(f"Matrice TF-IDF introuvable: {tfidf_path}")

        self._tfidf_matrix = load_npz(tfidf_path).tocsr()

        # Charger les IDs des items depuis le parquet source
        parquet_path = self.data_dir.parent / f"{self.variant}{GLOB_SUFFIX}"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet source introuvable: {parquet_path}")

        df_src = pd.read_parquet(parquet_path, columns=["parent_asin"])
        dedup = df_src.drop_duplicates(subset=["parent_asin"], keep="first")
        self._item_ids = dedup["parent_asin"].values

        # Créer l'index item_id -> row
        dedup["_row"] = np.arange(len(dedup))
        self._item_index = dict(zip(dedup["parent_asin"].values, dedup["_row"].values))

        if len(self._item_ids) != self._tfidf_matrix.shape[0]:
            raise ValueError(
                f"Incohérence dimensions: {len(self._item_ids)} item_ids vs "
                f"{self._tfidf_matrix.shape[0]} lignes TF-IDF"
            )

        # Charger les interactions d'entraînement
        if not self.train_path.exists():
            raise FileNotFoundError(f"Train interactions introuvable: {self.train_path}")

        self._train_df = pd.read_parquet(
            self.train_path,
            columns=["user_id", "parent_asin", "rating"]
        )

        if self.verbose:
            density = self._tfidf_matrix.nnz / (
                self._tfidf_matrix.shape[0] * self._tfidf_matrix.shape[1]
            )
            print(f"  TF-IDF shape: {self._tfidf_matrix.shape}, "
                  f"density={density:.6f}")
            print(f"  Items: {len(self._item_ids):,}")
            print(f"  Train interactions: {len(self._train_df):,}")
            print(f"  Time: {time.perf_counter() - t0:.2f}s")

        return self

    def build_tfidf_profiles(self) -> Tuple[csr_matrix, List[str]]:
        """Construit les profils utilisateurs dans l'espace TF-IDF.

        Algorithme:
        1. Filtrer les interactions pour ne garder que les items présents dans la matrice
        2. Construire une matrice de pondération R (users × items) où R[u,i] = rating
        3. Calculer les profils: P = (R @ TF-IDF) / sum(ratings)

        Returns:
            profiles_tfidf: matrice sparse (n_users × n_features_tfidf)
            user_ids: liste des user_ids correspondant aux lignes
        """
        if self.verbose:
            print(f"\n[Build TF-IDF Profiles]")

        t0 = time.perf_counter()

        # Filtrer pour ne garder que les items présents
        valid_mask = self._train_df["parent_asin"].isin(self._item_index)
        df_valid = self._train_df[valid_mask].copy()

        if self.verbose:
            print(f"  Valid interactions: {len(df_valid):,} / {len(self._train_df):,} "
                  f"({100*len(df_valid)/len(self._train_df):.1f}%)")

        # Récupérer tous les utilisateurs uniques
        user_ids = sorted(df_valid["user_id"].unique())
        user_to_idx = {u: i for i, u in enumerate(user_ids)}

        # Construire la matrice de pondération R
        row = df_valid["user_id"].map(user_to_idx).values
        col = df_valid["parent_asin"].map(self._item_index).values
        weights = df_valid["rating"].values.astype(np.float32)

        R = csr_matrix(
            (weights, (row, col)),
            shape=(len(user_ids), self._tfidf_matrix.shape[0]),
        )

        # Calculer les profils: P = (R @ TF-IDF) / sum(ratings)
        profiles_tfidf = (R @ self._tfidf_matrix).tocsr()

        # Normaliser par la somme des poids
        weight_sums = np.array(R.sum(axis=1)).ravel()
        weight_sums[weight_sums == 0] = 1.0

        profiles_tfidf.data = profiles_tfidf.data.astype(np.float32, copy=False)
        inv = (1.0 / weight_sums).astype(np.float32)
        row_ids = np.repeat(
            np.arange(profiles_tfidf.shape[0], dtype=np.int32),
            np.diff(profiles_tfidf.indptr)
        )
        profiles_tfidf.data *= inv[row_ids]

        if self.verbose:
            density = profiles_tfidf.nnz / (
                profiles_tfidf.shape[0] * profiles_tfidf.shape[1]
            )
            print(f"  Profiles shape: {profiles_tfidf.shape}")
            print(f"  Users: {len(user_ids):,}")
            print(f"  Density: {density:.6f}")
            print(f"  Time: {time.perf_counter() - t0:.2f}s")

        return profiles_tfidf, user_ids

    def project_profiles_to_latent(
        self,
        profiles_tfidf: csr_matrix,
        user_ids: List[str],
        dimension: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Projette les profils TF-IDF dans l'espace latent en utilisant le modèle SVD.

        Args:
            profiles_tfidf: profils dans l'espace TF-IDF (n_users × n_features_tfidf)
            user_ids: liste des user_ids
            dimension: dimension latente (50, 100, 200, 300)

        Returns:
            latent_profiles: matrice dense (n_users × dimension)
            metrics: dictionnaire de métriques
        """
        if self.verbose:
            print(f"\n[Project to Latent Space {dimension}D]")

        t0 = time.perf_counter()

        # Charger le modèle SVD
        model_path = self.results_dir / f"reducer_svd_{dimension}d.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle SVD introuvable: {model_path}")

        with open(model_path, "rb") as f:
            svd_model = pickle.load(f)

        # Projeter les profils
        # Les profils TF-IDF sont dans le même espace que les items TF-IDF
        # On peut donc directement appliquer la transformation SVD
        latent_profiles = svd_model.transform(profiles_tfidf)
        latent_profiles = latent_profiles.astype(np.float32)

        transform_time = time.perf_counter() - t0

        # Vérifier que les items projetés existent
        items_latent_path = self.results_dir / f"items_reduced_svd_{dimension}d.npy"
        if not items_latent_path.exists():
            raise FileNotFoundError(f"Items latents introuvables: {items_latent_path}")

        items_latent = np.load(items_latent_path)

        metrics = {
            "dimension": dimension,
            "n_users": len(user_ids),
            "n_items": items_latent.shape[0],
            "profile_shape": list(latent_profiles.shape),
            "item_shape": list(items_latent.shape),
            "transform_time_s": round(transform_time, 4),
            "transform_time_per_user_ms": round((transform_time / len(user_ids)) * 1000, 4),
            "same_vector_space": latent_profiles.shape[1] == items_latent.shape[1],
            "variance_explained": float(svd_model.explained_variance_ratio_.sum()),
            "variance_explained_pct": round(float(svd_model.explained_variance_ratio_.sum()) * 100, 2),
        }

        if self.verbose:
            print(f"  Latent profiles shape: {latent_profiles.shape}")
            print(f"  Latent items shape: {items_latent.shape}")
            print(f"  Same vector space: {metrics['same_vector_space']}")
            print(f"  Variance explained: {metrics['variance_explained_pct']:.2f}%")
            print(f"  Transform time: {transform_time:.4f}s")

        return latent_profiles, metrics

    def save_latent_profiles(
        self,
        latent_profiles: np.ndarray,
        user_ids: List[str],
        dimension: int,
        metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """Sauvegarde les profils latents et les métadonnées."""
        self.results_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Sauvegarder les profils latents
        profiles_path = self.results_dir / f"user_profiles_latent_{dimension}d.npy"
        np.save(profiles_path, latent_profiles)
        paths["latent_user_profiles"] = str(profiles_path)

        # Sauvegarder les user_ids (une seule fois, identiques pour toutes les dimensions)
        user_ids_path = self.results_dir / "user_ids_latent.npy"
        if not user_ids_path.exists():
            np.save(user_ids_path, np.array(user_ids))
            paths["user_ids"] = str(user_ids_path)

        # Sauvegarder les métriques
        metrics_path = self.results_dir / f"user_profile_projection_{dimension}d.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        paths["metrics"] = str(metrics_path)

        if self.verbose:
            for name, path in paths.items():
                size = Path(path).stat().st_size / 1024 / 1024
                print(f"  Saved {name}: {Path(path).name} ({size:.1f} MiB)")

        return paths


def run_projection_pipeline(
    data_dir: Path,
    train_path: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """Exécute le pipeline complet de projection des profils utilisateurs."""
    t0 = time.perf_counter()
    variant = data_dir.name
    results_dir = RESULTS_DIR / variant

    if verbose:
        print("\n" + "=" * 70)
        print(f"  PROJECTION DES PROFILS UTILISATEURS ({variant.upper()})")
        print("=" * 70)

    # Initialiser le projecteur
    projector = LatentUserProfileProjector(
        data_dir=data_dir,
        results_dir=results_dir,
        train_path=train_path,
        verbose=verbose
    )

    # Charger les artéfacts nécessaires
    projector.load_artifacts()

    # Construire les profils TF-IDF
    profiles_tfidf, user_ids = projector.build_tfidf_profiles()

    # Projeter dans chaque dimension latente
    all_metrics = []
    artifact_paths = {}

    for dim in LATENT_DIMENSIONS:
        latent_profiles, metrics = projector.project_profiles_to_latent(
            profiles_tfidf=profiles_tfidf,
            user_ids=user_ids,
            dimension=dim
        )

        paths = projector.save_latent_profiles(
            latent_profiles=latent_profiles,
            user_ids=user_ids,
            dimension=dim,
            metrics=metrics
        )

        all_metrics.append(metrics)
        artifact_paths[f"{dim}d"] = paths

    # Rapport final
    report = {
        "variant": variant,
        "method": "user_profile_projection",
        "output_dir": results_dir.as_posix(),
        "dimensions_tested": LATENT_DIMENSIONS,
        "projection_results": all_metrics,
        "artifact_paths": artifact_paths,
        "constraints_satisfied": {
            "same_vector_space": all(m["same_vector_space"] for m in all_metrics),
            "no_test_data_used": "Only train_interactions.parquet used",
            "consistent_with_items": "SVD model applied to user profiles in TF-IDF space",
        },
        "build_time_s": round(time.perf_counter() - t0, 2),
    }

    # Sauvegarder le rapport complet
    report_path = results_dir / "user_profile_projection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 70)
        print(f"  PROJECTION TERMINÉE ({variant.upper()})")
        print("=" * 70)
        print(f"  Résultats: {results_dir}")
        print(f"  Rapport: {report_path}")
        print(f"  Temps total: {report['build_time_s']:.1f}s")
        print("=" * 70 + "\n")

    return report


def main() -> None:
    """Point d'entrée principal."""
    # Découvrir tous les variants disponibles
    variants = []
    for train_path in sorted(Path("data/joining").glob("*/train_interactions.parquet")):
        variant_name = train_path.parent.name
        data_dir = Path("data/joining") / variant_name

        # Vérifier que les artéfacts SVD existent
        svd_results_dir = RESULTS_DIR / variant_name
        if not (svd_results_dir / "dimension_comparison.json").exists():
            print(f"[SKIP] {variant_name}: artéfacts SVD manquants")
            continue

        variants.append((data_dir, train_path))

    if not variants:
        print("Aucun variant trouvé avec les artéfacts nécessaires.")
        print("Exécutez d'abord dimension_reduction.py")
        return

    # Projeter les profils pour chaque variant
    for data_dir, train_path in variants:
        report = run_projection_pipeline(
            data_dir=data_dir,
            train_path=train_path,
            verbose=True
        )

        if report:
            print(f"\n--- Résumé {report['variant']} ---")
            for metrics in report["projection_results"]:
                print(f"  {metrics['dimension']}D: {metrics['n_users']:,} users, "
                      f"variance={metrics['variance_explained_pct']:.2f}%, "
                      f"time={metrics['transform_time_s']:.4f}s")


if __name__ == "__main__":
    main()
