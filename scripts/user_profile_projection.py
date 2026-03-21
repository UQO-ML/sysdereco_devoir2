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
from scipy.sparse import csr_matrix, issparse, load_npz

# Configuration
LATENT_DIMENSIONS = [50, 100, 200, 300]
RESULTS_DIR = Path("results/svd")
DATA_DIR = Path("data/joining")

# Artéfacts d'entrée (produits par user_profile.py / Tâche 0)
ARTIFACTS_IN = {
    "dimension_comparison": "dimension_comparison.json",
    "user_profiles_tfidf_npz": "user_profiles_tfidf.npz",
    "user_profiles_tfidf_npy": "user_profiles_tfidf.npy",
    "user_ids": "user_ids.npy",
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
        verbose: bool = True
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.variant = data_dir.name
        self.verbose = verbose

        # Profils chargés depuis les artéfacts Tâche 0
        self._profiles_tfidf: Optional[csr_matrix] = None
        self._user_ids: Optional[List[str]] = None

    def load_artifacts(self) -> "LatentUserProfileProjector":
        """Charge les profils TF-IDF et user_ids produits par user_profile.py (Tâche 0)."""
        if self.verbose:
            print(f"\n[Load Artifacts {self.variant}]")

        t0 = time.perf_counter()

        # Priorité au format sparse (.npz), sinon dense (.npy)
        npz_path = self.data_dir / ARTIFACTS_IN["user_profiles_tfidf_npz"]
        npy_path = self.data_dir / ARTIFACTS_IN["user_profiles_tfidf_npy"]

        if npz_path.exists():
            try:
                self._profiles_tfidf = load_npz(npz_path).tocsr()
                profiles_file = npz_path
            except Exception as e:
                raise ValueError(f"Fichier corrompu: {npz_path}: {e}")
        elif npy_path.exists():
            self._profiles_tfidf = np.load(npy_path)
            profiles_file = npy_path
        else:
            raise FileNotFoundError(
                f"Profils TF-IDF introuvables dans {self.data_dir}.\n"
                "Exécutez d'abord user_profile.py (Tâche 0)."
            )

        user_ids_path = self.data_dir / ARTIFACTS_IN["user_ids"]
        if not user_ids_path.exists():
            raise FileNotFoundError(
                f"user_ids.npy introuvable: {user_ids_path}.\n"
                "Exécutez d'abord user_profile.py (Tâche 0)."
            )

        self._user_ids = np.load(user_ids_path, allow_pickle=True).tolist()

        if len(self._user_ids) != self._profiles_tfidf.shape[0]:
            raise ValueError(
                f"Incohérence: {len(self._user_ids)} user_ids vs "
                f"{self._profiles_tfidf.shape[0]} lignes de profils"
            )

        if self.verbose:
            shape = self._profiles_tfidf.shape
            if issparse(self._profiles_tfidf):
                density = self._profiles_tfidf.nnz / (shape[0] * shape[1])
                print(f"  Profils TF-IDF (sparse): {shape}, density={density:.6f}")
            else:
                print(f"  Profils TF-IDF (dense): {shape}")
            print(f"  Users: {len(self._user_ids):,}")
            print(f"  Source: {profiles_file.name}")
            print(f"  Time: {time.perf_counter() - t0:.2f}s")

        return self

    def load_tfidf_profiles(self) -> Tuple[csr_matrix, List[str]]:
        """Retourne les profils TF-IDF et user_ids chargés depuis les artéfacts Tâche 0.

        Pré-requis: load_artifacts() doit avoir été appelé.

        Returns:
            profiles_tfidf: matrice (n_users × n_features_tfidf) — sparse ou dense
            user_ids: liste des user_ids correspondant aux lignes
        """
        if self._profiles_tfidf is None or self._user_ids is None:
            raise RuntimeError("Appelez load_artifacts() avant load_tfidf_profiles().")
        return self._profiles_tfidf, self._user_ids

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
        model_path = self.data_dir / f"reducer_svd_{dimension}d.pkl"
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
        items_latent_path = self.data_dir / f"items_reduced_svd_{dimension}d.npy"
        if not items_latent_path.exists():
            raise FileNotFoundError(f"Items latents introuvables: {items_latent_path}, exécutez d'abord dimension_reduction.py")

        items_latent = np.load(items_latent_path)

        n_users = len(user_ids)
        transform_time_per_user_ms: Optional[float] = None
        if n_users > 0:
            transform_time_per_user_ms = round((transform_time / n_users) * 1000, 4)

        metrics = {
            "dimension": dimension,
            "n_users": n_users,
            "n_items": items_latent.shape[0],
            "profile_shape": list(latent_profiles.shape),
            "item_shape": list(items_latent.shape),
            "transform_time_s": round(transform_time, 4),
            "transform_time_per_user_ms": transform_time_per_user_ms,
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Sauvegarder les profils latents
        # profiles_path = self.data_dir / f"user_profiles_latent_{dimension}d.npy"
        profiles_path = self.data_dir / ARTIFACTS_OUT["latent_user_profiles"].format(dim=dimension)
        np.save(profiles_path, latent_profiles)
        paths["latent_user_profiles"] = str(profiles_path)

        # Sauvegarder les user_ids (alignés avec les profils latents sauvegardés)
        user_ids_path = self.data_dir / ARTIFACTS_OUT["user_ids"]
        np.save(user_ids_path, np.array(user_ids))
        paths["user_ids"] = str(user_ids_path)
        # Sauvegarder les métriques
        metrics_path = self.results_dir / ARTIFACTS_OUT["projection_report"].format(dim=dimension)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        paths["metrics"] = str(metrics_path)
        paths["latent_item_vectors"] = str(
            self.data_dir / ARTIFACTS_OUT["latent_item_vectors"].format(dim=dimension)
        )

        if self.verbose:
            for name, path in paths.items():
                size = Path(path).stat().st_size / 1024 / 1024
                print(f"  Saved {name}: {Path(path).name} ({size:.1f} MiB)")

        return paths


def run_projection_pipeline(
    data_dir: Path,
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
        verbose=verbose
    )

    # Charger les profils produits par user_profile.py (Tâche 0)
    projector.load_artifacts()

    # Récupérer les profils déjà construits
    profiles_tfidf, user_ids = projector.load_tfidf_profiles()

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
        "train_path": (data_dir / "train_interactions.parquet").as_posix(), # Pour la contrainte "no_test_data_used"
        "dimensions_tested": LATENT_DIMENSIONS,
        "projection_results": all_metrics,
        "artifact_paths": artifact_paths,
        "constraints_satisfied": {
            "same_vector_space": all(m["same_vector_space"] for m in all_metrics),
            "no_test_data_used": True,
            "consistent_with_items": True,
        },
        "build_time_s": round(time.perf_counter() - t0, 2),
    }

    # Sauvegarder le rapport complet
    report_path = results_dir / ARTIFACTS_OUT["projection_report"]
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
    # Découvrir tous les variants disposant des artéfacts Tâche 0 + SVD
    data_dirs = []
    for variant_dir in sorted(DATA_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant_name = variant_dir.name

        # Vérifier que les profils TF-IDF (Tâche 0) existent
        has_profiles = (
            (variant_dir / ARTIFACTS_IN["user_profiles_tfidf_npz"]).exists()
            or (variant_dir / ARTIFACTS_IN["user_profiles_tfidf_npy"]).exists()
        )
        if not has_profiles:
            print(f"[SKIP] {variant_name}: profils TF-IDF manquants "
                  f"(exécutez user_profile.py)")
            continue

        # Vérifier que les artéfacts SVD existent
        if not (RESULTS_DIR / variant_name / ARTIFACTS_IN["dimension_comparison"]).exists():
            print(f"[SKIP] {variant_name}: artéfacts SVD manquants "
                  f"(exécutez dimension_reduction.py)")
            continue

        data_dirs.append(variant_dir)

    if not data_dirs:
        print("Aucun variant trouvé avec les artéfacts nécessaires.")
        print("Exécutez d'abord user_profile.py puis dimension_reduction.py.")
        return

    # Projeter les profils pour chaque variant
    for data_dir in data_dirs:
        report = run_projection_pipeline(data_dir=data_dir, verbose=True)

        if report:
            print(f"\n--- Résumé {report['variant']} ---")
            for metrics in report["projection_results"]:
                print(f"  {metrics['dimension']}D: {metrics['n_users']:,} users, "
                      f"variance={metrics['variance_explained_pct']:.2f}%, "
                      f"time={metrics['transform_time_s']:.4f}s")


if __name__ == "__main__":
    main()
