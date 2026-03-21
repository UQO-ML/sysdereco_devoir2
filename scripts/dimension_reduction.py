"""
Tâche 2 - Réduction de dimension SVD.

Ce script exécute une réduction TruncatedSVD sur la matrice TF-IDF de chaque
variant découvert dans `data/joining` et sauvegarde tous les artefacts dans
`results/svd`.

Usage en script : python dimension_reduction.py
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from sklearn.decomposition import TruncatedSVD


SEED = 42
LATENT_DIMENSIONS = [50, 100, 200, 300]
MIN_VARIANCE_PCT = 30.0
MAX_MARGINAL_GAIN_PCT = 2.0

GLOB_PATTERN = "*_clean_joined.parquet"
GLOB_SUFFIX = GLOB_PATTERN.replace("*", "")
TFIDF_FILENAME = "books_representation_sparse.npz"
RESULTS_DIR = Path("results/svd")


def load_tfidf(data_dir: Path, verbose: bool = True) -> Tuple[csr_matrix, np.ndarray]:
    """Charge la matrice TF-IDF et les IDs des items depuis data_dir."""
    t0 = time.perf_counter()
    variant = data_dir.name

    tfidf_path = data_dir / TFIDF_FILENAME
    if not tfidf_path.exists():
        raise FileNotFoundError(f"Matrice TF-IDF introuvable: {tfidf_path}")

    parquet_path = data_dir.parent / f"{variant}{GLOB_SUFFIX}"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet source introuvable: {parquet_path}")

    tfidf_matrix = load_npz(tfidf_path).tocsr()
    df_src = pd.read_parquet(parquet_path, columns=["parent_asin"])
    item_ids = df_src.drop_duplicates(subset=["parent_asin"], keep="first")["parent_asin"].values

    if len(item_ids) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"Incohérence dimensions: {len(item_ids)} item_ids vs {tfidf_matrix.shape[0]} lignes"
        )

    if verbose:
        density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
        print(
            f"[Load TF-IDF {variant}] shape={tfidf_matrix.shape}, nnz={tfidf_matrix.nnz:,}, "
            f"density={density:.6f}, {time.perf_counter() - t0:.2f}s"
        )

    return tfidf_matrix, item_ids


def run_svd(
    tfidf_matrix: csr_matrix,
    n_components: int,
    variant: str,
) -> Tuple[np.ndarray, TruncatedSVD, Dict[str, Any]]:
    """Applique TruncatedSVD et retourne la matrice réduite, le modèle et les métriques."""
    t_fit_start = time.perf_counter()

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=SEED,
        algorithm="randomized",
    )
    reduced_matrix = svd.fit_transform(tfidf_matrix)
    fit_time_s = time.perf_counter() - t_fit_start

    sample_size = min(1000, tfidf_matrix.shape[0])
    t_transform_start = time.perf_counter()
    _ = svd.transform(tfidf_matrix[:sample_size])
    transform_time_s = time.perf_counter() - t_transform_start

    explained = svd.explained_variance_ratio_
    metrics = {
        "variant": variant,
        "method": "svd",
        "n_components": n_components,
        "input_shape": list(tfidf_matrix.shape),
        "output_shape": [int(tfidf_matrix.shape[0]), int(n_components)],
        "fit_time_s": round(fit_time_s, 4),
        "transform_time_s": round(transform_time_s, 4),
        "transform_time_per_sample_ms": round((transform_time_s / sample_size) * 1000, 4),
        "variance_explained": round(float(explained.sum()), 6),
        "variance_explained_pct": round(float(explained.sum()) * 100, 2),
        "variance_per_component": [round(float(v), 6) for v in explained[:10]],
        "singular_values": [round(float(v), 4) for v in svd.singular_values_[:10]],
    }
    return reduced_matrix.astype(np.float32), svd, metrics


def analyze_tradeoffs(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule une recommandation de dimension basée sur variance/temps."""
    if not results:
        return {}

    # On impose un ordre croissant des dimensions pour que les gains marginaux
    # soient calculés entre dimensions successives.
    ordered_results = sorted(results, key=lambda r: r["n_components"])

    summary = [
        {
            "dimension": r["n_components"],
            "variance_pct": r["variance_explained_pct"],
            "fit_time_s": r["fit_time_s"],
            "transform_ms": r["transform_time_per_sample_ms"],
        }
        for r in ordered_results
    ]

    marginal_gains = []
    for i in range(1, len(ordered_results)):
        gain = (
            ordered_results[i]["variance_explained_pct"]
            - ordered_results[i - 1]["variance_explained_pct"]
        )
        marginal_gains.append(
            {
                "from_dim": ordered_results[i - 1]["n_components"],
                "to_dim": ordered_results[i]["n_components"],
                "variance_gain_pct": round(gain, 2),
            }
        )

    # Critère annoncé:
    # 1) variance expliquée > 30%
    # 2) gain marginal vers la dimension suivante < 2%
    # On recommande la première dimension qui satisfait ces deux conditions.
    recommendation = ordered_results[-1]
    recommendation_reason = (
        "Aucune dimension ne satisfait les seuils; fallback sur la variance maximale."
    )
    for i, current in enumerate(ordered_results[:-1]):
        gain_to_next = (
            ordered_results[i + 1]["variance_explained_pct"]
            - current["variance_explained_pct"]
        )
        if (
            current["variance_explained_pct"] > MIN_VARIANCE_PCT
            and gain_to_next < MAX_MARGINAL_GAIN_PCT
        ):
            recommendation = current
            recommendation_reason = (
                f"Premier point satisfaisant variance>{MIN_VARIANCE_PCT:.0f}% "
                f"et gain marginal<{MAX_MARGINAL_GAIN_PCT:.0f}%."
            )
            break

    recommended_dim = recommendation["n_components"]

    return {
        "summary": summary,
        "marginal_gains": marginal_gains,
        "recommendation": {
            "dimension": recommended_dim,
            "variance_pct": recommendation["variance_explained_pct"],
            "fit_time_s": recommendation["fit_time_s"],
            "rationale": [
                recommendation_reason,
                "Compromis entre capacité de représentation et coût de calcul.",
            ],
            "criteria": {
                "min_variance_pct": MIN_VARIANCE_PCT,
                "max_marginal_gain_pct": MAX_MARGINAL_GAIN_PCT,
            },
        },
    }


def save_dimension_artifacts(
    reduced_matrix: np.ndarray,
    svd: TruncatedSVD,
    item_ids: np.ndarray,
    metrics: Dict[str, Any],
    data_dir: Path,
    results_dir: Path,
) -> Dict[str, str]:
    """Sauvegarde les fichiers d'une dimension donnée dans results_dir et data_dir."""
    results_dir.mkdir(parents=True, exist_ok=True)
    dim = metrics["n_components"]

    reduced_path = data_dir / f"items_reduced_svd_{dim}d.npy"
    model_path = data_dir / f"reducer_svd_{dim}d.pkl"
    item_ids_path = data_dir / "item_ids.npy"
    metrics_path = results_dir / f"metrics_svd_{dim}d.json"

    np.save(reduced_path, reduced_matrix)
    with open(model_path, "wb") as handle:
        pickle.dump(svd, handle)
    np.save(item_ids_path, item_ids)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    return {
        "reduced_matrix": str(reduced_path),
        "model": str(model_path),
        "item_ids": str(item_ids_path),
        "metrics": str(metrics_path),
    }


def run_svd_pipeline(data_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Exécute tout le pipeline SVD pour un variant et écrit dans results/svd."""
    t0 = time.perf_counter()
    variant = data_dir.name
    variant_results_dir = RESULTS_DIR / variant
    tfidf_matrix, item_ids = load_tfidf(data_dir=data_dir, verbose=verbose)

    comparison_results: List[Dict[str, Any]] = []
    artifact_paths: Dict[str, Dict[str, str]] = {}

    for dim in LATENT_DIMENSIONS:
        if verbose:
            print(f"\n[{variant} SVD] Dimension {dim}D")
        reduced_matrix, svd, metrics = run_svd(
            tfidf_matrix=tfidf_matrix, n_components=dim, variant=variant
        )
        comparison_results.append(metrics)
        artifact_paths[f"{dim}d"] = save_dimension_artifacts(
            reduced_matrix=reduced_matrix,
            svd=svd,
            item_ids=item_ids,
            metrics=metrics,
            data_dir=data_dir,
            results_dir=variant_results_dir,
        )
        if verbose:
            print(
                f"  fit={metrics['fit_time_s']:.2f}s, "
                f"var={metrics['variance_explained_pct']:.2f}%"
            )

    analysis = analyze_tradeoffs(comparison_results)
    report = {
        "variant": variant,
        "method": "svd",
        "output_dir": str(variant_results_dir),
        "dimensions_tested": LATENT_DIMENSIONS,
        "comparison_results": comparison_results,
        "analysis": analysis,
        "artifact_paths": artifact_paths,
        "build_time_s": round(time.perf_counter() - t0, 2),
    }

    report_path = variant_results_dir / "dimension_comparison.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 70)
        print(f"  RÉDUCTION SVD TERMINÉE ({variant.upper()})")
        print("=" * 70)
        print(f"  Résultats: {RESULTS_DIR}")
        print(f"  Rapport: {report_path}")
        print(f"  Temps total: {report['build_time_s']:.1f}s")
        print("=" * 70 + "\n")

    return report


def main() -> None:
    for path in sorted(Path("data/joining").glob(GLOB_PATTERN)):
        variant = path.name.removesuffix(GLOB_SUFFIX)
        data_dir = path.parent / variant
        run_svd_pipeline(data_dir=data_dir, verbose=True)


if __name__ == "__main__":
    main()
