"""
Tâche 2 - Réduction de dimension SVD (temporal uniquement).

Ce script exécute une réduction TruncatedSVD sur la matrice TF-IDF du variant
`temporal_pre_split` et sauvegarde tous les artefacts dans `results/svd`.
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
TEMPORAL_VARIANT = "temporal_pre_split"
TEMPORAL_DIR = Path("data/joining") / TEMPORAL_VARIANT
RESULTS_DIR = Path("results/svd")


def load_temporal_tfidf(verbose: bool = True) -> Tuple[csr_matrix, np.ndarray]:
    """Charge la matrice TF-IDF et les IDs des items du variant temporal."""
    t0 = time.perf_counter()

    tfidf_path = TEMPORAL_DIR / "books_representation_sparse.npz"
    if not tfidf_path.exists():
        raise FileNotFoundError(f"Matrice TF-IDF introuvable: {tfidf_path}")

    parquet_path = TEMPORAL_DIR.parent / f"{TEMPORAL_VARIANT}_clean_joined.parquet"
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
            f"[Load TF-IDF temporal] shape={tfidf_matrix.shape}, nnz={tfidf_matrix.nnz:,}, "
            f"density={density:.6f}, {time.perf_counter() - t0:.2f}s"
        )

    return tfidf_matrix, item_ids


def run_svd(
    tfidf_matrix: csr_matrix,
    n_components: int,
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
        "variant": TEMPORAL_VARIANT,
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

    summary = [
        {
            "dimension": r["n_components"],
            "variance_pct": r["variance_explained_pct"],
            "fit_time_s": r["fit_time_s"],
            "transform_ms": r["transform_time_per_sample_ms"],
        }
        for r in results
    ]

    marginal_gains = []
    for i in range(1, len(results)):
        gain = results[i]["variance_explained_pct"] - results[i - 1]["variance_explained_pct"]
        marginal_gains.append(
            {
                "from_dim": results[i - 1]["n_components"],
                "to_dim": results[i]["n_components"],
                "variance_gain_pct": round(gain, 2),
            }
        )

    recommended_dim = results[len(results) // 2]["n_components"]
    recommendation = next(r for r in results if r["n_components"] == recommended_dim)

    return {
        "summary": summary,
        "marginal_gains": marginal_gains,
        "recommendation": {
            "dimension": recommended_dim,
            "variance_pct": recommendation["variance_explained_pct"],
            "fit_time_s": recommendation["fit_time_s"],
            "rationale": [
                "Compromis stable entre variance expliquée et coût de calcul",
                "Dimension médiane robuste pour le variant temporal",
            ],
        },
    }


def save_dimension_artifacts(
    reduced_matrix: np.ndarray,
    svd: TruncatedSVD,
    item_ids: np.ndarray,
    metrics: Dict[str, Any],
) -> Dict[str, str]:
    """Sauvegarde les fichiers d'une dimension donnée dans results/svd et data/ pour les .npz et .pkl."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dim = metrics["n_components"]

    reduced_path = TEMPORAL_DIR / f"items_reduced_svd_{dim}d.npy"
    model_path = TEMPORAL_DIR / f"reducer_svd_{dim}d.pkl"
    item_ids_path = TEMPORAL_DIR / "item_ids.npy"
    metrics_path = RESULTS_DIR / f"metrics_svd_{dim}d.json"

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


def run_temporal_svd_pipeline(verbose: bool = True) -> Dict[str, Any]:
    """Exécute tout le pipeline SVD sur temporal et écrit dans results/svd."""
    t0 = time.perf_counter()
    tfidf_matrix, item_ids = load_temporal_tfidf(verbose=verbose)

    comparison_results: List[Dict[str, Any]] = []
    artifact_paths: Dict[str, Dict[str, str]] = {}

    for dim in LATENT_DIMENSIONS:
        if verbose:
            print(f"\n[Temporal SVD] Dimension {dim}D")
        reduced_matrix, svd, metrics = run_svd(tfidf_matrix=tfidf_matrix, n_components=dim)
        comparison_results.append(metrics)
        artifact_paths[f"{dim}d"] = save_dimension_artifacts(
            reduced_matrix=reduced_matrix,
            svd=svd,
            item_ids=item_ids,
            metrics=metrics,
        )
        if verbose:
            print(
                f"  fit={metrics['fit_time_s']:.2f}s, "
                f"var={metrics['variance_explained_pct']:.2f}%"
            )

    analysis = analyze_tradeoffs(comparison_results)
    report = {
        "variant": TEMPORAL_VARIANT,
        "method": "svd",
        "output_dir": str(RESULTS_DIR),
        "dimensions_tested": LATENT_DIMENSIONS,
        "comparison_results": comparison_results,
        "analysis": analysis,
        "artifact_paths": artifact_paths,
        "build_time_s": round(time.perf_counter() - t0, 2),
    }

    report_path = RESULTS_DIR / "dimension_comparison.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if verbose:
        print("\n" + "=" * 70)
        print("  RÉDUCTION SVD TERMINÉE (TEMPORAL)")
        print("=" * 70)
        print(f"  Résultats: {RESULTS_DIR}")
        print(f"  Rapport: {report_path}")
        print(f"  Temps total: {report['build_time_s']:.1f}s")
        print("=" * 70 + "\n")

    return report


def main() -> None:
    run_temporal_svd_pipeline(verbose=True)


if __name__ == "__main__":
    main()
