"""
Benchmark comparatif SVD / NMF / PCA sur un sous-ensemble contrôlé.

Le but n'est pas de remplacer le pipeline principal SVD, mais de produire une
comparaison empirique reproductible entre méthodes sur une matrice où PCA
reste faisable en mémoire.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from dimension_reduction import (
    CLEAN_DATASETS_PATHS,
    SEED,
    apply_dimension_reduction,
    estimate_dense_memory_gb,
    load_tfidf_matrix,
)

BENCHMARK_METHODS = ("svd", "nmf", "pca")
BENCHMARK_DIMENSIONS = [50, 100, 200]
MAX_BENCHMARK_ITEMS = 10_000
MAX_BENCHMARK_FEATURES = 5_000
BENCHMARK_OUTPUT = "method_comparison_benchmark.json"
GLOBAL_SUMMARY_OUTPUT = Path("data/joining/method_comparison_summary.json")


def build_benchmark_subset(
    tfidf_matrix: csr_matrix,
    max_items: int = MAX_BENCHMARK_ITEMS,
    max_features: int = MAX_BENCHMARK_FEATURES,
) -> tuple[csr_matrix, Dict[str, Any]]:
    """Construit un sous-ensemble représentatif et compatible avec PCA."""
    rng = np.random.default_rng(SEED)

    if tfidf_matrix.shape[0] > max_items:
        row_idx = np.sort(rng.choice(tfidf_matrix.shape[0], size=max_items, replace=False))
    else:
        row_idx = np.arange(tfidf_matrix.shape[0])

    subset = tfidf_matrix[row_idx]

    if subset.shape[1] > max_features:
        doc_frequency = np.asarray((subset > 0).sum(axis=0)).ravel()
        top_feature_idx = np.argsort(doc_frequency)[-max_features:]
        top_feature_idx.sort()
        subset = subset[:, top_feature_idx]
    else:
        top_feature_idx = np.arange(subset.shape[1])

    metadata = {
        "row_count": int(subset.shape[0]),
        "feature_count": int(subset.shape[1]),
        "selected_items": int(len(row_idx)),
        "selected_features": int(len(top_feature_idx)),
        "density": round(subset.nnz / (subset.shape[0] * subset.shape[1]), 8),
        "estimated_dense_memory_gb": round(
            estimate_dense_memory_gb(subset.shape, dtype_bytes=4), 6
        ),
        "max_items": max_items,
        "max_features": max_features,
        "sampling_seed": SEED,
    }
    return subset.tocsr(), metadata


def get_valid_dimensions(subset: csr_matrix, dimensions: List[int]) -> List[int]:
    valid_dimensions = [dim for dim in dimensions if dim < min(subset.shape)]
    if not valid_dimensions:
        raise ValueError(
            f"Aucune dimension valide pour subset={subset.shape}; "
            "augmentez la taille du benchmark."
        )
    return valid_dimensions


def benchmark_method(subset: csr_matrix, method: str, dimensions: List[int]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for dim in dimensions:
        try:
            _, _, metrics = apply_dimension_reduction(
                subset,
                method=method,
                n_components=dim,
                verbose=False,
            )
            results.append(metrics)
        except Exception as exc:
            results.append(
                {
                    "method": method,
                    "n_components": dim,
                    "error": str(exc),
                }
            )
        gc.collect()
    return results


def build_summary_row(
    dim: int,
    method_results: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"n_components": dim}
    svd_result = next(
        (r for r in method_results["svd"] if r.get("n_components") == dim and "error" not in r),
        None,
    )

    for method in BENCHMARK_METHODS:
        result = next(
            (r for r in method_results[method] if r.get("n_components") == dim),
            None,
        )
        if not result:
            continue

        prefix = method
        if "error" in result:
            row[f"{prefix}_status"] = "error"
            row[f"{prefix}_error"] = result["error"]
            continue

        row[f"{prefix}_fit_time_s"] = result["fit_time_s"]
        row[f"{prefix}_transform_ms"] = result["transform_time_per_sample_ms"]

        if "variance_explained_pct" in result:
            row[f"{prefix}_variance_pct"] = result["variance_explained_pct"]
        if "reconstruction_error" in result:
            row[f"{prefix}_reconstruction_error"] = result["reconstruction_error"]
        if "dense_memory_gb" in result:
            row[f"{prefix}_dense_memory_gb"] = result["dense_memory_gb"]
        if svd_result and method != "svd":
            row[f"{prefix}_fit_vs_svd_ratio"] = round(
                result["fit_time_s"] / svd_result["fit_time_s"], 2
            )

    return row


def benchmark_variant(
    variant_dir: Path,
    dimensions: List[int] | None = None,
) -> Dict[str, Any]:
    dimensions = dimensions or BENCHMARK_DIMENSIONS
    tfidf_matrix, _ = load_tfidf_matrix(variant_dir, verbose=False)
    subset, subset_meta = build_benchmark_subset(tfidf_matrix)

    valid_dimensions = get_valid_dimensions(subset, dimensions)

    report: Dict[str, Any] = {
        "variant": variant_dir.name,
        "benchmark_subset": subset_meta,
        "dimensions_tested": valid_dimensions,
        "methods": {},
        "summary_by_dimension": [],
        "generated_at_unix_s": int(time.time()),
    }

    for method in BENCHMARK_METHODS:
        report["methods"][method] = benchmark_method(subset, method, valid_dimensions)

    for dim in valid_dimensions:
        report["summary_by_dimension"].append(build_summary_row(dim, report["methods"]))

    return report


def save_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def build_global_summary(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "variants": [report["variant"] for report in reports],
        "reports": [
            {
                "variant": report["variant"],
                "benchmark_subset": report["benchmark_subset"],
                "summary_by_dimension": report["summary_by_dimension"],
            }
            for report in reports
        ],
    }
    return summary


def main() -> None:
    t0 = time.perf_counter()

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARATIF - SVD VS NMF VS PCA")
    print("=" * 70)

    if not CLEAN_DATASETS_PATHS:
        print("\n[ERROR] Aucun dataset trouvé dans data/joining/")
        return

    reports = []
    for clean_path in CLEAN_DATASETS_PATHS:
        variant_name = clean_path.stem.replace("_clean_joined", "")
        variant_dir = clean_path.parent / variant_name
        sparse_path = variant_dir / "books_representation_sparse.npz"

        if not sparse_path.exists():
            print(f"\n[SKIP] {variant_name} - Matrice TF-IDF manquante")
            continue

        print(f"\n[Benchmark] {variant_name}", flush=True)
        report = benchmark_variant(variant_dir)
        out_path = variant_dir / BENCHMARK_OUTPUT
        save_report(report, out_path)
        reports.append(report)
        print(f"  Rapport: {out_path}", flush=True)

    if reports:
        save_report(build_global_summary(reports), GLOBAL_SUMMARY_OUTPUT)
        print(f"\nRésumé global: {GLOBAL_SUMMARY_OUTPUT}")

    print(f"\n{'=' * 70}")
    print(f"  Benchmark terminé en {time.perf_counter() - t0:.1f}s")
    print(f"  {len(reports)} variant(s) traité(s)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
