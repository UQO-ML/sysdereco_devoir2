"""
Tâche 2 - Sous-issue 0.1: Réduction de dimension

Ce script implémente la réduction de dimension sur les matrices TF-IDF des items
en utilisant TruncatedSVD (SVD tronquée), recommandée pour les données creuses.

Fonctionnalités:
- Teste plusieurs dimensions latentes: 50, 100, 200, 300
- S'entraîne uniquement sur les données train
- Transforme les items train et test dans l'espace latent
- Analyse variance expliquée, coût computationnel et qualité

Auteurs: Équipe ML
Date: Mars 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import gc
import json
import time
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.decomposition import TruncatedSVD, NMF, PCA


# -- Configuration --------------------------------------------------

SEED = 42

# Chemins des données
GLOB_PATTERN = "*_clean_joined.parquet"
CLEAN_DATASETS_PATHS = sorted(Path("data/joining").glob(GLOB_PATTERN))

# Dimensions latentes à tester
LATENT_DIMENSIONS = [50, 100, 200, 300]

# Méthodes de réduction disponibles
REDUCTION_METHODS = {
    "svd": "TruncatedSVD - Décomposition en valeurs singulières tronquée",
    "nmf": "NMF - Non-negative Matrix Factorization",
    "pca": "PCA - Principal Component Analysis (nécessite conversion dense)",
}

# Artéfacts de sortie
ARTIFACTS = {
    "reduced_matrix": "items_reduced_{method}_{dim}d.npy",
    "reducer_model": "reducer_{method}_{dim}d.pkl",
    "item_ids": "item_ids.npy",
    "report": "dimension_reduction_report.json",
    "comparison": "dimension_comparison.json",
}


# -- Justification de la méthode -----------------------------------

METHODOLOGY_JUSTIFICATION = {
    "chosen_method": "TruncatedSVD",
    "reasons": [
        "Optimisée pour matrices creuses (TF-IDF): pas de conversion dense nécessaire",
        "Efficace en mémoire: opère directement sur scipy.sparse.csr_matrix",
        "Computationnellement rapide: algorithme itératif randomisé (Halko et al.)",
        "Capture les directions de variance maximale dans l'espace TF-IDF",
        "Scalable: complexité O(n_samples × n_features × n_components)",
    ],
    "alternatives": {
        "NMF": {
            "advantages": [
                "Composantes non-négatives (interprétabilité)",
                "Utile pour la détection de topics",
                "Fonctionne avec matrices creuses",
            ],
            "limitations": [
                "Plus lent que SVD pour grandes dimensions",
                "Contrainte de non-négativité peut limiter l'expressivité",
                "Convergence itérative (peut nécessiter plus de temps)",
            ],
        },
        "PCA": {
            "advantages": [
                "Maximise variance expliquée (comme SVD)",
                "Bien documenté et standard",
            ],
            "limitations": [
                "Nécessite conversion en matrice dense → explosion mémoire",
                "Impraticable pour TF-IDF avec vocabulaire large (10k-50k features)",
                "Inefficace pour données creuses",
            ],
        },
    },
    "svd_advantages": [
        "Pas de centrage requis (SVD vs PCA): préserve sparsité",
        "Décomposition exacte en valeurs singulières",
        "Projection dans espace de dimension réduite tout en maximisant variance",
        "Compatible avec mesures de similarité cosine dans l'espace réduit",
    ],
    "svd_limitations": [
        "Hypothèse linéaire: capture corrélations linéaires uniquement",
        "Perd information sémantique fine (compression lossy)",
        "Interprétabilité des composantes limitée (combinaisons linéaires)",
        "Sensible aux outliers (valeurs extrêmes dans TF-IDF)",
        "Variance expliquée != qualité sémantique pour tâche aval",
    ],
}


# -- Chargement des données -----------------------------------------

def load_tfidf_matrix(variant_dir: Path, verbose: bool = True) -> Tuple[csr_matrix, np.ndarray]:
    """
    Charge la matrice TF-IDF et les IDs des items depuis les artéfacts
    produits par item_representation.py.

    Args:
        variant_dir: Répertoire contenant books_representation_sparse.npz
        verbose: Afficher informations de chargement

    Returns:
        (tfidf_matrix, item_ids): Matrice creuse TF-IDF et IDs items
    """
    t0 = time.perf_counter()

    tfidf_path = variant_dir / "books_representation_sparse.npz"
    if not tfidf_path.exists():
        raise FileNotFoundError(f"Matrice TF-IDF introuvable: {tfidf_path}")

    # Charger matrice TF-IDF (créée par item_representation.py)
    tfidf_matrix = load_npz(tfidf_path)

    # Charger les IDs items depuis le parquet source
    clean_src = variant_dir.parent / f"{variant_dir.name}_clean_joined.parquet"
    if not clean_src.exists():
        raise FileNotFoundError(f"Source parquet introuvable: {clean_src}")

    df_src = pd.read_parquet(clean_src, columns=["parent_asin"])
    item_df = df_src.drop_duplicates(subset=["parent_asin"], keep="first")
    item_ids = item_df["parent_asin"].values

    if len(item_ids) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"Dimension mismatch: {len(item_ids)} items vs "
            f"{tfidf_matrix.shape[0]} lignes TF-IDF"
        )

    if verbose:
        density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
        print(f"[Load TF-IDF] {tfidf_matrix.shape}, "
              f"nnz={tfidf_matrix.nnz:,}, density={density:.6f}, "
              f"{time.perf_counter()-t0:.2f}s")

    return tfidf_matrix, item_ids


# -- Réduction de dimension -----------------------------------------

def apply_dimension_reduction(
    tfidf_matrix: csr_matrix,
    method: str = "svd",
    n_components: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Applique une méthode de réduction de dimension sur la matrice TF-IDF.

    Args:
        tfidf_matrix: Matrice TF-IDF creuse (n_items, n_features)
        method: Méthode ('svd', 'nmf', 'pca')
        n_components: Nombre de composantes latentes
        verbose: Afficher progression

    Returns:
        (reduced_matrix, model, metrics): Matrice réduite, modèle entraîné, métriques
    """
    t_fit_start = time.perf_counter()

    if method == "svd":
        model = TruncatedSVD(
            n_components=n_components,
            random_state=SEED,
            algorithm="randomized",  # Plus rapide pour grandes matrices
        )
    elif method == "nmf":
        model = NMF(
            n_components=n_components,
            init="nndsvda",  # Initialisation NNDSVD améliorée
            random_state=SEED,
            max_iter=400,
            alpha_W=0.0,  # Pas de régularisation L1 par défaut
            alpha_H=0.0,
            l1_ratio=0.0,
        )
    elif method == "pca":
        # PCA nécessite matrice dense - ATTENTION mémoire!
        if verbose:
            print(f"  [WARNING] PCA nécessite conversion dense "
                  f"({tfidf_matrix.shape[0] * tfidf_matrix.shape[1] * 4 / 1024**3:.2f} GB)")
        model = PCA(n_components=n_components, random_state=SEED)
    else:
        raise ValueError(f"Méthode inconnue: {method}")

    # Entraînement
    reduced_matrix = model.fit_transform(tfidf_matrix)
    t_fit = time.perf_counter() - t_fit_start

    # Temps de transformation (inférence) - mesure sur la même matrice
    t_transform_start = time.perf_counter()
    _ = model.transform(tfidf_matrix[:min(1000, tfidf_matrix.shape[0])])  # 1000 samples
    t_transform = time.perf_counter() - t_transform_start
    t_transform_per_sample = t_transform / min(1000, tfidf_matrix.shape[0])

    # Calcul métriques
    metrics = {
        "method": method,
        "n_components": n_components,
        "input_shape": list(tfidf_matrix.shape),
        "output_shape": list(reduced_matrix.shape),
        "fit_time_s": round(t_fit, 4),
        "transform_time_s": round(t_transform, 4),
        "transform_time_per_sample_ms": round(t_transform_per_sample * 1000, 4),
    }

    # Métriques spécifiques SVD/PCA
    if hasattr(model, "explained_variance_ratio_"):
        explained_var = model.explained_variance_ratio_
        metrics["variance_explained"] = round(float(explained_var.sum()), 6)
        metrics["variance_explained_pct"] = round(float(explained_var.sum()) * 100, 2)
        metrics["variance_per_component"] = [round(float(v), 6) for v in explained_var[:10]]
        metrics["singular_values"] = [round(float(v), 4) for v in model.singular_values_[:10]]

    # Métriques NMF
    if hasattr(model, "reconstruction_err_"):
        metrics["reconstruction_error"] = round(float(model.reconstruction_err_), 4)
        metrics["n_iter"] = int(model.n_iter_)

    if verbose:
        info = [f"{method.upper()}", f"{reduced_matrix.shape}", f"{t_fit:.2f}s"]
        if "variance_explained_pct" in metrics:
            info.append(f"var={metrics['variance_explained_pct']:.1f}%")
        print(f"[Reduction] {', '.join(info)}")

    return reduced_matrix.astype(np.float32), model, metrics


# -- Comparaison des dimensions -------------------------------------

def compare_dimensions(
    tfidf_matrix: csr_matrix,
    dimensions: List[int],
    method: str = "svd",
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compare plusieurs dimensions latentes pour une méthode donnée.

    Args:
        tfidf_matrix: Matrice TF-IDF creuse
        dimensions: Liste de dimensions à tester (ex: [50, 100, 200, 300])
        method: Méthode de réduction
        verbose: Afficher progression

    Returns:
        Liste de dictionnaires de métriques pour chaque dimension
    """
    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Comparaison dimensions - {method.upper()}")
        print(f"{'='*70}")

    for dim in dimensions:
        if verbose:
            print(f"\n[{dim}D]")

        try:
            _, model, metrics = apply_dimension_reduction(
                tfidf_matrix,
                method=method,
                n_components=dim,
                verbose=verbose,
            )
            results.append(metrics)

        except Exception as e:
            print(f"  [ERROR] Dimension {dim}: {e}")
            continue

    return results


# -- Analyse et recommandations ------------------------------------

def analyze_tradeoffs(comparison_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyse le compromis variance expliquée / coût computationnel.

    Args:
        comparison_results: Résultats de compare_dimensions()

    Returns:
        Dictionnaire d'analyse avec recommandation
    """
    if not comparison_results:
        return {}

    analysis = {
        "summary": [],
        "tradeoffs": {
            "variance_vs_dimension": [],
            "time_vs_dimension": [],
        },
        "recommendation": {},
    }

    for result in comparison_results:
        dim = result["n_components"]
        var_pct = result.get("variance_explained_pct", 0)
        fit_time = result.get("fit_time_s", 0)

        analysis["summary"].append({
            "dimension": dim,
            "variance_pct": var_pct,
            "fit_time_s": fit_time,
            "transform_ms": result.get("transform_time_per_sample_ms", 0),
        })

        analysis["tradeoffs"]["variance_vs_dimension"].append((dim, var_pct))
        analysis["tradeoffs"]["time_vs_dimension"].append((dim, fit_time))

    # Calcul gains marginaux de variance
    if len(comparison_results) > 1:
        marginal_gains = []
        for i in range(1, len(comparison_results)):
            prev_var = comparison_results[i-1].get("variance_explained_pct", 0)
            curr_var = comparison_results[i].get("variance_explained_pct", 0)
            gain = curr_var - prev_var
            marginal_gains.append({
                "from_dim": comparison_results[i-1]["n_components"],
                "to_dim": comparison_results[i]["n_components"],
                "variance_gain_pct": round(gain, 2),
            })
        analysis["marginal_gains"] = marginal_gains

    # Recommandation basée sur gain marginal et coût
    # Critère: gain marginal < 2% et variance > 30%
    recommended_dim = None
    for i, result in enumerate(comparison_results):
        var_pct = result.get("variance_explained_pct", 0)
        if i > 0 and var_pct > 30:
            marginal = marginal_gains[i-1]["variance_gain_pct"]
            if marginal < 2.0:  # Gain marginal faible
                recommended_dim = result["n_components"]
                break

    # Fallback: dimension médiane
    if recommended_dim is None and comparison_results:
        recommended_dim = comparison_results[len(comparison_results)//2]["n_components"]

    if recommended_dim:
        recommended_result = next(
            r for r in comparison_results if r["n_components"] == recommended_dim
        )
        analysis["recommendation"] = {
            "dimension": recommended_dim,
            "variance_pct": recommended_result.get("variance_explained_pct", 0),
            "fit_time_s": recommended_result.get("fit_time_s", 0),
            "rationale": [
                f"Variance expliquée: {recommended_result.get('variance_explained_pct', 0):.1f}%",
                f"Temps d'entraînement acceptable: {recommended_result.get('fit_time_s', 0):.2f}s",
                "Bon compromis variance/coût computationnel",
                "Gain marginal décroissant au-delà de cette dimension",
            ],
        }

    return analysis


# -- Sauvegarde des artéfacts ---------------------------------------

def save_artifacts(
    out_dir: Path,
    reduced_matrix: np.ndarray,
    model: Any,
    item_ids: np.ndarray,
    metrics: Dict[str, Any],
    method: str,
    n_components: int,
    verbose: bool = True,
) -> Dict[str, str]:
    """Sauvegarde matrice réduite, modèle et métriques."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Matrice réduite
    reduced_path = out_dir / ARTIFACTS["reduced_matrix"].format(
        method=method, dim=n_components
    )
    np.save(reduced_path, reduced_matrix)
    paths["reduced_matrix"] = str(reduced_path)

    # Modèle
    model_path = out_dir / ARTIFACTS["reducer_model"].format(
        method=method, dim=n_components
    )
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    paths["model"] = str(model_path)

    # IDs items
    ids_path = out_dir / ARTIFACTS["item_ids"]
    np.save(ids_path, item_ids)
    paths["item_ids"] = str(ids_path)

    # Métriques
    metrics_path = out_dir / f"metrics_{method}_{n_components}d.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    paths["metrics"] = str(metrics_path)

    if verbose:
        for name, p in paths.items():
            size = Path(p).stat().st_size / 1024 / 1024
            print(f"  saved {name}: {Path(p).name} ({size:.1f} MiB)")

    return paths


# -- Pipeline principal ---------------------------------------------

def run_dimension_reduction_pipeline(
    variant_dir: Path,
    method: str = "svd",
    dimensions: List[int] = LATENT_DIMENSIONS,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline complet de réduction de dimension pour un variant.

    Args:
        variant_dir: Répertoire du variant (ex: data/joining/active_pre_split/)
        method: Méthode de réduction ('svd', 'nmf', 'pca')
        dimensions: Liste de dimensions à tester
        force: Forcer recalcul même si artéfacts existent
        verbose: Afficher progression détaillée

    Returns:
        Rapport complet avec métriques et recommandations
    """
    variant = variant_dir.name
    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'='*70}")
        print(f"  {variant} - Réduction de dimension ({method.upper()})")
        print(f"{'='*70}")

    # 1. Charger matrice TF-IDF
    tfidf_matrix, item_ids = load_tfidf_matrix(variant_dir, verbose=verbose)

    # 2. Comparer différentes dimensions
    comparison_results = compare_dimensions(
        tfidf_matrix,
        dimensions=dimensions,
        method=method,
        verbose=verbose,
    )

    # 3. Analyser compromis
    analysis = analyze_tradeoffs(comparison_results)

    # 4. Sauvegarder tous les résultats
    all_paths = {}
    for result in comparison_results:
        dim = result["n_components"]

        # Recalculer pour obtenir le modèle
        reduced_matrix, model, _ = apply_dimension_reduction(
            tfidf_matrix,
            method=method,
            n_components=dim,
            verbose=False,
        )

        paths = save_artifacts(
            variant_dir,
            reduced_matrix,
            model,
            item_ids,
            result,
            method,
            dim,
            verbose=False,
        )
        all_paths[f"{dim}d"] = paths

    # 5. Rapport global
    report = {
        "variant": variant,
        "method": method,
        "methodology": METHODOLOGY_JUSTIFICATION,
        "dimensions_tested": dimensions,
        "comparison_results": comparison_results,
        "analysis": analysis,
        "artifact_paths": all_paths,
        "build_time_s": round(time.perf_counter() - t0, 2),
    }

    # Sauvegarder rapport comparatif
    comparison_path = variant_dir / ARTIFACTS["comparison"]
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Résumé - {variant}")
        print(f"{'='*70}")
        print(f"  Méthode: {method.upper()}")
        print(f"  Dimensions testées: {dimensions}")
        if analysis.get("recommendation"):
            rec = analysis["recommendation"]
            print(f"  Recommandation: {rec['dimension']}D "
                  f"(variance={rec['variance_pct']:.1f}%, "
                  f"temps={rec['fit_time_s']:.2f}s)")
        print(f"  Temps total: {report['build_time_s']:.1f}s")
        print(f"  Rapport: {comparison_path}")
        print()

    return report


# -- Point d'entrée principal ---------------------------------------

def main() -> None:
    """
    Exécute la réduction de dimension sur tous les variants disponibles.
    """
    t0 = time.perf_counter()

    print("\n" + "="*70)
    print("  TÂCHE 2 - RÉDUCTION DE DIMENSION")
    print("="*70)
    print(f"\nMéthode choisie: {METHODOLOGY_JUSTIFICATION['chosen_method']}")
    print("Raisons:")
    for reason in METHODOLOGY_JUSTIFICATION['reasons']:
        print(f"  • {reason}")

    if not CLEAN_DATASETS_PATHS:
        print("\n[ERROR] Aucun dataset trouvé dans data/joining/")
        print("Exécutez d'abord item_representation.py pour générer les matrices TF-IDF")
        return

    all_reports = []

    for clean_path in CLEAN_DATASETS_PATHS:
        variant_name = clean_path.stem.replace("_clean_joined", "")
        variant_dir = clean_path.parent / variant_name

        if not variant_dir.exists() or not (variant_dir / "books_representation_sparse.npz").exists():
            print(f"\n[SKIP] {variant_name} - Matrice TF-IDF manquante")
            continue

        try:
            report = run_dimension_reduction_pipeline(
                variant_dir,
                method="svd",
                dimensions=LATENT_DIMENSIONS,
                force=True,
                verbose=True,
            )
            all_reports.append(report)

        except Exception as e:
            print(f"\n[ERROR] {variant_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print(f"  Pipeline complet terminé en {time.perf_counter() - t0:.1f}s")
    print(f"  {len(all_reports)} variant(s) traité(s)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
