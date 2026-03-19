from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gc
import json
import time
import string
import re
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


# -- Configuration --------------------------------------------------

SEED = 42

TRAIN_PATHS = sorted(Path("data/joining").glob("*/train_interactions.parquet"))

TEXT_COLS = ["title", "description", "categories", "features"]
NUMERIC_COLS = ["average_rating", "rating_number", "price"]

DESC_MAX_LEN = 2000 # Troncature des descriptions pour limiter bruit et coût.

TFIDF_PARAMS = {
    "max_features": 40_000,   # Plafond du vocabulaire : garde les termes les plus fréquents globalement au-delà des filtres min_df/max_df.
    "max_df": 0.95, # Ignore les termes présents dans > 95 % des items (quasi stopwords “métiers”).
    "min_df": 2,    # Un mot doit apparaître dans au moins 2 documents pour entrer dans le vocab → réduit bruit et taille.
    "stop_words": "english",    # Retire une liste fixe de mots vides anglais (en plus du filtrage par fréquence).
    "lowercase": True,
    "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
    "dtype": np.float32, # Matrice plus légère qu'en float64.
}

SVD_N_COMPONENTS = 600   # projection linéaire dans un espace de 300 dimensions denses

ARTIFACTS = {
    "tfidf_matrix": "tfidf_matrix.npz",
    "svd_matrix": "tfidf_svd_matrix.npy",
    "item_ids": "tfidf_item_ids.npy",
    "vectorizer": "tfidf_vectorizer.pkl",
    "numeric_features": "numeric_features.npy",
    "scaler": "numeric_scaler.pkl",
    "svd_model": "svd_model.pkl",
    "report": "representation_report.json",
}


# -- Prétraitement -------------------------------------------------

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def preprocess_text(series: pd.Series, max_len: Optional[int] = None) -> pd.Series:
    """Prétraitement explicite : lowercase, ponctuation, espaces multiples, troncature."""
    s = series.fillna("").astype(str)
    if max_len is not None:
        s = s.str[:max_len]
    s = s.str.lower()
    s = s.str.replace(_PUNCT_RE, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def build_corpus(item_df: pd.DataFrame) -> pd.Series:
    """Concatène les colonnes texte prétraitées en 1 document par item."""
    parts = []
    for col in TEXT_COLS:
        if col not in item_df.columns:
            continue
        ml = DESC_MAX_LEN if col == "description" else None
        parts.append(preprocess_text(item_df[col], max_len=ml))
    return parts[0].str.cat(parts[1:], sep=" ") if parts else pd.Series([""] * len(item_df))


# -- Construction des représentations ------------------------------

def build_tfidf(corpus: pd.Series, verbose: bool = True) -> Tuple[csr_matrix, TfidfVectorizer]:
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    matrix = vectorizer.fit_transform(corpus)
    if verbose:
        print(f"[TF-IDF] {matrix.shape}, nnz={matrix.nnz:,}, "
              f"density={matrix.nnz / np.prod(matrix.shape):.4f}, "
              f"{time.perf_counter()-t0:.2f}s")
    return matrix, vectorizer


def build_svd(tfidf_matrix: csr_matrix, n_components: int = SVD_N_COMPONENTS,
              verbose: bool = True) -> Tuple[np.ndarray, TruncatedSVD]:
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    reduced = svd.fit_transform(tfidf_matrix)
    explained = svd.explained_variance_ratio_.sum()
    if verbose:
        print(f"[SVD] {reduced.shape}, variance expliquée={explained:.4f} "
              f"({explained*100:.1f}%), {time.perf_counter()-t0:.2f}s")
    return reduced.astype(np.float32), svd


def build_numeric_features(item_df: pd.DataFrame,
                           verbose: bool = True) -> Tuple[Optional[np.ndarray], Optional[StandardScaler]]:
    cols_present = [c for c in NUMERIC_COLS if c in item_df.columns]
    if not cols_present:
        return None, None
    t0 = time.perf_counter()
    raw = item_df[cols_present].copy()
    for c in cols_present:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.fillna(raw.median())
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw).astype(np.float32)
    if verbose:
        print(f"[Numeric] {scaled.shape}, cols={cols_present}, {time.perf_counter()-t0:.2f}s")
    return scaled, scaler


# -- Caractérisation -----------------------------------------------

def characterize_representation(
    tfidf_matrix: csr_matrix,
    svd_matrix: np.ndarray,
    vectorizer: TfidfVectorizer,
    svd_model: TruncatedSVD,
    numeric_features: Optional[np.ndarray],
    item_ids: np.ndarray,
    corpus: pd.Series,
) -> Dict[str, Any]:

    top_idf = sorted(zip(vectorizer.get_feature_names_out(),
                         vectorizer.idf_), key=lambda x: x[1])

    corpus_lengths = corpus.str.split().apply(len)

    report: Dict[str, Any] = {
        "preprocessing": {
            "text_cols": TEXT_COLS,
            "description_max_len": DESC_MAX_LEN,
            "steps": [
                "troncature description à {} chars".format(DESC_MAX_LEN),
                "lowercasing",
                "suppression ponctuation (regex)",
                "tokenisation alphabétique (tokens ≥2 chars)",
                "suppression stopwords anglais (scikit-learn, 318 termes)",
                "filtrage fréquence: min_df={}, max_df={}".format(TFIDF_PARAMS["min_df"], TFIDF_PARAMS["max_df"]),
            ],
        },
        "tfidf": {
            "n_items": tfidf_matrix.shape[0],
            "n_features": tfidf_matrix.shape[1],
            "nnz": int(tfidf_matrix.nnz),
            "density": round(tfidf_matrix.nnz / np.prod(tfidf_matrix.shape), 6),
            "corpus_tokens_avg": round(float(corpus_lengths.mean()), 1),
            "corpus_tokens_median": round(float(corpus_lengths.median()), 1),
            "top_10_lowest_idf": [(w, round(float(idf), 3)) for w, idf in top_idf[:10]],
            "top_10_highest_idf": [(w, round(float(idf), 3)) for w, idf in top_idf[-10:]],
            "sparsity_pct": round(100 * (1 - tfidf_matrix.nnz / np.prod(tfidf_matrix.shape)), 2),
        },
        "svd": {
            "n_components": svd_matrix.shape[1],
            "variance_explained": round(float(svd_model.explained_variance_ratio_.sum()), 4),
            "top_10_singular_values": [round(float(v), 4) for v in svd_model.singular_values_[:10]],
            "variance_explained_pct": round(100 * float(svd_model.explained_variance_ratio_.sum()), 2),
        },
        "numeric": {
            "cols": NUMERIC_COLS if numeric_features is not None else [],
            "n_features": numeric_features.shape[1] if numeric_features is not None else 0,
            "strategy": "StandardScaler + imputation médiane",
        },
        "context": {
            "purpose": "Représentation vectorielle des items (parent_asin) pour similarité item-item ou entrée modèle hybride.",
            "tfidf_note": "Matrice creuse : chaque item n'active qu'une petite fraction du vocabulaire (normal pour BoW + textes courts).",
            "svd_note": f"La variance expliquée mesure la reconstruction linéaire du sac de mots par {svd_matrix.shape[1]} axes, pas une qualité sémantique humaine.",
            "numeric_note": f"{len(NUMERIC_COLS) if numeric_features is not None else []} features z-score ; à pondérer ou concaténer avec SVD selon le modèle aval.",
        },
        "summary_one_liner": (
            f"{tfidf_matrix.shape[0]:,} items → TF-IDF {tfidf_matrix.shape[1]:,}D (densité {tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.4f}) "
            f"→ SVD {svd_matrix.shape[1]}D (var. {svd_model.explained_variance_ratio_.sum():.2%}) "
            f"+ {numeric_features.shape[1] if numeric_features is not None else 0} features numériques."
        ),
        "limits": [
            "TF-IDF: bag-of-words ignore l'ordre des mots et les relations sémantiques",
            "Descriptions longues diluent les termes discriminants malgré la troncature",
            f"Densité faible ({tfidf_matrix.nnz / np.prod(tfidf_matrix.shape):.4f}) — "
            "la plupart des features sont nulles pour un item donné",
            "SVD capture les corrélations linéaires mais pas les relations non-linéaires",
            "Attributs numériques peu nombreux (3) vs features textuelles (20k) — "
            "risque de dominance du signal textuel même après normalisation",
        ],
    }
    return report


# -- Persistance ---------------------------------------------------

def save_artifacts(
    out_dir: Path,
    tfidf_matrix: csr_matrix,
    svd_matrix: np.ndarray,
    vectorizer: TfidfVectorizer,
    svd_model: TruncatedSVD,
    item_ids: np.ndarray,
    numeric_features: Optional[np.ndarray],
    scaler: Optional[StandardScaler],
    report: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    save_npz(out_dir / ARTIFACTS["tfidf_matrix"], tfidf_matrix)
    paths["tfidf_matrix"] = str(out_dir / ARTIFACTS["tfidf_matrix"])

    np.save(out_dir / ARTIFACTS["svd_matrix"], svd_matrix)
    paths["svd_matrix"] = str(out_dir / ARTIFACTS["svd_matrix"])

    np.save(out_dir / ARTIFACTS["item_ids"], item_ids)
    paths["item_ids"] = str(out_dir / ARTIFACTS["item_ids"])

    with open(out_dir / ARTIFACTS["vectorizer"], "wb") as f:
        pickle.dump(vectorizer, f)
    paths["vectorizer"] = str(out_dir / ARTIFACTS["vectorizer"])

    with open(out_dir / ARTIFACTS["svd_model"], "wb") as f:
        pickle.dump(svd_model, f)
    paths["svd_model"] = str(out_dir / ARTIFACTS["svd_model"])

    if numeric_features is not None:
        np.save(out_dir / ARTIFACTS["numeric_features"], numeric_features)
        paths["numeric_features"] = str(out_dir / ARTIFACTS["numeric_features"])
    if scaler is not None:
        with open(out_dir / ARTIFACTS["scaler"], "wb") as f:
            pickle.dump(scaler, f)
        paths["scaler"] = str(out_dir / ARTIFACTS["scaler"])

    with open(out_dir / ARTIFACTS["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    paths["report"] = str(out_dir / ARTIFACTS["report"])

    if verbose:
        for name, p in paths.items():
            size = Path(p).stat().st_size / 1024 / 1024
            print(f"  saved {name}: {p} ({size:.1f} MiB)")

    return paths


def artifacts_exist(variant_dir: Path) -> bool:
    """Vérifie si les artéfacts existent déjà pour cette variante."""
    return all((variant_dir / v).exists() for v in ARTIFACTS.values())


# -- Pipeline principal --------------------------------------------

def build_representations(
    train_path: Path,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    variant_dir = train_path.parent
    variant = variant_dir.name

    if not force and artifacts_exist(variant_dir):
        if verbose:
            print(f"[{variant}] Artéfacts déjà présents — skip (force=False)")
        with open(variant_dir / ARTIFACTS["report"], encoding="utf-8") as f:
            return json.load(f)

    t0 = time.perf_counter()
    if verbose:
        print(f"\n{'='*70}\n  {variant} — Construction des représentations\n{'='*70}")

    # 1) Charger et dédupliquer au niveau item
    df = pd.read_parquet(train_path)
    item_df = df.drop_duplicates(subset=["parent_asin"]).reset_index(drop=True)
    item_ids = item_df["parent_asin"].values
    if verbose:
        print(f"[load] {len(df):,} interactions, {len(item_df):,} items uniques")
    del df
    gc.collect()

    # 2) Corpus texte prétraité
    corpus = build_corpus(item_df)

    # 3) TF-IDF
    tfidf_matrix, vectorizer = build_tfidf(corpus, verbose=verbose)

    # 4) SVD tronquée
    svd_matrix, svd_model = build_svd(tfidf_matrix, verbose=verbose)

    # 5) Attributs numériques
    numeric_features, scaler = build_numeric_features(item_df, verbose=verbose)

    # 6) Caractérisation
    report = characterize_representation(
        tfidf_matrix, svd_matrix, vectorizer, svd_model,
        numeric_features, item_ids, corpus,
    )
    report["variant"] = variant
    report["build_time_s"] = round(time.perf_counter() - t0, 2)

    # 7) Sauvegarde
    paths = save_artifacts(
        variant_dir, tfidf_matrix, svd_matrix, vectorizer, svd_model,
        item_ids, numeric_features, scaler, report, verbose=verbose,
    )
    report["artifact_paths"] = paths

    if verbose:
        print(f"\n[{variant}] done in {report['build_time_s']:.1f}s")
    if verbose and "summary_one_liner" in report:
        print(f"  → {report['summary_one_liner']}")
        
    del item_df, corpus, tfidf_matrix, svd_matrix, numeric_features
    gc.collect()

    return report


def main() -> None:
    for train_path in TRAIN_PATHS:
        report = build_representations(train_path, force=True, verbose=True)
        tfidf_info = report.get("tfidf", {})
        svd_info = report.get("svd", {})
        variant = report.get("variant", "?")
        

        print(f"\n{'---':^50}")
        print(f"  Résumé — {variant}")
        print(f"{'---':^50}")

        print(f"  TF-IDF: {tfidf_info.get('n_items')} items x {tfidf_info.get('n_features')} features, "
              f"density={tfidf_info.get('density')}")
        print(f"  SVD:    {svd_info.get('n_components')} composantes, "
              f"variance expliquée={svd_info.get('variance_explained')}"
              f" fraction de la variance du sac de mots (dans l'espace TF-IDF) capturée par ces {svd_info.get('n_components')} axes — ce n'est pas une “qualité sémantique” au sens humain, plutôt une compression qui préserve une partie de la structure co-occurrence des termes.")
        print(f"  Limites identifiées: {len(report.get('limits', []))}")

        print(f"  TF-IDF : {tfidf_info.get('n_items'):,} items × {tfidf_info.get('n_features'):,} features")
        print(f"           density={tfidf_info.get('density')}, sparsity≈{tfidf_info.get('sparsity_pct', '—')}%")
        print(f"  SVD    : {svd_info.get('n_components')} composantes, variance expliquée={svd_info.get('variance_explained')} ({svd_info.get('variance_explained_pct', '—')}%)")
        print(f"  Numériques : {report.get('numeric', {}).get('n_features', 0)} features (rating, nb_ratings, price)")
        print(f"  Build  : {report.get('build_time_s', '—')} s")
        print(f"  Limites : {len(report.get('limits', []))} points documentés dans le rapport")
        if "summary_one_liner" in report:
            print(f"\n  Contexte : {report['summary_one_liner']}")
        if "context" in report:
            print(f"  Note SVD : {report['context'].get('svd_note', '')}")
        print()

if __name__ == "__main__":
    main()