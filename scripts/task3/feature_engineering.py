"""
scripts/task3/feature_engineering.py

Tâche 3 — Construction des features pour l'apprentissage d'un score de pertinence.

Pour chaque paire (user, item) du train, construit un vecteur de features exploitables
par un modèle de classification / régression / LTR.

Artefacts produits (par variante) :
  - X_train.npy           : matrice de features (n_pairs, n_features)
  - y_train.npy           : labels (rating ou binaire)
  - groups_train.npy      : nb d'items par user (pour LTR grouping)
  - feature_names.json    : noms ordonnés des features
  - feature_report.json   : rapport détaillé
  - feature_report.md     : rapport Markdown lisible
"""
from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, issparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, normalize


# ======================================================================
#  CONFIGURATION
# ======================================================================

VARIANT_DIRS = sorted(Path("data/joining").glob("*_pre_split"))

# --- Colonnes requises dans train_interactions.parquet ----------------

REQUIRED_INTERACTION_COLS = [
    "user_id", "parent_asin", "rating", "timestamp",
]

REQUIRED_META_COLS = [
    "average_rating", "rating_number", "price",
    "categories", "author_name",
]

OPTIONAL_META_COLS = [
    "helpful_vote", "verified_purchase",
    "nb_pages", "pub_year", "book_format", "reading_age_min",
]

# --- Seuil de binarisation du label ----------------------------------
#   rating >= RELEVANCE_THRESHOLD  →  1 (pertinent)
#   rating <  RELEVANCE_THRESHOLD  →  0 (non pertinent)

RELEVANCE_THRESHOLD = 4.0

# --- Sampling négatif -------------------------------------------------
#   Pour chaque user, on échantillonne N_NEG_PER_POS items NON vus
#   par item positif (vu et noté) pour créer les paires négatives.

N_NEG_PER_POS = 3

# --- Cosine similarity batch size ------------------------------------

COSINE_BATCH_SIZE = 8192

# --- Artefacts en entrée (produits par les étapes précédentes) --------

ARTIFACTS_IN = {
    "item_matrix":   "books_representation_sparse.npz",
    "user_profiles": "user_profiles_tfidf.npz",
    "item_ids":      "item_ids.npy",
    "user_ids":      "user_ids.npy",
}

# --- Artefacts en sortie ----------------------------------------------

ARTIFACTS_OUT = {
    "X_train":        "X_train_features.npy",
    "y_train":        "y_train_labels.npy",
    "y_rating":       "y_train_ratings.npy",
    "groups":         "groups_train.npy",
    "feature_names":  "feature_names.json",
    "report_json":    "feature_report.json",
    "report_md":      "feature_report.md",
}

RANDOM_SEED = 42


# ======================================================================
#  UTILITAIRES
# ======================================================================

def _fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"


def _check_columns(df: pd.DataFrame, required: list[str], label: str) -> list[str]:
    """Vérifie la présence des colonnes requises, retourne les colonnes manquantes."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [WARN] {label}: colonnes manquantes = {missing}")
    return missing


def _parse_genre(cat_str: str) -> str:
    """Extrait le genre (niveau 2) depuis 'Books | Genre | Sub-genre'."""
    if not isinstance(cat_str, str) or not cat_str.strip():
        return "Unknown"
    parts = [p.strip() for p in cat_str.split("|")]
    return parts[1] if len(parts) > 1 else parts[0]


# ======================================================================
#  CHARGEMENT DES ARTEFACTS AMONT
# ======================================================================

def load_upstream_artifacts(variant_dir: Path, verbose: bool = True) -> Dict[str, Any]:
    """Charge item_matrix, user_profiles, item_ids, user_ids."""
    t0 = time.perf_counter()
    arts: Dict[str, Any] = {}

    item_matrix_path = variant_dir / ARTIFACTS_IN["item_matrix"]
    if not item_matrix_path.exists():
        raise FileNotFoundError(f"Matrice items manquante: {item_matrix_path}")
    arts["item_matrix"] = load_npz(item_matrix_path)

    profiles_path = variant_dir / ARTIFACTS_IN["user_profiles"]
    if not profiles_path.exists():
        profiles_path_npy = variant_dir / "user_profiles_tfidf.npy"
        if profiles_path_npy.exists():
            arts["user_profiles"] = np.load(profiles_path_npy)
        else:
            raise FileNotFoundError(f"Profils manquants: {profiles_path}")
    else:
        arts["user_profiles"] = load_npz(profiles_path)

    arts["item_ids"] = np.load(variant_dir / ARTIFACTS_IN["item_ids"], allow_pickle=True)
    arts["user_ids"] = np.load(variant_dir / ARTIFACTS_IN["user_ids"], allow_pickle=True)

    arts["item_to_idx"] = {asin: i for i, asin in enumerate(arts["item_ids"])}
    arts["user_to_idx"] = {uid: i for i, uid in enumerate(arts["user_ids"])}

    if verbose:
        print(f"  [load] item_matrix={arts['item_matrix'].shape}, "
              f"user_profiles={arts['user_profiles'].shape}, "
              f"items={len(arts['item_ids']):,}, users={len(arts['user_ids']):,}, "
              f"{time.perf_counter()-t0:.2f}s")

    return arts


# ======================================================================
#  CONSTRUCTION DES STATISTIQUES AGRÉGÉES
# ======================================================================

def build_user_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par utilisateur calculées sur le train."""
    stats = train_df.groupby("user_id", as_index=False).agg(
        user_nb_interactions=("rating", "count"),
        user_mean_rating=("rating", "mean"),
        user_std_rating=("rating", "std"),
    )
    stats["user_std_rating"] = stats["user_std_rating"].fillna(0.0)

    if "helpful_vote" in train_df.columns:
        hv = train_df.groupby("user_id", as_index=False)["helpful_vote"].mean()
        hv.columns = ["user_id", "user_mean_helpful"]
        stats = stats.merge(hv, on="user_id", how="left")

    if "verified_purchase" in train_df.columns:
        vp = train_df.groupby("user_id", as_index=False)["verified_purchase"].mean()
        vp.columns = ["user_id", "user_verified_ratio"]
        stats = stats.merge(vp, on="user_id", how="left")

    if "categories" in train_df.columns:
        genre_div = (
            train_df.assign(genre=train_df["categories"].apply(_parse_genre))
            .groupby("user_id")["genre"]
            .nunique()
            .rename("user_genre_diversity")
            .reset_index()
        )
        stats = stats.merge(genre_div, on="user_id", how="left")

    if "nb_pages" in train_df.columns:
        pages = train_df.groupby("user_id", as_index=False)["nb_pages"].mean()
        pages.columns = ["user_id", "user_mean_pages"]
        stats = stats.merge(pages, on="user_id", how="left")

    if "pub_year" in train_df.columns:
        years = train_df.groupby("user_id", as_index=False)["pub_year"].median()
        years.columns = ["user_id", "user_median_pub_year"]
        stats = stats.merge(years, on="user_id", how="left")

    return stats


def build_item_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par item calculées sur le train."""
    stats = train_df.groupby("parent_asin", as_index=False).agg(
        item_nb_interactions_train=("rating", "count"),
        item_mean_rating_train=("rating", "mean"),
    )

    if "helpful_vote" in train_df.columns:
        hv = train_df.groupby("parent_asin", as_index=False)["helpful_vote"].mean()
        hv.columns = ["parent_asin", "item_mean_helpful"]
        stats = stats.merge(hv, on="parent_asin", how="left")

    return stats


# ======================================================================
#  FEATURES PAR AUTEUR / GENRE (HISTORIQUE)
# ======================================================================

def build_user_author_history(train_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Pour chaque user, retourne {author: {count, mean_rating}}."""
    if "author_name" not in train_df.columns:
        return {}
    sub = train_df[train_df["author_name"].astype(str).str.strip() != ""]
    grouped = sub.groupby(["user_id", "author_name"]).agg(
        count=("rating", "count"),
        mean_rating=("rating", "mean"),
    ).reset_index()

    history: Dict[str, Dict[str, Any]] = {}
    for uid, grp in grouped.groupby("user_id"):
        history[uid] = {
            row["author_name"]: {"count": int(row["count"]), "mean_rating": float(row["mean_rating"])}
            for _, row in grp.iterrows()
        }
    return history


def build_user_genre_history(train_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Pour chaque user, retourne {genre: count}."""
    if "categories" not in train_df.columns:
        return {}
    sub = train_df.assign(genre=train_df["categories"].apply(_parse_genre))
    grouped = sub.groupby(["user_id", "genre"]).size().reset_index(name="count")

    history: Dict[str, Dict[str, int]] = {}
    for uid, grp in grouped.groupby("user_id"):
        history[uid] = dict(zip(grp["genre"], grp["count"]))
    return history


# ======================================================================
#  SAMPLING DES PAIRES (USER, ITEM)
# ======================================================================

def sample_training_pairs(
    train_df: pd.DataFrame,
    item_ids: np.ndarray,
    n_neg_per_pos: int = N_NEG_PER_POS,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Construit les paires d'entraînement :
      - Positives : toutes les interactions du train (user, item, rating)
      - Négatives : items NON vus échantillonnés aléatoirement (rating = 0)

    Le DataFrame est trié par user_id pour le grouping LTR.
    """
    rng = np.random.default_rng(seed)
    all_items = set(item_ids)

    seen_by_user = train_df.groupby("user_id")["parent_asin"].apply(set).to_dict()

    pos_pairs = train_df[["user_id", "parent_asin", "rating"]].copy()
    pos_pairs["is_positive"] = True

    neg_rows: list[dict] = []
    for uid, seen_items in seen_by_user.items():
        unseen = np.array(list(all_items - seen_items))
        n_neg = min(len(seen_items) * n_neg_per_pos, len(unseen))
        if n_neg == 0:
            continue
        sampled = rng.choice(unseen, size=n_neg, replace=False)
        for asin in sampled:
            neg_rows.append({"user_id": uid, "parent_asin": asin, "rating": 0.0, "is_positive": False})

    neg_df = pd.DataFrame(neg_rows)
    pairs = pd.concat([pos_pairs, neg_df], ignore_index=True)
    pairs = pairs.sort_values(["user_id", "is_positive"], ascending=[True, False]).reset_index(drop=True)

    if verbose:
        n_pos = pos_pairs.shape[0]
        n_neg = neg_df.shape[0]
        print(f"  [pairs] positives={n_pos:,}, négatives={n_neg:,}, "
              f"ratio neg/pos={n_neg/n_pos:.2f}, total={len(pairs):,}")

    del neg_rows, neg_df, pos_pairs
    gc.collect()

    return pairs


# ======================================================================
#  CONSTRUCTION DU VECTEUR DE FEATURES
# ======================================================================

def build_feature_matrix(
    pairs: pd.DataFrame,
    train_df: pd.DataFrame,
    artifacts: Dict[str, Any],
    user_stats: pd.DataFrame,
    item_stats: pd.DataFrame,
    user_author_hist: Dict[str, Dict[str, Any]],
    user_genre_hist: Dict[str, Dict[str, int]],
    verbose: bool = True,
) -> Tuple[np.ndarray, list[str]]:
    """
    Construit la matrice X de features pour toutes les paires.

    Retourne (X, feature_names).
    """
    t0 = time.perf_counter()
    n = len(pairs)

    item_to_idx = artifacts["item_to_idx"]
    user_to_idx = artifacts["user_to_idx"]
    user_profiles = artifacts["user_profiles"]
    item_matrix = artifacts["item_matrix"]

    # --- Pré-indexation des métadonnées item (une seule fois) ----------

    item_meta_cols = ["parent_asin", "average_rating", "rating_number", "price",
                      "categories", "author_name"]
    optional_item = ["nb_pages", "pub_year", "book_format", "reading_age_min"]
    item_meta_cols += [c for c in optional_item if c in train_df.columns]

    item_meta = (
        train_df[item_meta_cols]
        .drop_duplicates(subset="parent_asin", keep="first")
        .set_index("parent_asin")
    )

    # --- Merge des stats user/item sur les paires ---------------------

    pairs = pairs.merge(user_stats, on="user_id", how="left")
    pairs = pairs.merge(item_stats, on="parent_asin", how="left")

    # Pour les items négatifs (non vus) : merge metadata
    pairs = pairs.merge(
        item_meta[["average_rating", "rating_number", "price"]].reset_index(),
        on="parent_asin", how="left", suffixes=("", "_meta"),
    )

    # Utiliser les colonnes metadata si les colonnes item_stats sont NaN
    for c in ["average_rating", "rating_number", "price"]:
        meta_c = f"{c}_meta"
        if meta_c in pairs.columns:
            pairs[c] = pairs[c].fillna(pairs[meta_c])
            pairs.drop(columns=[meta_c], inplace=True)

    # --- Features fixes (colonnes directes) ----------------------------

    features: Dict[str, np.ndarray] = {}

    features["average_rating"] = pairs["average_rating"].fillna(0).values.astype(np.float32)
    features["log_rating_number"] = np.log1p(pairs["rating_number"].fillna(0).values).astype(np.float32)
    features["price"] = pairs["price"].fillna(0).values.astype(np.float32)
    features["item_nb_interactions_train"] = pairs["item_nb_interactions_train"].fillna(0).values.astype(np.float32)
    features["item_mean_rating_train"] = pairs["item_mean_rating_train"].fillna(0).values.astype(np.float32)

    features["user_nb_interactions"] = pairs["user_nb_interactions"].fillna(0).values.astype(np.float32)
    features["user_mean_rating"] = pairs["user_mean_rating"].fillna(0).values.astype(np.float32)
    features["user_std_rating"] = pairs["user_std_rating"].fillna(0).values.astype(np.float32)

    # --- Features optionnelles (dépendent des colonnes disponibles) ----

    if "user_mean_helpful" in pairs.columns:
        features["user_mean_helpful"] = pairs["user_mean_helpful"].fillna(0).values.astype(np.float32)

    if "user_verified_ratio" in pairs.columns:
        features["user_verified_ratio"] = pairs["user_verified_ratio"].fillna(0).values.astype(np.float32)

    if "user_genre_diversity" in pairs.columns:
        features["user_genre_diversity"] = pairs["user_genre_diversity"].fillna(0).values.astype(np.float32)

    if "item_mean_helpful" in pairs.columns:
        features["item_mean_helpful"] = pairs["item_mean_helpful"].fillna(0).values.astype(np.float32)

    # nb_pages et dérivées
    if "nb_pages" in item_meta.columns:
        item_pages = item_meta["nb_pages"].to_dict()
        pages_arr = pairs["parent_asin"].map(item_pages).fillna(0).values.astype(np.float32)
        features["nb_pages"] = pages_arr

        if "user_mean_pages" in pairs.columns:
            user_mean_p = pairs["user_mean_pages"].fillna(pages_arr.mean()).values.astype(np.float32)
            features["page_deviation"] = (pages_arr - user_mean_p).astype(np.float32)

    # pub_year et dérivées
    if "pub_year" in item_meta.columns:
        item_years = item_meta["pub_year"].to_dict()
        year_arr = pairs["parent_asin"].map(item_years).fillna(0).values.astype(np.float32)
        features["pub_year"] = year_arr
        features["book_age"] = (2023.0 - year_arr).clip(min=0).astype(np.float32)

        if "user_median_pub_year" in pairs.columns:
            user_med_y = pairs["user_median_pub_year"].fillna(year_arr.mean()).values.astype(np.float32)
            features["era_match"] = np.abs(year_arr - user_med_y).astype(np.float32)

    # book_format (one-hot)
    if "book_format" in item_meta.columns:
        fmt_map = item_meta["book_format"].to_dict()
        fmt_series = pairs["parent_asin"].map(fmt_map).fillna("unknown")
        for fmt_val in ["ebook", "hardcover", "paperback", "mass_market"]:
            features[f"fmt_{fmt_val}"] = (fmt_series == fmt_val).astype(np.float32).values

    # --- Cosine similarity (user profile vs item vector) ---------------

    if verbose:
        print("  [features] cosine similarity par batch...")

    cos_scores = np.zeros(n, dtype=np.float32)

    user_idx_arr = pairs["user_id"].map(user_to_idx).values
    item_idx_arr = pairs["parent_asin"].map(item_to_idx).values

    valid_mask = pd.notna(user_idx_arr) & pd.notna(item_idx_arr)
    valid_user_idx = user_idx_arr[valid_mask].astype(np.int64, copy=False)
    valid_item_idx = item_idx_arr[valid_mask].astype(np.int64, copy=False)
    valid_pos = np.where(valid_mask)[0]

    # Important: copy=False pour eviter une copie geante
    user_profiles_normed = normalize(user_profiles, norm="l2", copy=False)
    item_matrix_normed = normalize(item_matrix, norm="l2", copy=False)

    for start in range(0, len(valid_user_idx), COSINE_BATCH_SIZE):
        end = min(start + COSINE_BATCH_SIZE, len(valid_user_idx))

        u_batch = valid_user_idx[start:end]
        i_batch = valid_item_idx[start:end]

        u_vecs = user_profiles_normed[u_batch]   # sparse
        i_vecs = item_matrix_normed[i_batch]     # sparse

        # Dot product ligne a ligne sur sparse normalise
        batch_cos = np.asarray(u_vecs.multiply(i_vecs).sum(axis=1)).ravel().astype(np.float32, copy=False)
        cos_scores[valid_pos[start:end]] = batch_cos

    features["cosine_similarity"] = cos_scores

    # Cosine déviation (écart par rapport à la moyenne du user)
    user_mean_cos = pd.Series(cos_scores).groupby(user_idx_arr).transform("mean").values
    features["cosine_ecart"] = (cos_scores - user_mean_cos).astype(np.float32)

    # --- Features auteur (historique user) -----------------------------

    if user_author_hist:
        author_map = item_meta["author_name"].to_dict() if "author_name" in item_meta.columns else {}

        already_read = np.zeros(n, dtype=np.float32)
        nb_books_same_author = np.zeros(n, dtype=np.float32)
        user_avg_rating_author = np.zeros(n, dtype=np.float32)

        for i, (uid, asin) in enumerate(zip(pairs["user_id"], pairs["parent_asin"])):
            author = author_map.get(asin, "")
            if not author or author == "":
                continue
            hist = user_author_hist.get(uid, {})
            if author in hist:
                already_read[i] = 1.0
                nb_books_same_author[i] = hist[author]["count"]
                user_avg_rating_author[i] = hist[author]["mean_rating"]

        features["already_read_author"] = already_read
        features["nb_books_same_author"] = nb_books_same_author
        features["user_avg_rating_author"] = user_avg_rating_author

    # --- Features genre (historique user) ------------------------------

    if user_genre_hist and "categories" in item_meta.columns:
        genre_map = item_meta["categories"].apply(_parse_genre).to_dict()

        genre_match = np.zeros(n, dtype=np.float32)
        nb_genres_common = np.zeros(n, dtype=np.float32)

        for i, (uid, asin) in enumerate(zip(pairs["user_id"], pairs["parent_asin"])):
            item_genre = genre_map.get(asin, "Unknown")
            hist = user_genre_hist.get(uid, {})
            if item_genre in hist:
                genre_match[i] = 1.0
            nb_genres_common[i] = len(hist)

        features["genre_match"] = genre_match
        features["user_nb_genres_explored"] = nb_genres_common

    # --- Assemblage final ---------------------------------------------

    feature_names = list(features.keys())
    X = np.column_stack([features[f] for f in feature_names]).astype(np.float32)

    if verbose:
        print(f"  [features] X.shape={X.shape}, "
              f"{len(feature_names)} features, "
              f"{time.perf_counter()-t0:.1f}s")

    return X, feature_names


# ======================================================================
#  LABELS
# ======================================================================

def build_labels(pairs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (y_binary, y_rating)."""
    y_rating = pairs["rating"].values.astype(np.float32)
    y_binary = (y_rating >= RELEVANCE_THRESHOLD).astype(np.int32)
    return y_binary, y_rating


def build_groups(pairs: pd.DataFrame) -> np.ndarray:
    """Retourne le tableau de tailles de groupes par user (pour LGBMRanker)."""
    return pairs.groupby("user_id", sort=False).size().values.astype(np.int32)


# ======================================================================
#  RAPPORT
# ======================================================================

def build_report(
    variant: str,
    pairs: pd.DataFrame,
    X: np.ndarray,
    y_binary: np.ndarray,
    y_rating: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    elapsed: float,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "variant": variant,
        "n_pairs": int(len(pairs)),
        "n_positive": int((pairs["is_positive"]).sum()),
        "n_negative": int((~pairs["is_positive"]).sum()),
        "n_users": int(pairs["user_id"].nunique()),
        "n_items": int(pairs["parent_asin"].nunique()),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "label_distribution": {
            "binary_1_relevant": int(y_binary.sum()),
            "binary_0_irrelevant": int(len(y_binary) - y_binary.sum()),
            "relevance_threshold": RELEVANCE_THRESHOLD,
        },
        "rating_distribution": {
            f"rating_{r:.0f}": int((y_rating == r).sum())
            for r in sorted(np.unique(y_rating))
        },
        "groups_stats": {
            "n_groups": int(len(groups)),
            "mean_group_size": float(np.mean(groups)),
            "min_group_size": int(np.min(groups)),
            "max_group_size": int(np.max(groups)),
        },
        "feature_stats": {},
        "X_shape": list(X.shape),
        "X_dtype": str(X.dtype),
        "X_memory_MiB": round(X.nbytes / 1024**2, 1),
        "neg_sampling": {
            "n_neg_per_pos": N_NEG_PER_POS,
            "seed": RANDOM_SEED,
        },
        "elapsed_s": round(elapsed, 2),
    }

    for i, name in enumerate(feature_names):
        col = X[:, i]
        report["feature_stats"][name] = {
            "mean": round(float(np.nanmean(col)), 4),
            "std": round(float(np.nanstd(col)), 4),
            "min": round(float(np.nanmin(col)), 4),
            "max": round(float(np.nanmax(col)), 4),
            "pct_zero": round(float((col == 0).mean() * 100), 2),
            "pct_nan": round(float(np.isnan(col).mean() * 100), 2),
        }

    return report


def report_to_markdown(report: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Feature Engineering Report — {report['variant']}")
    lines.append("")
    lines.append("## Vue d'ensemble")
    lines.append("")
    lines.append(f"- **Paires totales** : {report['n_pairs']:,}")
    lines.append(f"- **Positives** : {report['n_positive']:,}")
    lines.append(f"- **Négatives** : {report['n_negative']:,}")
    lines.append(f"- **Users** : {report['n_users']:,}")
    lines.append(f"- **Items** : {report['n_items']:,}")
    lines.append(f"- **Features** : {report['n_features']}")
    lines.append(f"- **Matrice** : {report['X_shape']} ({report['X_memory_MiB']} MiB)")
    lines.append(f"- **Temps** : {report['elapsed_s']:.1f}s")
    lines.append("")

    lines.append("## Distribution des labels")
    lines.append("")
    ld = report["label_distribution"]
    lines.append(f"- Seuil de pertinence : rating >= {ld['relevance_threshold']}")
    lines.append(f"- Pertinents (1) : {ld['binary_1_relevant']:,}")
    lines.append(f"- Non pertinents (0) : {ld['binary_0_irrelevant']:,}")
    lines.append("")

    lines.append("### Ratings")
    lines.append("")
    for k, v in report["rating_distribution"].items():
        lines.append(f"- {k} : {v:,}")
    lines.append("")

    lines.append("## Groupes LTR")
    lines.append("")
    gs = report["groups_stats"]
    lines.append(f"- Groupes : {gs['n_groups']:,}")
    lines.append(f"- Taille moyenne : {gs['mean_group_size']:.1f}")
    lines.append(f"- Min / Max : {gs['min_group_size']} / {gs['max_group_size']}")
    lines.append("")

    lines.append("## Statistiques des features")
    lines.append("")
    lines.append("| Feature | Mean | Std | Min | Max | %Zero | %NaN |")
    lines.append("|---------|------|-----|-----|-----|-------|------|")
    for name, st in report["feature_stats"].items():
        lines.append(
            f"| {name} | {st['mean']:.4f} | {st['std']:.4f} | "
            f"{st['min']:.4f} | {st['max']:.4f} | {st['pct_zero']:.1f} | {st['pct_nan']:.1f} |"
        )
    lines.append("")

    lines.append("## Sampling négatif")
    lines.append("")
    ns = report["neg_sampling"]
    lines.append(f"- N négatifs par positif : {ns['n_neg_per_pos']}")
    lines.append(f"- Seed : {ns['seed']}")

    return "\n".join(lines)


# ======================================================================
#  SAUVEGARDE
# ======================================================================

def save_artifacts(
    variant_dir: Path,
    X: np.ndarray,
    y_binary: np.ndarray,
    y_rating: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    report: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, str]:
    out_dir = variant_dir / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results") / variant_dir.relative_to("data") / "task3"
    results_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    np.save(out_dir / ARTIFACTS_OUT["X_train"], X)
    paths["X_train"] = str(out_dir / ARTIFACTS_OUT["X_train"])

    np.save(out_dir / ARTIFACTS_OUT["y_train"], y_binary)
    paths["y_train"] = str(out_dir / ARTIFACTS_OUT["y_train"])

    np.save(out_dir / ARTIFACTS_OUT["y_rating"], y_rating)
    paths["y_rating"] = str(out_dir / ARTIFACTS_OUT["y_rating"])

    np.save(out_dir / ARTIFACTS_OUT["groups"], groups)
    paths["groups"] = str(out_dir / ARTIFACTS_OUT["groups"])

    with open(out_dir / ARTIFACTS_OUT["feature_names"], "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    paths["feature_names"] = str(out_dir / ARTIFACTS_OUT["feature_names"])

    json_path = results_dir / ARTIFACTS_OUT["report_json"]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    paths["report_json"] = str(json_path)

    md_path = results_dir / ARTIFACTS_OUT["report_md"]
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    paths["report_md"] = str(md_path)

    if verbose:
        for name, p in paths.items():
            size = Path(p).stat().st_size
            print(f"  saved {name}: {p} ({_fmt_size(size)})")

    return paths


# ======================================================================
#  PIPELINE PRINCIPAL
# ======================================================================

def run_feature_engineering(
    variant_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    variant = variant_dir.name
    t0 = time.perf_counter()

    print(f"\n{'='*70}")
    print(f"  Tâche 3 — Feature Engineering : {variant}")
    print(f"{'='*70}\n")

    # 1. Charger le train
    train_path = variant_dir / "train_interactions.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Train manquant: {train_path}")

    all_cols = REQUIRED_INTERACTION_COLS + REQUIRED_META_COLS + OPTIONAL_META_COLS
    available_cols = pd.read_parquet(train_path, columns=None, engine="pyarrow").columns.tolist()
    load_cols = [c for c in all_cols if c in available_cols]

    train_df = pd.read_parquet(train_path, columns=load_cols)
    if verbose:
        mem = train_df.memory_usage(deep=True).sum() / 1024**2
        print(f"  [train] {len(train_df):,} rows, {len(load_cols)} cols, {mem:.0f} MiB")

    missing_req = _check_columns(train_df, REQUIRED_INTERACTION_COLS, "interactions")
    if missing_req:
        raise ValueError(f"Colonnes obligatoires manquantes: {missing_req}")
    _check_columns(train_df, REQUIRED_META_COLS, "metadata")

    present_optional = [c for c in OPTIONAL_META_COLS if c in train_df.columns]
    absent_optional = [c for c in OPTIONAL_META_COLS if c not in train_df.columns]
    if verbose:
        print(f"  [cols] optionnelles présentes: {present_optional}")
        if absent_optional:
            print(f"  [cols] optionnelles absentes (ignorées): {absent_optional}")

    # 2. Charger les artefacts amont
    artifacts = load_upstream_artifacts(variant_dir, verbose=verbose)

    # 3. Statistiques agrégées
    if verbose:
        print("  [stats] user_stats + item_stats...")
    user_stats = build_user_stats(train_df)
    item_stats = build_item_stats(train_df)

    # 4. Historiques auteur / genre
    if verbose:
        print("  [history] auteur + genre...")
    user_author_hist = build_user_author_history(train_df)
    user_genre_hist = build_user_genre_history(train_df)

    # 5. Sampling des paires
    if verbose:
        print("  [sampling] paires user-item...")
    pairs = sample_training_pairs(
        train_df, artifacts["item_ids"],
        n_neg_per_pos=N_NEG_PER_POS, verbose=verbose,
    )

    # 6. Construction des features
    X, feature_names = build_feature_matrix(
        pairs=pairs,
        train_df=train_df,
        artifacts=artifacts,
        user_stats=user_stats,
        item_stats=item_stats,
        user_author_hist=user_author_hist,
        user_genre_hist=user_genre_hist,
        verbose=verbose,
    )

    # 7. Labels et groupes
    y_binary, y_rating = build_labels(pairs)
    groups = build_groups(pairs)

    elapsed = time.perf_counter() - t0

    # 8. Rapport
    report = build_report(
        variant=variant,
        pairs=pairs,
        X=X,
        y_binary=y_binary,
        y_rating=y_rating,
        groups=groups,
        feature_names=feature_names,
        elapsed=elapsed,
    )

    # 9. Console summary
    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Résumé — {variant}")
        print(f"{'─'*70}")
        print(f"  Paires:    {report['n_pairs']:,} (pos={report['n_positive']:,}, neg={report['n_negative']:,})")
        print(f"  Features:  {report['n_features']} → {feature_names}")
        print(f"  Labels:    relevant={report['label_distribution']['binary_1_relevant']:,}, "
              f"irrelevant={report['label_distribution']['binary_0_irrelevant']:,}")
        print(f"  Groupes:   {report['groups_stats']['n_groups']:,} "
              f"(mean={report['groups_stats']['mean_group_size']:.1f})")
        print(f"  Matrice:   {report['X_shape']} ({report['X_memory_MiB']} MiB)")
        print(f"  Temps:     {elapsed:.1f}s")

    # 10. Sauvegarde
    paths = save_artifacts(
        variant_dir=variant_dir,
        X=X, y_binary=y_binary, y_rating=y_rating,
        groups=groups, feature_names=feature_names,
        report=report, verbose=verbose,
    )

    # Nettoyage
    del train_df, pairs, X, y_binary, y_rating, groups
    del artifacts, user_stats, item_stats, user_author_hist, user_genre_hist
    gc.collect()

    return report


# ======================================================================
#  MAIN
# ======================================================================

def main() -> None:
    t0 = time.perf_counter()

    for variant_dir in VARIANT_DIRS:
        try:
            report = run_feature_engineering(variant_dir, verbose=True)
        except FileNotFoundError as e:
            print(f"\n  [SKIP] {variant_dir.name}: {e}")
            continue
        except Exception as e:
            print(f"\n  [ERROR] {variant_dir.name}: {e}")
            raise

    print(f"\n{'='*70}")
    print(f"  Feature engineering terminé — total {time.perf_counter()-t0:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()