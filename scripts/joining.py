from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
from typing import Tuple

import gc
import json
import os
import time
import re

import pandas as pd
import pyarrow.parquet as pq







@dataclass
class SourceInfo:
    name: str
    stage: str
    variant: str
    role: str                 # interactions | metadata | optional_debug
    kind: str                 # single | union
    paths: List[str]
    exists: bool
    format: str
    size_bytes: int
    n_rows: int
    n_cols: int
    columns: List[str]
    details: Dict[str, Any]



_HTML_PATTERN = re.compile(r"<[^>]+>|&[a-zA-Z]+;|&\#\d+;")

INTERACTION_MIN_COLS = ["user_id", "parent_asin", "rating", "timestamp"]

METADATA_SCALAR_COLS = ["title", "subtitle"]
METADATA_LIST_COLS = ["features", "description", "categories"]
METADATA_STRUCT_COLS = ["average_rating", "rating_number", "price"]
METADATA_NESTED_COLS = ["author", "details"]


METADATA_TEXT_COLS = METADATA_SCALAR_COLS + METADATA_LIST_COLS + METADATA_NESTED_COLS

REQUIRED_INTERACTION_COLS = {"user_id", "parent_asin", "rating", "timestamp"}
REQUIRED_METADATA_COLS = {"parent_asin"}

COLUMN_JUSTIFICATIONS: Dict[str, str] = {
    "user_id": "Identifiant unique de l'utilisateur, clé primaire pour les interactions.",
    "parent_asin": "Clé de jointure commune interactions ↔ metadata.",
    "rating": "Note attribuée par l'utilisateur, variable cible du système de recommandation.",
    "timestamp": "Horodatage de l'interaction, nécessaire pour le split temporel.",
    "title": "Titre du livre, exploitable pour la représentation textuelle (TF-IDF, embeddings).",
    "subtitle": "Sous-titre du livre, complément textuel du titre.",
    "description": "Description éditoriale, riche en contenu sémantique pour le content-based filtering.",
    "categories": "Taxonomie hiérarchique Amazon, utile pour le filtrage par genre.",
    "features": "Points clés marketing, contenu textuel complémentaire.",
    "author": "Structure imbriquée → extraction de author_name. Permet le filtrage par auteur.",
    "details": "Structure imbriquée → extraction de Publisher et Language. Attributs exploitables.",
    "average_rating": "Note moyenne agrégée, signal de popularité pour le modèle.",
    "rating_number": "Nombre total de notes, indicateur de popularité/confiance.",
    "price": "Prix du livre, 23.9% manquant → imputation médiane retenue car distribution asymétrique.",
    # Colonnes exclues
    "main_category": "Exclu : valeur quasi-constante ('Books' pour >99% des lignes).",
    "images": "Exclu : données binaires/URL non exploitables pour le filtrage collaboratif.",
    "videos": "Exclu : quasi-vide, non pertinent pour la recommandation textuelle.",
    "store": "Exclu : redondant avec author (contient 'Author Name (Author)').",
    "bought_together": "Exclu : colonne entièrement nulle dans le parquet.",
    "text": "Exclu des colonnes retenues P2 : texte libre de review, très volumineux, non nécessaire pour la jointure metadata.",
    "asin": "Exclu : redondant avec parent_asin pour les éditions groupées.",
    "helpful_vote": "Exclu : signal faible, non nécessaire pour la jointure P2.",
    "verified_purchase": "Exclu : booléen de confiance, hors périmètre P2.",
}

CONTENT_REPRESENTATION_COLS = {
    "title": "TF-IDF / embeddings — titre du livre",
    "description": "TF-IDF / embeddings — description éditoriale",
    "categories": "Encodage catégoriel / multi-hot — taxonomie Amazon",
    "features": "TF-IDF — points clés marketing",
    "author_name": "Encodage catégoriel — filtrage par auteur",
}

LEARNING_FEATURE_COLS = {
    "average_rating": "Variable continue — popularité agrégée de l'item",
    "rating_number": "Variable continue — volume de notes (confiance)",
    "price": "Variable continue — prix (imputé médiane)",
    "details_publisher": "Variable catégorielle — éditeur",
    "details_language": "Variable catégorielle — langue",
    "author_name": "Variable catégorielle — auteur (partagé avec contenu)",
}
MISSINGNESS_JUSTIFICATIONS: Dict[str, str] = {
    "parent_asin": "Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable.",
    "user_id": "Clé primaire interactions — ligne non identifiable sans user_id.",
    "timestamp": "Nécessaire au split temporel train/test — ligne inutilisable sans date.",
    "rating": "Variable cible du système de recommandation — ligne sans note exclue.",
    "title": "Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description.",
    "subtitle": "Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation.",
    "description": "Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features.",
    "categories": "Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires.",
    "features": "Points clés marketing ; vide tolérable, données complémentaires au content-based.",
    "author": "Structure imbriquée → extraction de author_name ; vide si auteur inconnu, impact limité.",
    "details": "Structure imbriquée → publisher/language ; vide tolérable, attributs secondaires.",
    "author_name": "Extrait de struct author ; vide si auteur inconnu, impact limité.",
    "details_publisher": "Extrait de struct details ; vide tolérable, attribut catégoriel secondaire.",
    "details_language": "Extrait de struct details ; vide tolérable, quasi-constant ('English').",
    "average_rating": "Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%.",
    "rating_number": "Volume de notes ; NaN rare, indicateur de confiance secondaire.",
    "price": "Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne.",
}

COLUMN_TYPE_MAP: Dict[str, str] = {}
for _c in INTERACTION_MIN_COLS:
    COLUMN_TYPE_MAP[_c] = "numérique" if _c in ("rating", "timestamp") else "clé (identifiant)"
for _c in METADATA_SCALAR_COLS:
    COLUMN_TYPE_MAP[_c] = "textuelle (scalaire)"
for _c in METADATA_LIST_COLS:
    COLUMN_TYPE_MAP[_c] = "textuelle (liste)"
for _c in METADATA_NESTED_COLS:
    COLUMN_TYPE_MAP[_c] = "catégorielle (struct imbriqué)"
for _c in METADATA_STRUCT_COLS:
    COLUMN_TYPE_MAP[_c] = "numérique"
# Colonnes post-normalisation dérivées de structs
COLUMN_TYPE_MAP["author_name"] = "catégorielle (extraite de struct)"
COLUMN_TYPE_MAP["details_publisher"] = "catégorielle (extraite de struct)"
COLUMN_TYPE_MAP["details_language"] = "catégorielle (extraite de struct)"





def _required_cols_for_role(
    role: str
) -> set[str]:
    if role == "interactions":
        return REQUIRED_INTERACTION_COLS
    if role == "metadata":
        return REQUIRED_METADATA_COLS
    return set()





def get_manifest(
    include_optional_raw: bool = False
) -> Dict[str, Dict[str, Any]]:
    base = Path("data/processed")
    raw_base = Path("data/raw/parquet")

    manifest: Dict[str, Dict[str, Any]] = {

        "active_pre_split": {
            "stage": "pre_split",
            "variant": "active",
            "role": "interactions",
            "kind": "single",
            "paths": [str(base / "sample-active-users" / "active_users_filtered.parquet")],
        },
        "active_post_split_union": {
            "stage": "post_split",
            "variant": "active",
            "role": "interactions",
            "kind": "union",
            "paths": [
                str(base / "sample-active-users" / "splits" / "train.parquet"),
                str(base / "sample-active-users" / "splits" / "test.parquet"),
            ],
        },
        "temporal_pre_split": {
            "stage": "pre_split",
            "variant": "temporal",
            "role": "interactions",
            "kind": "single",
            "paths": [str(base / "sample-temporal" / "temporal_filtered.parquet")],
        },
        "temporal_post_split_union": {
            "stage": "post_split",
            "variant": "temporal",
            "role": "interactions",
            "kind": "union",
            "paths": [
                str(base / "sample-temporal" / "splits" / "train.parquet"),
                str(base / "sample-temporal" / "splits" / "test.parquet"),
            ],
        },
        "metadata": {
            "stage": "raw",
            "variant": "meta_books",
            "role": "metadata",
            "kind": "single",
            "paths": [str(raw_base / "meta_Books.parquet")],
        },
    }

    # Optional debug source (not part of official P2 ratios)
    if include_optional_raw:
        manifest["books_reviews_raw"] = {
            "stage": "raw",
            "variant": "books",
            "role": "optional_debug",
            "kind": "single",
            "paths": [str(raw_base / "Books.parquet")],
        }

    return manifest





def build_p1_reuse_note(
    manifest: Dict[str, Dict[str, Any]], 
    sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    # sources = result["sources"] (liste de dict après asdict)
    interactions = [s for s in sources if s.get("role") == "interactions"]

    return {
        "statement": "P2 réutilise les sous-ensembles P1 (active/temporal, filtered + splits).",
        "interaction_targets": [s["name"] for s in interactions],
        "rows_by_target": {s["name"]: s["n_rows"] for s in interactions},
        "paths_by_target": {s["name"]: s["paths"] for s in interactions},
        "methodological_note": (
            "Aucun nouvel échantillonnage massif du corpus complet n'est effectué; "
            "les jeux issus de P1 sont réexploités pour la fusion avec meta_Books."
        ),
    }





def validate_manifest_paths(manifest: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    status: Dict[str, bool] = {}
    for name, cfg in manifest.items():
        status[name] = all(os.path.exists(p) for p in cfg["paths"])
    return status





def _safe_size(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0





def _parquet_doc(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "exists": False,
            "abs_path": str(Path(path).resolve()),
            "format": "parquet",
            "size_bytes": 0,
            "n_rows": 0,
            "columns": [],
            "n_cols": 0,
        }

    pf = pq.ParquetFile(path)
    cols = list(pf.schema_arrow.names)
    return {
        "exists": True,
        "abs_path": str(Path(path).resolve()),
        "format": "parquet",
        "size_bytes": _safe_size(path),
        "n_rows": int(pf.metadata.num_rows) if pf.metadata is not None else 0,
        "columns": cols,
        "n_cols": len(cols),
    }





def _union_doc(train_path: str, test_path: str) -> Dict[str, Any]:
    train_doc = _parquet_doc(train_path)
    test_doc = _parquet_doc(test_path)

    cols_train = set(train_doc["columns"])
    cols_test = set(test_doc["columns"])
    common = sorted(cols_train.intersection(cols_test))
    train_only = sorted(cols_train - cols_test)
    test_only = sorted(cols_test - cols_train)

    return {
        "exists": train_doc["exists"] and test_doc["exists"],
        "format": "parquet_union",
        "size_bytes": int(train_doc["size_bytes"]) + int(test_doc["size_bytes"]),
        "n_rows": int(train_doc["n_rows"]) + int(test_doc["n_rows"]),
        "columns": common,   # conservative, shared schema
        "n_cols": len(common),
        "parts": {
            "train": train_doc,
            "test": test_doc,
        },
        "columns_intersection": common,
        "columns_train_only": train_only,
        "columns_test_only": test_only,
    }





def _compute_source_info(name: str, cfg: Dict[str, Any]) -> SourceInfo:
    kind = cfg["kind"]
    paths = cfg["paths"]

    if kind == "single":
        doc = _parquet_doc(paths[0])
    elif kind == "union":
        doc = _union_doc(paths[0], paths[1])
    else:
        raise ValueError(f"Unknown kind={kind} for source={name}")

    return SourceInfo(
        name=name,
        stage=cfg["stage"],
        variant=cfg["variant"],
        role=cfg["role"],
        kind=kind,
        paths=paths,
        exists=bool(doc["exists"]),
        format=str(doc["format"]),
        size_bytes=int(doc["size_bytes"]),
        n_rows=int(doc["n_rows"]),
        n_cols=int(doc["n_cols"]),
        columns=list(doc["columns"]),
        details=doc,
    )





def collect_source_documentation(
    manifest: Dict[str, Dict[str, Any]],
    verbose: bool = True,
) -> List[SourceInfo]:
    infos: List[SourceInfo] = []
    for name, cfg in manifest.items():
        info = _compute_source_info(name, cfg)
        infos.append(info)
        if verbose:
            print(
                f"[{name}] exists={info.exists} "
                f"rows={info.n_rows:,} cols={info.n_cols} "
                f"size={info.size_bytes / (1024**2):.2f} MB"
            )
    return infos





def load_target_df(
    cfg: Dict[str, Any],
    columns: List[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    kind = cfg["kind"]
    paths = cfg["paths"]

    if verbose:
        print(
            f"\nload_target_df \n"
            f"kind: {kind}, paths: {paths} \n"
            f"columns: {columns} \n"
        )
    if kind == "single":
        return pd.read_parquet(paths[0], columns=columns, 
            engine='pyarrow')

    if kind == "union":
        train_df = pd.read_parquet(paths[0], columns=columns, 
            engine='pyarrow')
        test_df = pd.read_parquet(paths[1], columns=columns, 
            engine='pyarrow')
        combined = pd.concat([train_df, test_df], ignore_index=True)
        del train_df, test_df
        return combined

    raise ValueError(f"Unknown kind={kind}")





def check_required_columns(df: pd.DataFrame, required_cols: set[str]) -> Dict[str, Any]:
    present = set(df.columns)
    missing = sorted(required_cols - present)
    return {
        "required": sorted(required_cols),
        "missing_required": missing,
        "ok": len(missing) == 0,
    }


def missingness_report(
    df: pd.DataFrame,
    cols: List[str],
) -> List[Dict[str, Any]]:
    import numpy as np
    out = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            out.append({
                "column": col, "missing_count": None, "missing_pct": None,
                "empty_count": None, "empty_pct": None,
                "effective_missing_pct": None, "strategy": "absent",
            })
            continue

        s = df[col]
        m_null = int(s.isna().sum())

        def _is_empty(val):
            if val is None:
                return True
            if isinstance(val, str) and val.strip() == "":
                return True
            if isinstance(val, (list, tuple)) and len(val) == 0:
                return True
            if isinstance(val, np.ndarray) and len(val) == 0:
                return True
            return False

        m_empty = int(s.dropna().apply(_is_empty).sum())
        m_effective = m_null + m_empty

        out.append({
            "column": col,
            "missing_count": m_null,
            "missing_pct": round(m_null / n * 100, 4) if n else 0.0,
            "empty_count": m_empty,
            "empty_pct": round(m_empty / n * 100, 4) if n else 0.0,
            "effective_missing_pct": round(m_effective / n * 100, 4) if n else 0.0,
            "strategy": None,
        })
    return out





def attach_missingness_strategy(
    report_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    scalar_cols = set(METADATA_SCALAR_COLS)
    list_cols = set(METADATA_LIST_COLS)
    nested_cols = set(METADATA_NESTED_COLS)
    numeric_cols = set(METADATA_STRUCT_COLS)
    key_cols = {"parent_asin", "user_id", "timestamp"}

    for r in report_rows:
        col = r["column"]
        r["column_type"] = COLUMN_TYPE_MAP.get(col, "autre")
        if r["missing_pct"] is None:
            r["strategy"] = "colonne absente"
        elif col in key_cols:
            r["strategy"] = "supprimer lignes incomplètes (clé obligatoire)"
        elif col in scalar_cols:
            r["strategy"] = "remplacer NaN par chaîne vide"
        elif col in list_cols:
            r["strategy"] = "joindre éléments en string, vide si absent"
        elif col in nested_cols:
            r["strategy"] = "aplatir struct (extraire champ clé en string)"
        elif col in numeric_cols:
            r["strategy"] = "imputation médiane (ou exclusion si trop manquant)"
        elif col == "rating":
            r["strategy"] = "supprimer lignes incomplètes (variable cible)"
        else:
            r["strategy"] = "au cas par cas / hors périmètre"
        r["justification"] = MISSINGNESS_JUSTIFICATIONS.get(col, "—")
    return report_rows



def coerce_parent_asin_to_string(
    df: pd.DataFrame
) -> Dict[str, Any]:
    if "parent_asin" not in df.columns:
        return {
            "coercion_applied": False,
            "warning": "parent_asin absent",
            "changed_non_null_values": None,
            "non_null_before": 0,
            "non_null_after": 0,
        }

    s_before = df["parent_asin"]
    non_null_before = int(s_before.notna().sum())

    # Type string pandas (préserve les NA)
    s_after = s_before.astype("string")
    non_null_after = int(s_after.notna().sum())

    # Comparaison seulement sur les non-null
    before_nn = s_before[s_before.notna()].astype(str)
    after_nn = s_after[s_after.notna()].astype(str)
    changed_non_null = int((before_nn.values != after_nn.values).sum())

    df["parent_asin"] = s_after

    warning = None
    if changed_non_null > 0:
        warning = f"{changed_non_null} valeur(s) non nulle(s) modifiée(s) par la coercition"

    return {
        "coercion_applied": True,
        "warning": warning,
        "changed_non_null_values": changed_non_null,
        "non_null_before": non_null_before,
        "non_null_after": non_null_after,
    }





def count_missing_parent_asin(
    df: pd.DataFrame
) -> Dict[str, Any]:
    if "parent_asin" not in df.columns:
        return {"missing_count": None, "missing_pct": None, "n_rows": len(df)}

    n_rows = len(df)
    missing_count = int(df["parent_asin"].isna().sum())
    missing_pct = (missing_count / n_rows * 100.0) if n_rows else 0.0

    return {
        "missing_count": missing_count,
        "missing_pct": round(missing_pct, 6),
        "n_rows": n_rows,
    }




def check_duplicates(
    df: pd.DataFrame,
    role: str,
) -> Dict[str, Any]:
    n = len(df)
    
    hashable_cols = [c for c in df.columns
                    if df[c].dropna().apply(lambda v: isinstance(v, (str, int, float, bool))).all()]
    exact_dups = int(df[hashable_cols].duplicated().sum()) if hashable_cols else 0

    result: Dict[str, Any] = {
        "n_rows": n,
        "exact_duplicates": exact_dups,
        "exact_duplicates_pct": round(exact_dups / n * 100, 4) if n else 0.0,
    }

    if role == "interactions" and "user_id" in df.columns and "parent_asin" in df.columns:
        pair_dups = int(df.duplicated(subset=["user_id", "parent_asin"]).sum())
        result["user_item_duplicates"] = pair_dups
        result["user_item_duplicates_pct"] = round(pair_dups / n * 100, 4) if n else 0.0

    if role == "metadata" and "parent_asin" in df.columns:
        key_dups = int(df.duplicated(subset=["parent_asin"]).sum())
        result["parent_asin_duplicates"] = key_dups
        result["parent_asin_duplicates_pct"] = round(key_dups / n * 100, 4) if n else 0.0

    return result





def validate_rating_range(
    df: pd.DataFrame,
    col: str = "rating",
    valid_min: float = 1.0,
    valid_max: float = 5.0,
) -> Dict[str, Any]:
    if col not in df.columns:
        return {"column": col, "present": False}

    s = df[col].dropna()
    out_of_range = int(((s < valid_min) | (s > valid_max)).sum())
    return {
        "column": col,
        "present": True,
        "count": len(s),
        "min": float(s.min()) if len(s) else None,
        "max": float(s.max()) if len(s) else None,
        "mean": round(float(s.mean()), 4) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "out_of_range_count": out_of_range,
        "out_of_range_pct": round(out_of_range / len(s) * 100, 4) if len(s) else 0.0,
        "valid_range": [valid_min, valid_max],
        "ok": out_of_range == 0,
    }





def validate_timestamp(
    df: pd.DataFrame,
    col: str = "timestamp",
    min_year: int = 2000,
    max_year: int = 2026,
) -> Dict[str, Any]:
    if col not in df.columns:
        return {"column": col, "present": False}

    s = df[col].dropna()
    result: Dict[str, Any] = {
        "column": col,
        "present": True,
        "dtype": str(s.dtype),
        "count": len(s),
        "ok": True,
        "warnings": [],
    }

    try:
        if pd.api.types.is_numeric_dtype(s):
            ts = s
            if s.max() > 1e12:
                ts = s / 1000  # ms → s
            dt = pd.to_datetime(ts, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(s, errors="coerce")

        n_unconvertible = int(dt.isna().sum()) - int(s.isna().sum())
        result["unconvertible_count"] = max(n_unconvertible, 0)

        dt_valid = dt.dropna()
        if len(dt_valid):
            result["min_date"] = str(dt_valid.min())
            result["max_date"] = str(dt_valid.max())
            result["min_year"] = int(dt_valid.dt.year.min())
            result["max_year"] = int(dt_valid.dt.year.max())

            too_early = int((dt_valid.dt.year < min_year).sum())
            too_late = int((dt_valid.dt.year > max_year).sum())
            result["outliers_before_" + str(min_year)] = too_early
            result["outliers_after_" + str(max_year)] = too_late

            if too_early or too_late or n_unconvertible > 0:
                result["ok"] = False
                if too_early:
                    result["warnings"].append(f"{too_early} timestamps avant {min_year}")
                if too_late:
                    result["warnings"].append(f"{too_late} timestamps après {max_year}")
                if n_unconvertible > 0:
                    result["warnings"].append(f"{n_unconvertible} valeurs non convertibles")
        else:
            result["ok"] = False
            result["warnings"].append("Aucun timestamp valide après conversion")
    except Exception as e:
        result["ok"] = False
        result["warnings"].append(f"Erreur de conversion: {e}")

    return result






def text_quality_report(
    df: pd.DataFrame,
    cols: List[str],
) -> List[Dict[str, Any]]:
    out = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            out.append({"column": col, "present": False})
            continue

        s = df[col].dropna()
        str_s = s.astype(str)
        lengths = str_s.str.len()

        n_empty = int((str_s.str.strip() == "").sum())
        n_html = int(str_s.apply(lambda x: bool(_HTML_PATTERN.search(x))).sum())

        out.append({
            "column": col,
            "present": True,
            "non_null_count": len(s),
            "empty_or_blank_count": n_empty,
            "empty_or_blank_pct": round(n_empty / n * 100, 4) if n else 0.0,
            "avg_length": round(float(lengths.mean()), 1) if len(lengths) else 0.0,
            "min_length": int(lengths.min()) if len(lengths) else 0,
            "max_length": int(lengths.max()) if len(lengths) else 0,
            "median_length": int(lengths.median()) if len(lengths) else 0,
            "html_noise_count": n_html,
            "html_noise_pct": round(n_html / len(s) * 100, 4) if len(s) else 0.0,
        })
    return out






def run_schema_key_checks_for_target(
    name: str, 
    cfg: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "target": name,
        "stage": cfg["stage"],
        "variant": cfg["variant"],
        "role": cfg["role"],
        "ok": True,
        "required_columns": {},
        "coercion": {},
        "missing_keys": {},
        "warnings": [],
    }

    # # Ignore les sources debug dans les checks officiels
    # if cfg["role"] not in {"interactions", "metadata"}:
    #     out["warnings"].append("Source non officielle Task 3 (ignorée)")
    #     return out


    required_cols = _required_cols_for_role(cfg["role"])
    req = check_required_columns(df, required_cols)
    out["required_columns"] = req

    if not req["ok"]:
        out["ok"] = False
        out["warnings"].append(f"Colonnes obligatoires manquantes: {req['missing_required']}")
        return out

    coercion = coerce_parent_asin_to_string(df)
    out["coercion"] = coercion
    if coercion.get("warning"):
        out["warnings"].append(coercion["warning"])

    missing = count_missing_parent_asin(df)
    out["missing_keys"] = {
        "missing_parent_asin_count": missing["missing_count"],
        "missing_parent_asin_pct": missing["missing_pct"],
        "n_rows": missing["n_rows"],
    }

    return out





def select_exploitable_columns(
    inter_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    source_inter_cols: List[str] | None = None,
    source_meta_cols: List[str] | None = None,
) -> Dict[str, Any]:
    if source_inter_cols is None:
        source_inter_cols = list(inter_df.columns)
    if source_meta_cols is None:
        source_meta_cols = list(meta_df.columns)

    inter_available = [c for c in INTERACTION_MIN_COLS if c in inter_df.columns]
    meta_scalar = [c for c in METADATA_SCALAR_COLS if c in meta_df.columns]
    meta_list = [c for c in METADATA_LIST_COLS if c in meta_df.columns]
    meta_nested = [c for c in METADATA_NESTED_COLS if c in meta_df.columns]
    meta_struct = [c for c in METADATA_STRUCT_COLS if c in meta_df.columns]

    all_meta_kept = set(meta_scalar + meta_list + meta_nested + meta_struct + ["parent_asin"])
    all_inter_kept = set(inter_available)

    return {
        "interactions_kept": inter_available,
        "metadata_text_kept": meta_scalar + meta_list + meta_nested,
        "metadata_struct_kept": meta_struct,
        "metadata_scalar": meta_scalar,
        "metadata_list": meta_list,
        "metadata_nested": meta_nested,
        "ignored_interactions_cols": [c for c in source_inter_cols if c not in all_inter_kept],
        "ignored_metadata_cols": [c for c in source_meta_cols if c not in all_meta_kept],
        "justifications": {
            c: COLUMN_JUSTIFICATIONS.get(c, "")
            for c in inter_available + meta_scalar + meta_list + meta_nested + meta_struct
        },
        "exclusion_reasons": {
            c: COLUMN_JUSTIFICATIONS.get(c, "au cas par cas / hors périmètre")
            for c in source_inter_cols + source_meta_cols
            if c not in all_inter_kept and c not in all_meta_kept
        },
    }





def compute_join_quality_metrics(
    inter_df: pd.DataFrame,
    meta_df: pd.DataFrame = None,
    meta_key_set: set[str] = None,
    meta_total_count: int | None = None,
) -> Dict[str, Any]:
    inter_key = inter_df["parent_asin"].astype("string")
    inter_items = set(inter_key.dropna().unique().tolist())

    if meta_key_set is None:
        if meta_df is None:
            raise ValueError("Provide meta_df or meta_key_set")
        meta_key_set = set(meta_df["parent_asin"].astype("string").dropna().unique().tolist())

    if meta_total_count is None:
        meta_total_count = len(meta_key_set)

    common_items = inter_items.intersection(meta_key_set)
    matched_mask = inter_key.isin(meta_key_set)

    n_inter_total = len(inter_df)
    n_inter_joined = int(matched_mask.sum())
    n_items_total = len(inter_items)
    n_items_with_meta = len(common_items)
    n_meta_orphan = meta_total_count - n_items_with_meta

    return {
        "nb_parent_asin_communs": n_items_with_meta,
        "nb_interactions_totales": n_inter_total,
        "nb_interactions_jointes": n_inter_joined,
        "ratio_interactions_jointes": (n_inter_joined / n_inter_total) if n_inter_total else 0.0,
        "nb_items_totaux": n_items_total,
        "nb_items_avec_meta": n_items_with_meta,
        "ratio_items_avec_meta": (n_items_with_meta / n_items_total) if n_items_total else 0.0,
        "interactions_non_jointes_si_inner_join": n_inter_total - n_inter_joined,
        "items_sans_meta": n_items_total - n_items_with_meta,
        "nb_meta_total": meta_total_count,
        "nb_meta_orphelines": n_meta_orphan,
        "ratio_meta_utilisees": (n_items_with_meta / meta_total_count) if meta_total_count else 0.0,
        "interpretation": (
            f"{n_items_with_meta:,} items sur {meta_total_count:,} metadata "
            f"({n_items_with_meta / meta_total_count * 100:.1f}%), "
            f"cohérent avec le sous-échantillonnage P1."
            if meta_total_count else "N/A"
        ),
    }






def _flatten_struct_col(
    series: pd.Series, 
    key: str
) -> pd.Series:
    """Extract a single key from a struct/dict column, return as string."""
    def _extract(val):
        if isinstance(val, dict):
            return str(val.get(key, ""))
        return ""
    return series.apply(_extract)






def _join_list_col(
    series: pd.Series, 
    sep: str = " | "
) -> pd.Series:
    """Join list/array elements into a single string."""
    def _join(val):
        if isinstance(val, (list, tuple)):
            return sep.join(str(x) for x in val if x)
        try:
            import numpy as np
            if isinstance(val, np.ndarray):
                return sep.join(str(x) for x in val if x)
        except ImportError:
            pass
        return ""
    return series.apply(_join)





def normalize_metadata_columns(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalize metadata columns in-place for parquet-safe serialization.

    - Scalar text cols (title): fillna("")
    - List cols (description, categories, features): join elements → string
    - Struct cols (author → author_name, details → details_publisher): extract key → string
    """
    for c in METADATA_SCALAR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("")

    for c in METADATA_LIST_COLS:
        if c in df.columns:
            df[c] = _join_list_col(df[c])

    if "author" in df.columns:
        df["author_name"] = _flatten_struct_col(df["author"], "name")
        df.drop(columns=["author"], inplace=True)

    if "details" in df.columns:
        df["details_publisher"] = _flatten_struct_col(df["details"], "Publisher")
        df["details_language"] = _flatten_struct_col(df["details"], "Language")
        df.drop(columns=["details"], inplace=True)

    if "price" in df.columns:
        median_price = df["price"].median()
        df["price"] = df["price"].fillna(median_price)

    return df





def build_joined_dataset(
    inter_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    meta_keep_cols: List[str],
    verbose: bool = True,
) -> pd.DataFrame:

    if verbose:
        print("\nbuild_joined_dataset")
    inter_df = inter_df.copy()
    inter_df["parent_asin"] = inter_df["parent_asin"].astype("string")
    if verbose:
        print(f"\nlen(inter_df.columns): {len(inter_df.columns)}\n"
            f"inter_df.columns: {inter_df.columns}")


    keep = ["parent_asin"] + [c for c in meta_keep_cols if c in meta_df.columns]
    if verbose:
        print(f"\nlen(keep): {len(keep)}\n")
        print(f"keep: {keep}")
    
    meta_slim = meta_df[keep].drop_duplicates(subset=["parent_asin"], keep="first")
    if verbose:
        print(
            f"\nlen(meta_slim.columns): {len(meta_slim.columns)}\n"
            f"meta_slim.columns: {meta_slim.columns}\n"
        )
    
    joined = inter_df.merge(meta_slim, on="parent_asin", how="left")
    del inter_df, meta_slim

    joined = joined[joined["parent_asin"].notna()].copy()

    joined = normalize_metadata_columns(joined)

    return joined




def clean_joined_dataset(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Nettoie le dataset joint : suppression NaN sur clés, dédoublonnage interactions.

    Returns
    -------
    (cleaned_df, cleaning_report)
    """
    n_before = len(df)
    items_before = df["parent_asin"].nunique()
    users_before = df["user_id"].nunique() if "user_id" in df.columns else None

    drop_reasons: Dict[str, int] = {}

    # 1) Supprimer les lignes avec NaN sur les colonnes clés
    key_cols = [c for c in ["user_id", "parent_asin", "rating", "timestamp"] if c in df.columns]
    before_key_drop = len(df)
    df = df.dropna(subset=key_cols).copy()
    drop_reasons["missing_key_cols"] = before_key_drop - len(df)

    # 2) Dédoublonner les interactions (garder la plus récente par timestamp)
    dedup_subset = ["user_id", "parent_asin"]
    if all(c in df.columns for c in dedup_subset):
        before_dedup = len(df)
        sort_col = "timestamp" if "timestamp" in df.columns else None
        if sort_col is not None:
            df = df.sort_values(sort_col, ascending=False)
        df = df.drop_duplicates(subset=dedup_subset, keep="first")
        drop_reasons["interaction_duplicates"] = before_dedup - len(df)

    n_after = len(df)
    items_after = df["parent_asin"].nunique()
    users_after = df["user_id"].nunique() if "user_id" in df.columns else None

    report = {
        "before": {"n_rows": n_before, "n_items": items_before, "n_users": users_before},
        "after":  {"n_rows": n_after,  "n_items": items_after,  "n_users": users_after},
        "dropped_rows": n_before - n_after,
        "dropped_reason": drop_reasons,
    }

    if verbose:
        print(f"\n[clean] {n_before} → {n_after} lignes "
              f"(−{n_before - n_after}: keys={drop_reasons.get('missing_key_cols', 0)}, "
              f"dups={drop_reasons.get('interaction_duplicates', 0)})")

    return df, report




def post_cleaning_checks(
    df: pd.DataFrame,
    items_before: set,
) -> Dict[str, Any]:
    """Vérifications de cohérence après nettoyage."""
    checks: Dict[str, Any] = {}

    # a) Doublons résiduels (user_id, parent_asin)
    if "user_id" in df.columns and "parent_asin" in df.columns:
        residual_dups = int(df.duplicated(subset=["user_id", "parent_asin"]).sum())
        checks["residual_pair_duplicates"] = residual_dups
        checks["residual_pair_duplicates_ok"] = residual_dups == 0

    # b) Distribution rating inchangée / pas d'aberrations
    if "rating" in df.columns:
        checks["rating_post_clean"] = validate_rating_range(df)

    # c) Vérifier que pas de parent_asin critique perdu
    items_after = set(df["parent_asin"].dropna().unique())
    lost_items = items_before - items_after
    checks["parent_asin_integrity"] = {
        "items_before": len(items_before),
        "items_after": len(items_after),
        "items_lost": len(lost_items),
        "items_lost_pct": round(len(lost_items) / len(items_before) * 100, 4) if items_before else 0.0,
        "ok": len(lost_items) == 0,
    }

    # d) NaN résiduel sur clés (devrait être 0)
    key_cols = [c for c in ["user_id", "parent_asin", "rating", "timestamp"] if c in df.columns]
    residual_na = {c: int(df[c].isna().sum()) for c in key_cols}
    checks["residual_key_nan"] = residual_na
    checks["residual_key_nan_ok"] = all(v == 0 for v in residual_na.values())

    return checks




def save_diagnostics(
    result: Dict[str, Any], 
    out_dir: str = "results/joining"
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "joining_diagnostics.json"
    md_path = out / "joining_diagnostics.md"

    # JSON complet (machine-readable)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines: List[str] = []
    lines.append("# Diagnostic Task 0 — Préparation des données (P2)")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # ---------------------------------------------------------------
    # A) Réutilisation du sous-ensemble P1
    # ---------------------------------------------------------------
    p1_note = result.get("p1_reuse_note", {})
    lines.append("## A. Réutilisation du sous-ensemble de travail")
    lines.append(f"- note: `{p1_note.get('statement', 'N/A')}`")
    lines.append(f"- methodological_note: `{p1_note.get('methodological_note', 'N/A')}`")
    lines.append("")

    # ---------------------------------------------------------------
    # B) Documentation source (chemin, format, taille, shape)
    # ---------------------------------------------------------------
    lines.append("## B. Documentation des sources")
    lines.append("")
    for s in result.get("sources", []):
        lines.append(f"### {s['name']}")
        lines.append(f"- stage: `{s.get('stage')}`")
        lines.append(f"- variant: `{s.get('variant')}`")
        lines.append(f"- role: `{s.get('role')}`")
        lines.append(f"- kind: `{s.get('kind')}`")
        lines.append(f"- exists: `{s.get('exists')}`")
        lines.append(f"- format: `{s.get('format')}`")
        lines.append(f"- rows: `{s.get('n_rows')}`")
        lines.append(f"- cols: `{s.get('n_cols')}`")
        lines.append(f"- size_bytes: `{s.get('size_bytes')}`")
        lines.append(f"- paths: `{s.get('paths')}`")
        lines.append(f"- columns names: `{s.get('columns')}`")
        lines.append("")

    # ---------------------------------------------------------------
    # C) Vérifications schéma et clés
    # ---------------------------------------------------------------
    lines.append("## C. Vérifications schéma et clés (`parent_asin`)")
    lines.append("")
    schema_checks = result.get("schema_checks", {})
    for name, check in schema_checks.items():
        lines.append(f"### {name}")
        lines.append(f"- ok: `{check.get('ok')}`")
        req = check.get("required_columns", {})
        lines.append(f"- missing_required: `{req.get('missing_required', [])}`")
        mk = check.get("missing_keys", {})
        lines.append(f"- missing_parent_asin_count: `{mk.get('missing_parent_asin_count')}`")
        lines.append(f"- missing_parent_asin_pct: `{mk.get('missing_parent_asin_pct')}`")
        coercion = check.get("coercion", {})
        lines.append(f"- coercion_warning: `{coercion.get('warning')}`")
        lines.append(f"- warnings: `{check.get('warnings', [])}`")
        lines.append("")

    lines.append("## C2. Détection de doublons")
    lines.append("")
    dup_checks = result.get("duplicate_checks", {})
    for name, dc in dup_checks.items():
        lines.append(f"### {name}")
        lines.append(f"- n_rows: `{dc.get('n_rows')}`")
        lines.append(f"- doublons exacts: `{dc.get('exact_duplicates')}` ({dc.get('exact_duplicates_pct')}%)")
        if "user_item_duplicates" in dc:
            lines.append(f"- doublons (user_id, parent_asin): `{dc.get('user_item_duplicates')}` ({dc.get('user_item_duplicates_pct')}%)")
        if "parent_asin_duplicates" in dc:
            lines.append(f"- doublons parent_asin: `{dc.get('parent_asin_duplicates')}` ({dc.get('parent_asin_duplicates_pct')}%)")
        lines.append("")
        
    lines.append("## C3. Validation des valeurs (rating, timestamp)")
    lines.append("")
    val_checks = result.get("validation_checks", {})
    for name, vc in val_checks.items():
        lines.append(f"### {name}")
        rt = vc.get("rating", {})
        if rt.get("present"):
            lines.append(f"- rating: min=`{rt.get('min')}`, max=`{rt.get('max')}`, "
                         f"mean=`{rt.get('mean')}`, median=`{rt.get('median')}`, "
                         f"hors intervalle=`{rt.get('out_of_range_count')}` ({rt.get('out_of_range_pct')}%), "
                         f"ok=`{rt.get('ok')}`")
        ts = vc.get("timestamp", {})
        if ts.get("present"):
            lines.append(f"- timestamp: dtype=`{ts.get('dtype')}`, "
                         f"min=`{ts.get('min_date', 'N/A')}`, max=`{ts.get('max_date', 'N/A')}`, "
                         f"non convertibles=`{ts.get('unconvertible_count', 0)}`, "
                         f"ok=`{ts.get('ok')}`")
            for w in ts.get("warnings", []):
                lines.append(f"  - ⚠ {w}")
        lines.append("")

    # ---------------------------------------------------------------
    # D) Qualité de jointure interactions ↔ metadata
    # ---------------------------------------------------------------
    lines.append("## D. Qualité de jointure via `parent_asin`")
    lines.append("")
    join_metrics = result.get("join_metrics", {})
    if not join_metrics:
        lines.append("- (pas encore calculé)")
        lines.append("")
    else:
        for name, jm in join_metrics.items():
            lines.append(f"### {name}")
            lines.append(f"- nb_parent_asin_communs: `{jm.get('nb_parent_asin_communs')}`")
            lines.append(
                "- nb_interactions_jointes / nb_interactions_totales: "
                f"`{jm.get('nb_interactions_jointes')} / {jm.get('nb_interactions_totales')}`"
            )
            lines.append(f"- ratio_interactions_jointes: `{jm.get('ratio_interactions_jointes')}`")
            lines.append(
                "- nb_items_avec_meta / nb_items_totaux: "
                f"`{jm.get('nb_items_avec_meta')} / {jm.get('nb_items_totaux')}`"
            )
            lines.append(f"- ratio_items_avec_meta: `{jm.get('ratio_items_avec_meta')}`")
            lines.append(f"- interactions_non_jointes_si_inner_join: `{jm.get('interactions_non_jointes_si_inner_join')}`")
            lines.append(f"- items_sans_meta: `{jm.get('items_sans_meta')}`")
            lines.append("")

    # ---------------------------------------------------------------
    # E) Attributs exploitables
    # ---------------------------------------------------------------
    lines.append("## E. Attributs exploitables")
    lines.append("")
    exploitable = result.get("exploitable_columns", {})
    if not exploitable:
        lines.append("- (pas encore défini)")
        lines.append("")
    else:
        for name, ex in exploitable.items():
            lines.append(f"### {name}")
            lines.append(f"- interactions_kept: `{ex.get('interactions_kept', [])}`")
            lines.append(f"- metadata_text_kept: `{ex.get('metadata_text_kept', [])}`")
            lines.append(f"- metadata_struct_kept: `{ex.get('metadata_struct_kept', [])}`")
            lines.append(f"- ignored_interactions_cols: `{ex.get('ignored_interactions_cols', [])}`")
            lines.append(f"- ignored_metadata_cols: `{ex.get('ignored_metadata_cols', [])}`")
            lines.append("")

    # ---------------------------------------------------------------
    # F) Valeurs manquantes + stratégie
    # ---------------------------------------------------------------
    lines.append("## F. Valeurs manquantes et stratégie")
    lines.append("")
    missingness = result.get("missingness", {})
    for name, miss_data in missingness.items():
        lines.append(f"### {name}")
        lines.append("")
        if isinstance(miss_data, dict) and ("on_meta_global" in miss_data or "on_interactions_raw" in miss_data):
            scope_order = ["on_interactions_raw", "on_meta_global", "on_joined_subset"]
            scope_labels = {
                "on_interactions_raw": "Interactions brutes",
                "on_meta_global": "Métadonnées globales",
                "on_joined_subset": "Sous-ensemble joint",
            }
            for scope in scope_order:
                rows = miss_data.get(scope)
                if not rows:
                    continue
                lines.append(f"#### {scope_labels.get(scope, scope)}")
                lines.append("")
                lines.append("| colonne | type | % NaN | % vide | % effectif | stratégie | justification |")
                lines.append("|---------|------|-------|--------|------------|-----------|---------------|")
                for r in rows:
                    col = r.get("column", "?")
                    ctype = r.get("column_type", "—")
                    nan_pct = r.get("missing_pct")
                    empty_pct = r.get("empty_pct")
                    eff_pct = r.get("effective_missing_pct", nan_pct)
                    strat = r.get("strategy", "—")
                    justif = r.get("justification", "—")
                    lines.append(
                        f"| {col} | {ctype} | "
                        f"{nan_pct if nan_pct is not None else 'N/A'}% | "
                        f"{empty_pct if empty_pct is not None else 'N/A'}% | "
                        f"{eff_pct if eff_pct is not None else 'N/A'}% | "
                        f"{strat} | {justif} |"
                    )
                lines.append("")
        else:
            lines.append("| colonne | % NaN | stratégie |")
            lines.append("|---------|-------|-----------|")
            for r in miss_data:
                lines.append(
                    f"| {r['column']} | {r.get('missing_pct')}% | {r.get('strategy')} |"
                )
            lines.append("")
    lines.append("")

    # ---------------------------------------------------------------
    # F2) Nettoyage appliqué (avant/après)
    # ---------------------------------------------------------------
    lines.append("## F2. Qualité des champs textuels")
    lines.append("")
    tq = result.get("text_quality", {})
    for name, cols_report in tq.items():
        lines.append(f"### {name}")
        for r in cols_report:
            if not r.get("present"):
                lines.append(f"- {r.get('column')}: absent")
                continue
            lines.append(
                f"- {r.get('column')}: "
                f"avg_len=`{r.get('avg_length')}`, "
                f"median_len=`{r.get('median_length')}`, "
                f"vides=`{r.get('empty_or_blank_count')}` ({r.get('empty_or_blank_pct')}%), "
                f"HTML=`{r.get('html_noise_count')}` ({r.get('html_noise_pct')}%)"
            )
        lines.append("")

    # ---------------------------------------------------------------
    # F3) Nettoyage appliqué (avant/après)
    # ---------------------------------------------------------------
    lines.append("## F3. Nettoyage appliqué (avant / après)")
    lines.append("")
    cleaning = result.get("cleaning_reports", {})
    for name, rpt in cleaning.items():
        lines.append(f"### {name}")
        bef = rpt.get("before", {})
        aft = rpt.get("after", {})
        lines.append("")
        lines.append("| métrique | avant | après | delta |")
        lines.append("|----------|-------|-------|-------|")
        for key, label in [("n_rows", "lignes"), ("n_items", "items"), ("n_users", "users")]:
            b = bef.get(key)
            a = aft.get(key)
            delta = (b - a) if b is not None and a is not None else "N/A"
            lines.append(f"| {label} | {b} | {a} | −{delta} |")
        lines.append("")
        reasons = rpt.get("dropped_reason", {})
        if reasons:
            lines.append("**Raisons de suppression :**")
            for reason, count in reasons.items():
                lines.append(f"- `{reason}`: {count} lignes")
            lines.append("")

    # ---------------------------------------------------------------
    # F4) Vérifications post-nettoyage
    # ---------------------------------------------------------------
    lines.append("## F4. Vérifications post-nettoyage")
    lines.append("")
    post_checks = result.get("post_cleaning_checks", {})
    for name, checks in post_checks.items():
        lines.append(f"### {name}")
        lines.append("")

        res_dups = checks.get("residual_pair_duplicates", "N/A")
        res_dups_ok = checks.get("residual_pair_duplicates_ok", None)
        status = "OK" if res_dups_ok else "ALERTE"
        lines.append(f"- Doublons résiduels `(user_id, parent_asin)`: **{res_dups}** — {status}")

        rating_post = checks.get("rating_post_clean", {})
        if rating_post.get("present"):
            oor = rating_post.get("out_of_range_count", 0)
            r_ok = "OK" if rating_post.get("ok") else f"ALERTE ({oor} hors [1,5])"
            lines.append(f"- Distribution rating post-nettoyage: min={rating_post.get('min')}, "
                         f"max={rating_post.get('max')}, mean={rating_post.get('mean')} — {r_ok}")

        integrity = checks.get("parent_asin_integrity", {})
        lost = integrity.get("items_lost", 0)
        i_ok = "OK" if integrity.get("ok") else f"ALERTE ({lost} items perdus, {integrity.get('items_lost_pct')}%)"
        lines.append(f"- Intégrité parent_asin: {integrity.get('items_before')} → "
                     f"{integrity.get('items_after')} items — {i_ok}")

        key_na = checks.get("residual_key_nan", {})
        key_ok = "OK" if checks.get("residual_key_nan_ok") else f"ALERTE {key_na}"
        lines.append(f"- NaN résiduel sur clés: {key_na} — {key_ok}")
        lines.append("")

    # ---------------------------------------------------------------
    # G) Datasets finaux produits
    # ---------------------------------------------------------------

    lines.append("## G. Jeux de données finaux")
    lines.append("")
    finals = result.get("final_datasets", {})
    if not finals:
        lines.append("- (pas encore matérialisé)")
        lines.append("")
    else:
        for name, fd in finals.items():
            lines.append(f"### {name}")
            lines.append(f"- path: `{fd.get('path')}`")
            lines.append(f"- rows: `{fd.get('n_rows')}`")
            lines.append(f"- cols: `{fd.get('n_cols')}`")
            lines.append("")


    # ---------------------------------------------------------------
    # H) Usage des colonnes par tâche
    # ---------------------------------------------------------------

    lines.append("## H. Usage des colonnes par tâche")
    lines.append("")
    col_purpose = result.get("column_purpose", {})
    cr = col_purpose.get("content_representation", {})
    lf = col_purpose.get("learning_features", {})
    if cr:
        lines.append("### Représentation de contenu (Tâches 0-2)")
        for col, desc in cr.items():
            lines.append(f"- `{col}`: {desc}")
        lines.append("")
    if lf:
        lines.append("### Variables explicatives (Tâche 3)")
        for col, desc in lf.items():
            lines.append(f"- `{col}`: {desc}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": str(json_path), "md": str(md_path)}





def save_joined_dataset(
    df: pd.DataFrame,
    name: str,
    out_dir: str = "data/joining",
    verbose: bool = True,
) -> str:
    if verbose:
        print(
            "\nsave_joined_dataset\n"
            f"out_dir: {out_dir}\n",
            f"len(df.columns): {len(df.columns)}\n"
            f"df.columns: {df.columns}\n"
            # f"df: {df}\n"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_joined.parquet"
    if verbose:
        print(f"\npath: {path}")

    for col in df.columns:
        types = df[col].dropna().apply(type).value_counts()
        if len(types) > 1 and verbose:
            print(f"\nMIXED TYPES in {col}:")
            print(f"\n{types}")

    df.to_parquet(path, index=False, engine='pyarrow')

    return str(path)





def run_all(
    verbose: bool = True,
    include_optional_raw: bool = False,
    export_artifacts: bool = True,
    materialize_joined: bool = True,
) -> Dict[str, Any]:
    """
    Base pipeline (Tasks 1-2 ready):
    - build manifest
    - validate paths
    - collect source docs
    - export source diagnostics

    Task 3+ hooks:
    - schema/key checks
    - duplicate diagnostics
    - join metrics
    """
    manifest = get_manifest(include_optional_raw=include_optional_raw)
    path_status = validate_manifest_paths(manifest)
    source_infos = collect_source_documentation(manifest, verbose=verbose)
    
    meta_source_info = next(s for s in source_infos if s.role == "metadata")
    source_meta_all_cols = meta_source_info.columns

    schema_checks: Dict[str, Any] = {}
    duplicate_checks: Dict[str, Any] = {}
    validation_checks: Dict[str, Any] = {}
    text_quality_checks: Dict[str, Any] = {}
    join_metrics = {}
    exploitable_cols = {}
    missingness = {}
    final_datasets = {}
    cleaning_reports = {}
    post_clean_checks = {}

    # metadata
    meta_cfg = manifest["metadata"]
    meta_cols = ["parent_asin"] + METADATA_TEXT_COLS + METADATA_STRUCT_COLS
    if verbose:
        print(f"\nmeta_cols: {meta_cols}")
    meta_df = load_target_df(meta_cfg, meta_cols, verbose=verbose)
    meta_df["parent_asin"] = meta_df["parent_asin"].astype("string")
    meta_key_set = set(meta_df["parent_asin"].dropna().unique().tolist())
    duplicate_checks["metadata"] = check_duplicates(meta_df, role="metadata")

    if path_status.get("metadata", False):
        schema_checks["metadata"] = run_schema_key_checks_for_target("metadata", meta_cfg, meta_df)
    else:
        schema_checks["metadata"] = {
            "target": "metadata",
            "ok": False,
            "warnings": ["Chemin(s) manquant(s), vérification Task 3 ignorée"],
        }
    inter_cols = INTERACTION_MIN_COLS   
    if verbose:
        print(f"inter_cols: {inter_cols}")
    for name, cfg in manifest.items():
        if cfg["role"] != "interactions":
            continue

        if not path_status.get(name, False):
            schema_checks[name] = {
                "target": name,
                "ok": False,
                "warnings": ["Chemin(s) manquant(s), vérification Task 3 ignorée"],
            }
            continue

        inter_df = load_target_df(cfg, columns=inter_cols, verbose=verbose)

        duplicate_checks[name] = check_duplicates(inter_df, role="interactions")

        miss_inter_raw = missingness_report(inter_df, inter_cols)
        miss_inter_raw = attach_missingness_strategy(miss_inter_raw)

        validation_checks[name] = {
            "rating": validate_rating_range(inter_df),
            "timestamp": validate_timestamp(inter_df),
        }

        schema_checks[name] = run_schema_key_checks_for_target(name, cfg, inter_df)

        # 1) join quality
        jm = compute_join_quality_metrics(inter_df, meta_key_set=meta_key_set, meta_total_count=len(meta_key_set))
        join_metrics[name] = jm

        # 2) exploitable columns
        inter_source_info = next(s for s in source_infos if s.name == name)
        source_inter_all_cols = inter_source_info.columns
        ex = select_exploitable_columns(
            inter_df, meta_df,
            source_inter_cols=source_inter_all_cols,
            source_meta_cols=source_meta_all_cols,
        )        
        exploitable_cols[name] = ex

        # 3) missing values report + strategy on Raw Parquet
        cols_to_check = ["parent_asin"] + ex["metadata_text_kept"] + ex["metadata_struct_kept"]

        if "meta_missingness_full" not in locals():
            meta_missingness_full = missingness_report(meta_df, list(meta_df.columns))
        if isinstance(meta_missingness_full, pd.DataFrame):
            available_cols = [c for c in cols_to_check if c in meta_missingness_full.index]
            miss = meta_missingness_full.loc[available_cols]
        else:
            miss = missingness_report(meta_df, cols_to_check)

        miss = attach_missingness_strategy(miss)
        missingness[name] = miss

        # 4) final joined dataset
        meta_keep = ex["metadata_text_kept"] + ex["metadata_struct_kept"]
        if verbose:
            print(f"meta_keep: {meta_keep}")
        joined_df = build_joined_dataset(inter_df, meta_df, meta_keep_cols=meta_keep, verbose=verbose)

        # 4b) text quality (avant nettoyage, sur données normalisées)
        text_cols_to_check = [c for c in ["title", "subtitle", "description",
            "categories", "features", "author_name",
            "details_publisher", "details_language"] if c in joined_df.columns]
        text_quality_checks[name] = text_quality_report(joined_df, text_cols_to_check)
        
        # 5) Nettoyage : suppression NaN clés + dédoublonnage interactions
        items_before_clean = set(joined_df["parent_asin"].dropna().unique())
        joined_df, cleaning_rpt = clean_joined_dataset(joined_df, verbose=verbose)
        cleaning_reports[name] = cleaning_rpt

        # 6) Vérifications post-nettoyage
        post_clean_checks[name] = post_cleaning_checks(joined_df, items_before_clean)

        if verbose:
            print(
                # f"\njoined_df: {joined_df}"
                f"\nlen(joined_df.columns).: {len(joined_df.columns)}"
                f"\njoined_df.columns: {joined_df.columns}"
            )

        # 7) Missingness sur dataset joint nettoyé
        miss_joined = missingness_report(joined_df, list(joined_df.columns))
        miss_joined = attach_missingness_strategy(miss_joined)
        missingness[name] = {
            "on_interactions_raw": miss_inter_raw,
            "on_meta_global": miss,
            "on_joined_subset": miss_joined,
        }

        out_path = None
        if materialize_joined:
            out_path = save_joined_dataset(joined_df, name=name + "_clean", out_dir="data/joining", verbose=verbose)
        final_datasets[name] = {
            "path": out_path,
            "n_rows": len(joined_df),
            "n_cols": len(joined_df.columns),
        }

        del inter_df, joined_df
        gc.collect()


    result: Dict[str, Any] = {
        "manifest": manifest,
        "path_status": path_status,
        "sources": [asdict(s) for s in source_infos],
        "schema_checks": schema_checks,
        "duplicate_checks": duplicate_checks,
        "validation_checks": validation_checks,
        "join_metrics": join_metrics,
        "exploitable_columns": exploitable_cols,
        "missingness": missingness,
        "text_quality": text_quality_checks,
        "cleaning_reports": cleaning_reports,
        "post_cleaning_checks": post_clean_checks,
        "final_datasets": final_datasets,
        "column_purpose": {
            "content_representation": CONTENT_REPRESENTATION_COLS,
            "learning_features": LEARNING_FEATURE_COLS,
        },
    }
    result["p1_reuse_note"] = build_p1_reuse_note(manifest, result["sources"])

    if export_artifacts:
        result["artifacts"] = save_diagnostics(result, out_dir="results/joining")

    return result





def _fmt_pct(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return str(x)





def _fmt_num(x: Any) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)





def cli_print_results(
    result: Dict[str, Any] | None = None,
    t_start: float | None = None,
    verbose: bool = True,
) -> None:

    if result is None:
        result = run_all()
    if t_start is None:
        t_start = time.time()

    if not verbose:
        return

    print("\n" + "=" * 86)
    print("TÂCHE 0 (P2) — PRÉPARATION DES DONNÉES / CADRE EXPÉRIMENTAL")
    print("=" * 86)

    # ------------------------------------------------------------------
    # 1) Réutilisation du sous-ensemble P1
    # ------------------------------------------------------------------
    p1_note = result.get("p1_reuse_note", {})
    print("\n[1] Réutilisation du sous-ensemble de travail")
    print(f"- Note méthodologique: {p1_note.get('statement', 'N/A')}")
    print("- Sources interactions retenues:")
    for s in result.get("sources", []):
        if s.get("role") == "interactions":
            print(
                f"  • {s['name']} | rows={_fmt_num(s.get('n_rows'))} "
                f"| cols={_fmt_num(s.get('n_cols'))} | exists={s.get('exists')}"
            )
            for p in s.get("paths", []):
                print(f"      path: {p}")

    print("\n- Source metadata:")
    meta_src = next((s for s in result.get("sources", []) if s.get("role") == "metadata"), None)
    if meta_src:
        print(
            f"  • {meta_src['name']} | rows={_fmt_num(meta_src.get('n_rows'))} "
            f"| cols={_fmt_num(meta_src.get('n_cols'))} | exists={meta_src.get('exists')}"
        )
        for p in meta_src.get("paths", []):
            print(f"      path: {p}")

    # ------------------------------------------------------------------
    # 2) Clés manquantes + checks schéma
    # ------------------------------------------------------------------
    print("\n[2] Vérification schéma et clés (`parent_asin`)")
    schema_checks = result.get("schema_checks", {})
    for name, chk in schema_checks.items():
        if chk.get("role") not in {"interactions", "metadata"}:
            continue
        mk = chk.get("missing_keys", {})
        req = chk.get("required_columns", {})
        print(f"  • {name}")
        print(f"      ok: {chk.get('ok')}")
        print(f"      colonnes_manquantes: {req.get('missing_required', [])}")
        print(f"      missing_parent_asin_count: {_fmt_num(mk.get('missing_parent_asin_count'))}")
        print(f"      missing_parent_asin_pct: {mk.get('missing_parent_asin_pct')}%")
        print(f"      coercion_warning: {chk.get('coercion', {}).get('warning')}")

    # ------------------------------------------------------------------
    # 2b) Doublons
    # ------------------------------------------------------------------
    print("\n[2b] Détection de doublons")
    dup_checks = result.get("duplicate_checks", {})
    for name, dc in dup_checks.items():
        print(f"  • {name} ({_fmt_num(dc.get('n_rows'))} rows)")
        print(f"      doublons exacts: {_fmt_num(dc.get('exact_duplicates'))} ({dc.get('exact_duplicates_pct')}%)")
        if "user_item_duplicates" in dc:
            print(f"      doublons (user, item): {_fmt_num(dc.get('user_item_duplicates'))} ({dc.get('user_item_duplicates_pct')}%)")
        if "parent_asin_duplicates" in dc:
            print(f"      doublons parent_asin: {_fmt_num(dc.get('parent_asin_duplicates'))} ({dc.get('parent_asin_duplicates_pct')}%)")

    # ------------------------------------------------------------------
    # 2c) Validation rating + timestamp
    # ------------------------------------------------------------------
    print("\n[2c] Validation rating et timestamp")
    val_checks = result.get("validation_checks", {})
    for name, vc in val_checks.items():
        print(f"  • {name}")
        rt = vc.get("rating", {})
        if rt.get("present"):
            status = "OK" if rt.get("ok") else "PROBLÈME"
            print(f"      rating: [{rt.get('min')} – {rt.get('max')}], "
                  f"mean={rt.get('mean')}, hors [1,5]={_fmt_num(rt.get('out_of_range_count'))} → {status}")
        ts = vc.get("timestamp", {})
        if ts.get("present"):
            status = "OK" if ts.get("ok") else "PROBLÈME"
            print(f"      timestamp: {ts.get('min_date', 'N/A')} → {ts.get('max_date', 'N/A')}, "
                  f"dtype={ts.get('dtype')} → {status}")
            for w in ts.get("warnings", []):
                print(f"        ⚠ {w}")

    # ------------------------------------------------------------------
    # 3) Qualité de jointure interactions ↔ metadata
    # ------------------------------------------------------------------
    print("\n[3] Qualité de jointure via `parent_asin`")
    join_metrics = result.get("join_metrics", {})
    if not join_metrics:
        print("  (pas encore calculé)")
    else:
        for name, jm in join_metrics.items():
            print(f"  • {name}")
            print(f"      nb_parent_asin_communs: {_fmt_num(jm.get('nb_parent_asin_communs'))}")
            print(
                "      interactions jointes / totales: "
                f"{_fmt_num(jm.get('nb_interactions_jointes'))} / {_fmt_num(jm.get('nb_interactions_totales'))} "
                f"({_fmt_pct(jm.get('ratio_interactions_jointes'))})"
            )
            print(
                "      items avec meta / totaux: "
                f"{_fmt_num(jm.get('nb_items_avec_meta'))} / {_fmt_num(jm.get('nb_items_totaux'))} "
                f"({_fmt_pct(jm.get('ratio_items_avec_meta'))})"
            )
            print(f"      interactions perdues (inner join): {_fmt_num(jm.get('interactions_non_jointes_si_inner_join'))}")
            print(f"      items sans meta: {_fmt_num(jm.get('items_sans_meta'))}")
            if "nb_meta_orphelines" in jm:
                print(
                    f"      meta orphelines: {_fmt_num(jm.get('nb_meta_orphelines'))} / "
                    f"{_fmt_num(jm.get('nb_meta_total'))} "
                    f"({_fmt_pct(1.0 - jm.get('ratio_meta_utilisees', 0))})"
                )
            if "interpretation" in jm:
                print(f"      interprétation: {jm['interpretation']}")

    # ------------------------------------------------------------------
    # 4) Attributs exploitables + justifications
    # ------------------------------------------------------------------
    print("\n[4] Attributs exploitables")
    exploitable = result.get("exploitable_columns", {})
    for name in sorted(exploitable.keys()):
        ex = exploitable[name]
        print(f"  • {name}")
        print(f"      interactions_kept: {ex.get('interactions_kept', [])}")
        print(f"      metadata_scalar:   {ex.get('metadata_scalar', [])}")
        print(f"      metadata_list:     {ex.get('metadata_list', [])}")
        print(f"      metadata_nested:   {ex.get('metadata_nested', [])}")
        print(f"      metadata_struct:   {ex.get('metadata_struct_kept', [])}")

        ignored_inter = ex.get("ignored_interactions_cols", [])
        ignored_meta = ex.get("ignored_metadata_cols", [])
        if ignored_inter:
            print(f"      ignorées (interactions source): {ignored_inter}")
        if ignored_meta:
            print(f"      ignorées (metadata source):     {ignored_meta}")

        justifications = ex.get("justifications", {})
        if justifications:
            print("      justifications (colonnes retenues):")
            for col, reason in justifications.items():
                print(f"        + {col}: {reason}")

        exclusions = ex.get("exclusion_reasons", {})
        if exclusions:
            print("      raisons d'exclusion:")
            for col, reason in exclusions.items():
                print(f"        - {col}: {reason}")

    # ------------------------------------------------------------------
    # 5) Valeurs manquantes (interactions brutes + meta global + joint)
    # ------------------------------------------------------------------
    print("\n[5] Valeurs manquantes et stratégie")
    miss = result.get("missingness", {})
    for name in sorted(miss.keys()):
        miss_data = miss[name]
        print(f"  • {name}")
        if isinstance(miss_data, dict) and ("on_meta_global" in miss_data or "on_interactions_raw" in miss_data):
            scope_order = ["on_interactions_raw", "on_meta_global", "on_joined_subset"]
            scope_labels = {
                "on_interactions_raw": "Interactions brutes",
                "on_meta_global": "Meta global",
                "on_joined_subset": "Sous-ensemble joint",
            }
            for scope in scope_order:
                rows = miss_data.get(scope)
                if not rows:
                    continue
                print(f"      [{scope_labels.get(scope, scope)}]")
                for r in rows:
                    eff = r.get("effective_missing_pct")
                    eff_str = f", effectif={eff}%" if eff is not None else ""
                    empty = r.get("empty_count")
                    empty_str = f", vides={_fmt_num(empty)}" if empty is not None else ""
                    ctype = r.get("column_type", "")
                    type_str = f" [{ctype}]" if ctype else ""
                    justif = r.get("justification", "")
                    justif_str = f" — {justif}" if justif and justif != "—" else ""
                    print(
                        f"        - {r.get('column')}{type_str}: "
                        f"NaN={r.get('missing_pct')}%{empty_str}{eff_str} "
                        f"| {r.get('strategy')}{justif_str}"
                    )
        else:
            for r in miss_data:
                print(
                    f"        - {r.get('column')}: "
                    f"{r.get('missing_pct')}% | {r.get('strategy')}"
                )

    # ------------------------------------------------------------------
    # 5b) Qualité textuelle
    # ------------------------------------------------------------------
    print("\n[5b] Qualité des champs textuels")
    tq = result.get("text_quality", {})
    for name, cols_report in tq.items():
        print(f"  • {name}")
        for r in cols_report:
            if not r.get("present"):
                continue
            html_note = f", HTML={r.get('html_noise_count')}" if r.get("html_noise_count", 0) > 0 else ""
            print(f"      {r.get('column')}: avg_len={r.get('avg_length')}, "
                  f"median_len={r.get('median_length')}, "
                  f"vides={r.get('empty_or_blank_count')}{html_note}")

    # ------------------------------------------------------------------
    # 5c) Nettoyage appliqué
    # ------------------------------------------------------------------
    print("\n[5c] Nettoyage appliqué (avant / après)")
    cleaning = result.get("cleaning_reports", {})
    for name, rpt in cleaning.items():
        bef = rpt.get("before", {})
        aft = rpt.get("after", {})
        print(f"  • {name}")
        print(f"      lignes: {bef.get('n_rows')} → {aft.get('n_rows')} "
              f"(−{rpt.get('dropped_rows', 0)})")
        print(f"      items:  {bef.get('n_items')} → {aft.get('n_items')}")
        print(f"      users:  {bef.get('n_users')} → {aft.get('n_users')}")
        reasons = rpt.get("dropped_reason", {})
        if reasons:
            for reason, count in reasons.items():
                print(f"        → {reason}: {count}")

    # ------------------------------------------------------------------
    # 5d) Vérifications post-nettoyage
    # ------------------------------------------------------------------
    print("\n[5d] Vérifications post-nettoyage")
    post_checks = result.get("post_cleaning_checks", {})
    for name, checks in post_checks.items():
        print(f"  • {name}")
        res_dups = checks.get("residual_pair_duplicates", "N/A")
        ok_tag = "OK" if checks.get("residual_pair_duplicates_ok") else "ALERTE"
        print(f"      doublons résiduels (user,item): {res_dups} — {ok_tag}")

        rating_post = checks.get("rating_post_clean", {})
        if rating_post.get("present"):
            r_ok = "OK" if rating_post.get("ok") else "ALERTE"
            print(f"      rating: [{rating_post.get('min')}, {rating_post.get('max')}] "
                  f"mean={rating_post.get('mean')} — {r_ok}")

        integrity = checks.get("parent_asin_integrity", {})
        i_ok = "OK" if integrity.get("ok") else "ALERTE"
        print(f"      parent_asin: {integrity.get('items_before')} → "
              f"{integrity.get('items_after')} ({i_ok})")

        key_ok = "OK" if checks.get("residual_key_nan_ok") else "ALERTE"
        print(f"      NaN clés résiduels: {checks.get('residual_key_nan')} — {key_ok}")

    # ------------------------------------------------------------------
    # 6) Jeux finaux produits
    # ------------------------------------------------------------------
    print("\n[6] Jeux finaux cohérents produits")
    finals = result.get("final_datasets", {})
    if not finals:
        print("  (pas encore matérialisé)")
    else:
        for name, fd in finals.items():
            print(
                f"  • {name}: path={fd.get('path')} | "
                f"rows={_fmt_num(fd.get('n_rows'))} | cols={_fmt_num(fd.get('n_cols'))}"
            )

    # ------------------------------------------------------------------
    # 7) Usage des colonnes par tâche
    # ------------------------------------------------------------------
    print("\n[7] Mapping colonnes → tâches")
    col_purpose = result.get("column_purpose", {})
    cr = col_purpose.get("content_representation", {})
    lf = col_purpose.get("learning_features", {})
    if cr:
        print("  Représentation de contenu (Tâches 0-2):")
        for col, desc in cr.items():
            print(f"    • {col}: {desc}")
    if lf:
        print("  Variables explicatives (Tâche 3):")
        for col, desc in lf.items():
            print(f"    • {col}: {desc}")

    # Artifacts
    artifacts = result.get("artifacts", {})
    print("\n[Artifacts]")
    print(f"- JSON: {artifacts.get('json', 'N/A')}")
    print(f"- MD:   {artifacts.get('md', 'N/A')}")
    elapsed = time.time() - t_start if t_start else 0.0
    print("\n" + "=" * 86)
    print(f"FIN — Résumé prêt pour notebook/rapport  ||  Pipeline complet en {elapsed:.1f}s")
    print("=" * 86)






def cli_print_md_results(verbose: bool = True) -> None:
    if verbose:
        print(Path("results/joining/joining_diagnostics.md").read_text(encoding="utf-8"))






def main() -> None:
    t_start = time.time()
    result = run_all(
        verbose=True,
        include_optional_raw=False,   
        export_artifacts=True,
        materialize_joined=True
    )

    if result:
        cli_print_results(result, t_start)




if __name__ == "__main__":
    main()