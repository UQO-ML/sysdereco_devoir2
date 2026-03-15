from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import gc
import json
import os
import time

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




INTERACTION_MIN_COLS = ["user_id", "parent_asin", "rating", "timestamp"]

METADATA_SCALAR_COLS = ["title"]
METADATA_LIST_COLS = ["features", "description", "categories"]
METADATA_STRUCT_COLS = ["average_rating", "rating_number", "price"]
METADATA_NESTED_COLS = ["author", "details"]

METADATA_TEXT_COLS = METADATA_SCALAR_COLS + METADATA_LIST_COLS + METADATA_NESTED_COLS

REQUIRED_INTERACTION_COLS = {"user_id", "parent_asin", "rating", "timestamp"}
REQUIRED_METADATA_COLS = {"parent_asin"}



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
            )

    if kind == "union":
        train_df = pd.read_parquet(paths[0], columns=columns, 
            )
        test_df = pd.read_parquet(paths[1], columns=columns, 
            )
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


def missingness_report(df: pd.DataFrame, cols: List[str]) -> List[Dict[str, Any]]:
    out = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            out.append({"column": col, "missing_count": None, "missing_pct": None, "strategy": "absent"})
            continue
        m = int(df[col].isna().sum())
        pct = (m / n * 100.0) if n else 0.0
        out.append({
            "column": col,
            "missing_count": m,
            "missing_pct": round(pct, 4),
            "strategy": None,  # rempli ensuite
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
        else:
            r["strategy"] = "au cas par cas / hors périmètre"
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
    meta_df: pd.DataFrame
) -> Dict[str, Any]:

    inter_available = [c for c in INTERACTION_MIN_COLS if c in inter_df.columns]
    meta_scalar = [c for c in METADATA_SCALAR_COLS if c in meta_df.columns]
    meta_list = [c for c in METADATA_LIST_COLS if c in meta_df.columns]
    meta_nested = [c for c in METADATA_NESTED_COLS if c in meta_df.columns]
    meta_struct = [c for c in METADATA_STRUCT_COLS if c in meta_df.columns]

    all_meta_kept = meta_scalar + meta_list + meta_nested + meta_struct

    return {
        "interactions_kept": inter_available,
        "metadata_text_kept": meta_scalar + meta_list + meta_nested,
        "metadata_struct_kept": meta_struct,
        "metadata_scalar": meta_scalar,
        "metadata_list": meta_list,
        "metadata_nested": meta_nested,
        "ignored_interactions_cols": [c for c in inter_df.columns if c not in inter_available],
        "ignored_metadata_cols": [c for c in meta_df.columns if c not in all_meta_kept + ["parent_asin"]],
    }




def compute_join_quality_metrics(inter_df: pd.DataFrame, meta_df: pd.DataFrame = None, meta_key_set: set[str] = None) -> Dict[str, Any]:
    inter_key = inter_df["parent_asin"].astype("string")
    inter_items = set(inter_key.dropna().unique().tolist())

    if meta_key_set is None:
        if meta_df is None:
            raise ValueError("Provide meta_df or meta_key_set")
        meta_key_set = set(meta_df["parent_asin"].astype("string").dropna().unique().tolist())

    common_items = inter_items.intersection(meta_key_set)
    matched_mask = inter_key.isin(meta_key_set)

    n_inter_total = len(inter_df)
    n_inter_joined = int(matched_mask.sum())
    n_items_total = len(inter_items)
    n_items_with_meta = len(common_items)

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
    }






def _flatten_struct_col(series: pd.Series, key: str) -> pd.Series:
    """Extract a single key from a struct/dict column, return as string."""
    def _extract(val):
        if isinstance(val, dict):
            return str(val.get(key, ""))
        return ""
    return series.apply(_extract)






def _join_list_col(series: pd.Series, sep: str = " | ") -> pd.Series:
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





def normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
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
        lines.append(f"- columns names: '{s.get('columns')}")
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
    if not missingness:
        lines.append("- (pas encore calculé)")
        lines.append("")
    else:
        for name, rows in missingness.items():
            lines.append(f"### {name}")
            if not rows:
                lines.append("- (aucune ligne)")
                lines.append("")
                continue
            for r in rows:
                lines.append(
                    f"- {r.get('column')}: missing_count=`{r.get('missing_count')}`, "
                    f"missing_pct=`{r.get('missing_pct')}`, strategy=`{r.get('strategy')}`"
                )
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

    df.to_parquet(path)

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
    
    schema_checks: Dict[str, Any] = {}
    join_metrics = {}
    exploitable_cols = {}
    missingness = {}
    final_datasets = {}

    # metadata
    meta_cfg = manifest["metadata"]
    meta_cols = ["parent_asin"] + METADATA_TEXT_COLS + METADATA_STRUCT_COLS
    if verbose:
        print(f"\nmeta_cols: {meta_cols}")
    meta_df = load_target_df(meta_cfg, meta_cols, verbose=verbose)
    meta_df["parent_asin"] = meta_df["parent_asin"].astype("string")
    meta_key_set = set(meta_df["parent_asin"].dropna().unique().tolist())

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

        schema_checks[name] = run_schema_key_checks_for_target(name, cfg, inter_df)

        # 1) join quality
        jm = compute_join_quality_metrics(inter_df, meta_key_set=meta_key_set)
        join_metrics[name] = jm

        # 2) exploitable columns
        ex = select_exploitable_columns(inter_df, meta_df)
        exploitable_cols[name] = ex

        # 3) missing values report + strategy
        cols_to_check = ["parent_asin"] + ex["metadata_text_kept"] + ex["metadata_struct_kept"]
        miss = missingness_report(meta_df, cols_to_check)
        miss = attach_missingness_strategy(miss)
        missingness[name] = miss

        # 4) final joined dataset
        meta_keep = ex["metadata_text_kept"] + ex["metadata_struct_kept"]
        if verbose:
            print(f"meta_keep: {meta_keep}")
        joined_df = build_joined_dataset(inter_df, meta_df, meta_keep_cols=meta_keep, verbose=verbose)
        if verbose:
            print(
                # f"\njoined_df: {joined_df}"
                f"\nlen(joined_df.columns).: {len(joined_df.columns)}"
                f"\njoined_df.columns: {joined_df.columns}"
            )
        out_path = None
        if materialize_joined:
            out_path = save_joined_dataset(joined_df, name=name, out_dir="data/joining", verbose=verbose)
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
        "join_metrics": join_metrics,
        "exploitable_columns": exploitable_cols,
        "missingness": missingness,
        "final_datasets": final_datasets,
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
                f"{s['columns']}"
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
                "      nb_interactions_jointes / nb_interactions_totales: "
                f"{_fmt_num(jm.get('nb_interactions_jointes'))} / {_fmt_num(jm.get('nb_interactions_totales'))} "
                f"({_fmt_pct(jm.get('ratio_interactions_jointes'))})"
            )
            print(
                "      nb_items_avec_meta / nb_items_totaux: "
                f"{_fmt_num(jm.get('nb_items_avec_meta'))} / {_fmt_num(jm.get('nb_items_totaux'))} "
                f"({_fmt_pct(jm.get('ratio_items_avec_meta'))})"
            )
            print(f"      interactions_non_jointes_si_inner_join: {_fmt_num(jm.get('interactions_non_jointes_si_inner_join'))}")
            print(f"      items_sans_meta: {_fmt_num(jm.get('items_sans_meta'))}")

    # ------------------------------------------------------------------
    # 4) Attributs exploitables + manquants
    # ------------------------------------------------------------------
    print("\n[4] Attributs exploitables et valeurs manquantes")
    exploitable = result.get("exploitable_columns", {})
    miss = result.get("missingness", {})
    for name in sorted(exploitable.keys()):
        ex = exploitable[name]
        print(f"  • {name}")
        print(f"      interactions_kept: {ex.get('interactions_kept', [])}")
        print(f"      metadata_text_kept: {ex.get('metadata_text_kept', [])}")
        print(f"      metadata_struct_kept: {ex.get('metadata_struct_kept', [])}")
        rows = miss.get(name, [])
        if rows:
            print("      missingness (colonne -> % -> stratégie):")
            for r in rows:
                print(f"        - {r.get('column')}: {r.get('missing_pct')}% | {r.get('strategy')}")

    # ------------------------------------------------------------------
    # 5) Jeux finaux produits
    # ------------------------------------------------------------------
    print("\n[5] Jeux finaux cohérents produits")
    finals = result.get("final_datasets", {})
    if not finals:
        print("  (pas encore matérialisé)")
    else:
        for name, fd in finals.items():
            print(
                f"  • {name}: path={fd.get('path')} | "
                f"rows={_fmt_num(fd.get('n_rows'))} | cols={_fmt_num(fd.get('n_cols'))}"
            )

    # Artifacts
    artifacts = result.get("artifacts", {})
    print("\n[Artifacts]")
    print(f"- JSON: {artifacts.get('json', 'N/A')}")
    print(f"- MD:   {artifacts.get('md', 'N/A')}")
    if t_start:
        elapsed = time.time() - t_start
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