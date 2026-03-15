from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import json
import os

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




REQUIRED_INTERACTION_COLS = {"user_id", "parent_asin", "timestamp"}
REQUIRED_METADATA_COLS = {"parent_asin"}

INTERACTION_MIN_COLS = ["user_id", "parent_asin", "rating", "timestamp"]
METADATA_TEXT_COLS = ["title", "description", "categories"]
METADATA_STRUCT_COLS = ["average_rating", "rating_number", "price"]  # optionnels

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



def load_target_df(cfg: Dict[str, Any]) -> pd.DataFrame:
    kind = cfg["kind"]
    paths = cfg["paths"]

    if kind == "single":
        return pd.read_parquet(paths[0])

    if kind == "union":
        train_df = pd.read_parquet(paths[0])
        test_df = pd.read_parquet(paths[1])
        return pd.concat([train_df, test_df], ignore_index=True)

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


def attach_missingness_strategy(report_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    text_cols = set(METADATA_TEXT_COLS)
    numeric_cols = set(METADATA_STRUCT_COLS)
    key_cols = {"parent_asin", "user_id", "timestamp"}

    for r in report_rows:
        col = r["column"]
        if r["missing_pct"] is None:
            r["strategy"] = "colonne absente"
        elif col in key_cols:
            r["strategy"] = "supprimer lignes incomplètes (clé obligatoire)"
        elif col in text_cols:
            r["strategy"] = "remplacer NaN par chaîne vide"
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
    cfg: Dict[str, Any]
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

    df = load_target_df(cfg)

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


def select_exploitable_columns(inter_df: pd.DataFrame, meta_df: pd.DataFrame) -> Dict[str, Any]:
    inter_available = [c for c in INTERACTION_MIN_COLS if c in inter_df.columns]
    meta_text_available = [c for c in METADATA_TEXT_COLS if c in meta_df.columns]
    meta_struct_available = [c for c in METADATA_STRUCT_COLS if c in meta_df.columns]

    return {
        "interactions_kept": inter_available,
        "metadata_text_kept": meta_text_available,
        "metadata_struct_kept": meta_struct_available,
        "ignored_interactions_cols": [c for c in inter_df.columns if c not in inter_available],
        "ignored_metadata_cols": [c for c in meta_df.columns if c not in (meta_text_available + meta_struct_available + ["parent_asin"])],
    }




def compute_join_quality_metrics(
    inter_df: pd.DataFrame, 
    meta_df: pd.DataFrame
) -> Dict[str, Any]:

    # Harmonisation clé
    inter_key = inter_df["parent_asin"].astype("string")
    meta_key = meta_df["parent_asin"].astype("string")

    inter_non_null = inter_key.dropna()
    meta_non_null = meta_key.dropna()

    inter_items = set(inter_non_null.unique().tolist())
    meta_items = set(meta_non_null.unique().tolist())
    common_items = inter_items.intersection(meta_items)

    # Couverture interactions
    matched_mask = inter_key.isin(meta_items)
    n_inter_total = len(inter_df)
    n_inter_joined = int(matched_mask.sum())

    # Couverture items
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


def build_joined_dataset(
    inter_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    meta_keep_cols: List[str],
) -> pd.DataFrame:
    # clés string
    inter_df = inter_df.copy()
    meta_df = meta_df.copy()
    inter_df["parent_asin"] = inter_df["parent_asin"].astype("string")
    meta_df["parent_asin"] = meta_df["parent_asin"].astype("string")

    # garder colonnes utiles metadata
    keep = ["parent_asin"] + [c for c in meta_keep_cols if c in meta_df.columns]
    meta_slim = meta_df[keep].drop_duplicates(subset=["parent_asin"], keep="first")

    # left join pour mesurer couverture + conserver interactions
    joined = inter_df.merge(meta_slim, on="parent_asin", how="left")

    # nettoyage minimal cohérent Task0
    # - clé obligatoire: on retire parent_asin manquant
    joined = joined[joined["parent_asin"].notna()].copy()

    # texte: NaN -> ""
    for c in METADATA_TEXT_COLS:
        if c in joined.columns:
            joined[c] = joined[c].fillna("")

    return joined


def save_source_diagnostics(result: Dict[str, Any], out_dir: str = "results/joining") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "source_diagnostics.json"
    md_path = out / "source_diagnostics.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines: List[str] = []
    lines.append("# Source Diagnostics")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    for s in result.get("sources", []):
        lines.append(f"## {s['name']}")
        lines.append(f"- stage: `{s['stage']}`")
        lines.append(f"- variant: `{s['variant']}`")
        lines.append(f"- role: `{s['role']}`")
        lines.append(f"- kind: `{s['kind']}`")
        lines.append(f"- exists: `{s['exists']}`")
        lines.append(f"- format: `{s['format']}`")
        lines.append(f"- rows: `{s['n_rows']}`")
        lines.append(f"- cols: `{s['n_cols']}`")
        lines.append(f"- size_bytes: `{s['size_bytes']}`")
        lines.append(f"- paths: `{s['paths']}`")
        lines.append("")

    lines.append("# Vérifications schéma et clés")
    lines.append("")
    for name, check in result.get("schema_checks", {}).items():
        lines.append(f"## {name}")
        lines.append(f"- ok: `{check.get('ok')}`")

        req = check.get("required_columns", {})
        lines.append(f"- colonnes_manquantes: `{req.get('missing_required', [])}`")

        mk = check.get("missing_keys", {})
        lines.append(f"- missing_parent_asin_count: `{mk.get('missing_parent_asin_count')}`")
        lines.append(f"- missing_parent_asin_pct: `{mk.get('missing_parent_asin_pct')}`")

        coercion = check.get("coercion", {})
        lines.append(f"- coercion_warning: `{coercion.get('warning')}`")
        lines.append(f"- warnings: `{check.get('warnings', [])}`")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": str(json_path), "md": str(md_path)}


def save_joined_dataset(df: pd.DataFrame, name: str, out_dir: str = "results/joining") -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_joined.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def run_all(
    verbose: bool = True,
    include_optional_raw: bool = False,
    export_artifacts: bool = True,
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
    meta_df = load_target_df(meta_cfg)


    for name, cfg in manifest.items():
        if not path_status.get(name, False):
            schema_checks[name] = {
                "target": name,
                "ok": False,
                "warnings": ["Chemin(s) manquant(s), vérification Task 3 ignorée"],
            }
            continue

        schema_checks[name] = run_schema_key_checks_for_target(name, cfg)

        if verbose and cfg["role"] in {"interactions", "metadata"}:
            mk = schema_checks[name].get("missing_keys", {})
            print(
                f"[{name}] ok={schema_checks[name]['ok']} "
                f"missing_parent_asin={mk.get('missing_parent_asin_count')} "
                f"({mk.get('missing_parent_asin_pct')}%)"
            )

        if cfg["role"] != "interactions":
            continue

        if not path_status.get(name, False):
            continue

        inter_df = load_target_df(cfg)

        # 1) join quality
        jm = compute_join_quality_metrics(inter_df, meta_df)
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
        joined_df = build_joined_dataset(inter_df, meta_df, meta_keep_cols=meta_keep)
        out_path = save_joined_dataset(joined_df, name=name, out_dir="results/joining")
        final_datasets[name] = {
            "path": out_path,
            "n_rows": len(joined_df),
            "n_cols": len(joined_df.columns),
        }


    result: Dict[str, Any] = {
        "manifest": manifest,
        "path_status": path_status,
        "sources": [asdict(s) for s in source_infos],
        "schema_checks": schema_checks,
        "join_metrics": join_metrics,
        "p1_reuse_note": build_p1_reuse_note(manifest, result["sources"]),
        "exploitable_columns": exploitable_cols,
        "missingness": missingness,
        "final_datasets": final_datasets,
    }

    if export_artifacts:
        result["artifacts"] = save_source_diagnostics(result, out_dir="results/joining")

    return result


def main() -> None:
    result = run_all(verbose=True, include_optional_raw=False, export_artifacts=True)

    print(f"\nLoaded {len(result['sources'])} source targets")
    for s in result["sources"]:
        print(f"\n- {s['name']}: rows={s['n_rows']:,}, cols={s['n_cols']}, exists={s['exists']}")
        print(f"   columns= {s['columns']}")  
        for p in s['paths']:
            print(f"   path: {p}")  

    if "artifacts" in result:
        print("\nArtifacts written:")
        print(f"  JSON: {result['artifacts']['json']}")
        print(f"  MD  : {result['artifacts']['md']}")



if __name__ == "__main__":
    main()