from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = sorted(Path("data/joining").glob("*_pre_split"))
TOP_N = 10



class DualLogger:
    """
    Logger simple:
    - affiche dans stdout (visible dans le notebook via subprocess)
    - memorise les memes lignes pour export Markdown
    """
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def add_table(self, title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
        self.log("")
        self.log(title)
        if df.empty:
            self.log("(vide)")
            return
        # On evite to_markdown pour ne pas dependre de tabulate
        table_txt = df.head(max_rows).to_string(index=False)
        self.log("```")
        self.log(table_txt)
        self.log("```")

    def to_markdown(self) -> str:
        return "\n".join(self.lines)


def parse_categories(x: Any) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        txt = x.strip().replace("[", "").replace("]", "")
        if not txt:
            return []
        return [p.strip(" '\"") for p in txt.split(",") if p.strip(" '\"")]
    return [str(x).strip()]


def flatten_categories(series: pd.Series) -> list[str]:
    out: list[str] = []
    for v in series:
        out.extend(parse_categories(v))
    return out


def top3_categories(series: pd.Series) -> str:
    cats = flatten_categories(series)
    if not cats:
        return "N/A"
    vc = pd.Series(cats).value_counts(normalize=True).head(3)
    return ", ".join([f"{k} ({100*v:.1f}%)" for k, v in vc.items()])


def select_representative_users(train_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """
    Selection de 5 profils:
    U1 gros lecteur, U2 modere, U3 petit, U4 eclectique, U5 specialise.
    """
    stats = train_df.groupby("user_id", as_index=False).agg(
        nb_train=("parent_asin", "nunique"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
    )
    stats["rating_std"] = stats["rating_std"].fillna(0.0)

    div = (
        train_df.groupby("user_id")["categories"]
        .apply(lambda s: len(set(flatten_categories(s))))
        .rename("n_categories")
        .reset_index()
    )
    stats = stats.merge(div, on="user_id", how="left")
    stats["n_categories"] = stats["n_categories"].fillna(0).astype(int)

    def dominant_ratio(df_u: pd.DataFrame) -> float:
        cats = flatten_categories(df_u["categories"])
        if not cats:
            return 0.0
        vc = pd.Series(cats).value_counts()
        return float(vc.iloc[0] / vc.sum())

    spec = train_df.groupby("user_id").apply(dominant_ratio).rename("dom_ratio").reset_index()
    stats = stats.merge(spec, on="user_id", how="left")

    used: set[str] = set()
    selected: list[tuple[str, str, str]] = []

    def pick(candidates: pd.Series, label: str, reason: str) -> None:
        for uid in candidates.tolist():
            if uid not in used:
                used.add(uid)
                selected.append((label, uid, reason))
                return

    q95_nb = stats["nb_train"].quantile(0.95)
    pick(
        stats[stats["nb_train"] >= q95_nb].sort_values("nb_train", ascending=False)["user_id"],
        "U1_gros_lecteur",
        f"nb_train >= q95 ({q95_nb:.1f})",
    )

    med = stats["nb_train"].median()
    pick(
        stats.assign(delta=(stats["nb_train"] - med).abs()).sort_values("delta")["user_id"],
        "U2_lecteur_modere",
        f"nb_train proche mediane ({med:.1f})",
    )

    small = stats[(stats["nb_train"] >= 3) & (stats["nb_train"] <= 5)].sort_values("nb_train")["user_id"]
    if len(small) == 0:
        small = stats[stats["nb_train"] > 0].sort_values("nb_train")["user_id"]
    pick(small, "U3_petit_lecteur", "nb_train 3-5 (ou minimum disponible)")

    q95_cat = stats["n_categories"].quantile(0.95)
    pick(
        stats[stats["n_categories"] >= q95_cat].sort_values("n_categories", ascending=False)["user_id"],
        "U4_lecteur_eclectique",
        f"n_categories >= q95 ({q95_cat:.1f})",
    )

    pick(
        stats[stats["dom_ratio"] >= 0.8].sort_values("dom_ratio", ascending=False)["user_id"],
        "U5_lecteur_specialise",
        "plus de 80% des lectures dans une categorie dominante",
    )

    return selected


def qualitative_analysis(variant_dir: Path, top_n: int, output_md: Path) -> None:
    log = DualLogger()

    # Chargement artefacts existants
    user_ids = np.load(variant_dir / "user_ids.npy", allow_pickle=True)
    item_ids = np.load(variant_dir / "item_ids.npy", allow_pickle=True)
    item_titles = np.load(variant_dir / "item_titles.npy", allow_pickle=True)

    top_idx_path = variant_dir / f"top_n_indices_{top_n}.npy"
    if not top_idx_path.exists():
        raise FileNotFoundError(
            f"Artefact manquant: {top_idx_path}. "
            "Lance d'abord scripts/similarity.py pour générer les top_n_indices."
        )

    top_idx_all = np.load(top_idx_path, allow_pickle=False)
    if top_idx_all.ndim != 2 or top_idx_all.shape[0] != len(user_ids):
        raise ValueError(
            f"Shape invalide pour {top_idx_path.name}: {top_idx_all.shape}, "
            f"attendu (*, {top_n}) avec nb_users={len(user_ids)}"
        )
    if top_idx_all.shape[1] < top_n:
        raise ValueError(
            f"{top_idx_path.name} contient {top_idx_all.shape[1]} colonnes < top_n={top_n}"
        )

    item_matrix = load_npz(variant_dir / "books_representation_sparse.npz")

    # On limite les colonnes lues pour economiser la memoire
    cols = ["user_id", "parent_asin", "rating", "timestamp", "title", "categories"]
    train = pd.read_parquet(variant_dir / "train_interactions.parquet", columns=cols)
    test = pd.read_parquet(variant_dir / "test_interactions.parquet", columns=cols)

    # Mappings utiles
    user_to_row = {u: i for i, u in enumerate(user_ids)}
    item_to_col = {a: j for j, a in enumerate(item_ids)}
    col_to_asin = {j: a for a, j in item_to_col.items()}

    meta = pd.concat(
        [train[["parent_asin", "title", "categories"]], test[["parent_asin", "title", "categories"]]],
        ignore_index=True,
    ).drop_duplicates(subset=["parent_asin"], keep="first")
    asin_to_title = dict(zip(meta["parent_asin"], meta["title"]))
    asin_to_cat = dict(zip(meta["parent_asin"], meta["categories"]))

    selected = select_representative_users(train)

    log.log("# Analyse qualitative 3.2.3")
    log.log(f"Variante: {variant_dir.name}")
    log.log(f"Utilisateurs analyses: {len(selected)}")

    summary_rows: list[dict[str, Any]] = []

    for label, uid, reason in selected:
        if uid not in user_to_row:
            continue
        row = user_to_row[uid]

        # Point cle memory safe: un seul utilisateur a la fois
        # Recommandations déjà calculées par similarity.py (alignées avec user_ids)
        top_idx = top_idx_all[row, :top_n].astype(np.int64, copy=False)

        rec_asins = [col_to_asin[int(j)] for j in top_idx]
        rec_titles = [asin_to_title.get(a, str(item_titles[int(j)])) for a, j in zip(rec_asins, top_idx)]
        rec_cats = [", ".join(parse_categories(asin_to_cat.get(a))) for a in rec_asins]

        u_train = train[train["user_id"] == uid].copy()
        u_test = test[test["user_id"] == uid].copy()

        hist = u_train[["title", "categories", "rating", "timestamp"]].copy()
        hist.columns = ["titre", "categories", "rating", "date"]
        hist["categories"] = hist["categories"].apply(lambda x: ", ".join(parse_categories(x)))

        rec_tbl = pd.DataFrame(
            {
                "rang": np.arange(1, top_n + 1),
                "asin": rec_asins,
                "titre_recommande": rec_titles,
                "categories": rec_cats,
                "score_source": ["top_n_indices_similarity.py"] * top_n,
            }
        )

        gt = u_test[["title", "categories", "rating"]].copy()
        gt.columns = ["titre", "categories", "rating"]
        gt["categories"] = gt["categories"].apply(lambda x: ", ".join(parse_categories(x)))

        # Axe 1: coherence thematique
        train_cats = set(flatten_categories(u_train["categories"]))
        reco_cats = set([c for s in rec_cats for c in parse_categories(s)])
        overlap = (len(train_cats & reco_cats) / len(train_cats)) if train_cats else 0.0

        # Axe 2: proximite apparente (top 3 recos les plus proches du livre prefere)
        best_train = u_train.sort_values("rating", ascending=False).head(1)
        prox_examples: list[tuple[str, str, float]] = []
        if len(best_train) > 0:
            best_asin = best_train["parent_asin"].iloc[0]
            if best_asin in item_to_col:
                best_idx = item_to_col[best_asin]
                best_vec = item_matrix[best_idx]
                rec_vecs = item_matrix[top_idx]
                sim_best_to_rec = cosine_similarity(best_vec, rec_vecs).ravel()
                top_ex = np.argsort(-sim_best_to_rec)[:3]
                for k in top_ex:
                    prox_examples.append(
                        (str(best_train["title"].iloc[0]), rec_titles[int(k)], float(sim_best_to_rec[int(k)]))
                    )
                del best_vec, rec_vecs, sim_best_to_rec, top_ex

        # Axe 3: redondance intra liste
        rec_mat = item_matrix[top_idx]
        sim_mat = cosine_similarity(rec_mat, rec_mat)
        iu = np.triu_indices(sim_mat.shape[0], k=1)
        intra_vals = sim_mat[iu] if len(iu[0]) > 0 else np.array([0.0], dtype=np.float32)
        intra_mean = float(np.mean(intra_vals))
        intra_max = float(np.max(intra_vals))

        # Axe 4: sur specialisation
        train_vc = pd.Series(flatten_categories(u_train["categories"])).value_counts(normalize=True)
        reco_vc = pd.Series([c for s in rec_cats for c in parse_categories(s)]).value_counts(normalize=True)
        dom_train_cat = train_vc.index[0] if len(train_vc) else None
        concentration = float(reco_vc.get(dom_train_cat, 0.0)) if dom_train_cat else 0.0

        # Hit rate de support
        test_asins = set(u_test["parent_asin"].astype(str).unique())
        hits = len(set(map(str, rec_asins)) & test_asins)

        log.log("")
        log.log("=" * 80)
        log.log(f"{label} | user_id={uid}")
        log.log(f"Selection: {reason}")

        log.add_table("[Historique train]", hist, max_rows=20)
        log.add_table("[Recommandations top N]", rec_tbl, max_rows=top_n)
        log.add_table("[Verite terrain test]", gt, max_rows=20)

        log.log("")
        log.log("[Analyse 4 axes]")
        log.log(f"- Coherence thematique: recouvrement categories = {overlap:.2f}")
        if prox_examples:
            log.log("- Proximite apparente (exemples):")
            for t_train, t_rec, s in prox_examples:
                log.log(f"  * '{t_train}' -> '{t_rec}' (sim={s:.2f})")
        else:
            log.log("- Proximite apparente: pas d exemple calcule")
        log.log(f"- Redondance: similarite intra liste moyenne = {intra_mean:.2f}, max = {intra_max:.2f}")
        log.log(f"- Sur specialisation: concentration categorie dominante train = {concentration:.2f}")
        log.log(f"- Hit rate test: {hits}/{top_n}")
        log.log(f"- Top categories train: {top3_categories(u_train['categories'])}")
        log.log(f"- Top categories recos: {top3_categories(pd.Series(rec_cats))}")

        summary_rows.append(
            {
                "utilisateur": uid,
                "profil": label,
                "nb_train": int(u_train["parent_asin"].nunique()),
                "recouvrement": round(overlap, 3),
                "intra_moy": round(intra_mean, 3),
                "intra_max": round(intra_max, 3),
                "hit_rate": f"{hits}/{top_n}",
                "concentration": round(concentration, 3),
                "sur_specialisation_probable": concentration >= 0.70,
            }
        )

        # Liberation explicite des temporaires de boucle
        del (
            top_idx, rec_asins, rec_titles, rec_cats,
            u_train, u_test, hist, rec_tbl, gt, train_cats, reco_cats,
            best_train, prox_examples, rec_mat, sim_mat, iu, intra_vals,
            train_vc, reco_vc, test_asins
        )
        gc.collect()

    log.log("")
    log.log("=" * 80)
    log.log("Synthese finale")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        log.add_table("[Tableau recapitulatif]", summary_df, max_rows=20)
    else:
        log.log("Aucun utilisateur selectionne")

    # Ecriture markdown avec le meme contenu que la sortie console
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(log.to_markdown(), encoding="utf-8")
    log.log("")
    log.log(f"Fichier markdown ecrit: {output_md}")

    # Nettoyage final
    del (
        user_ids, item_ids, item_titles, item_matrix, train, test, meta,
        asin_to_title, asin_to_cat, user_to_row, item_to_col, col_to_asin, selected, summary_rows
    )
    gc.collect()


def main() -> None:
    for variant_dir in DATA_DIR:
        
        results_report_dir = Path("results") / variant_dir.relative_to("data")
        results_report_dir.mkdir(parents=True, exist_ok=True)
        output_md = Path(results_report_dir / "qualitative_analysis.md")
        qualitative_analysis(variant_dir=variant_dir, top_n=TOP_N, output_md=output_md)


if __name__ == "__main__":
    main()