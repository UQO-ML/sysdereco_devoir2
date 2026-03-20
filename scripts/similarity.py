from typing import Any, Tuple
import time
import gc
import os
import psutil

from pathlib import Path

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

import numpy as np
import pandas as pd

BATCH_SIZE = 10_000  # Super safe value are 256 or 512
DATA_DIR = sorted(Path("data/joining").glob("*_pre_split"))
TOP_N = 10




def _fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"
    



def build_seen_indices(data_dir: Path, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit les indices (rows, cols) des interactions déjà vues dans le train:
    - rows: index utilisateur dans user_profiles
    - cols: index item dans item_matrix
    """
    # Ordre des utilisateurs = ordre des lignes de user_profiles/scores
    user_ids = np.load(data_dir / "user_ids.npy", allow_pickle=True)
    user_to_row = {u: i for i, u in enumerate(user_ids)}

    # Ordre des items = ordre des colonnes de score (item_matrix rows)
    item_to_col = {asin: j for j, asin in enumerate(item_ids)}

    train_df = pd.read_parquet(
        data_dir / "train_interactions.parquet",
        columns=["user_id", "parent_asin"],
    ).drop_duplicates(subset=["user_id", "parent_asin"], keep="first")

    rows = train_df["user_id"].map(user_to_row).to_numpy()
    cols = train_df["parent_asin"].map(item_to_col).to_numpy()
    del train_df
    gc.collect()

    valid = (~pd.isna(rows)) & (~pd.isna(cols))

    rows = rows[valid].astype(np.int64, copy=False)
    cols = cols[valid].astype(np.int64, copy=False)
     
    return rows, cols




def mask_seen_items(scores: np.ndarray, seen_rows: np.ndarray, seen_cols: np.ndarray) -> None:
    """
    Masque in-place les items déjà vus: ils ne pourront pas sortir dans le top-N.
    """
    scores[seen_rows, seen_cols] = -np.inf





def build_seen_by_user_row(seen_rows, seen_cols):
    d = defaultdict(list)
    for r, c in zip(seen_rows, seen_cols):
        d[int(r)].append(int(c))
    return {r: np.asarray(cols, dtype=np.int64) for r, cols in d.items()}




def compute_similarity(user_profiles, item_matrix, batch_size=500) -> Any:
    """Calcule la similarité cosinus profil-item par batch."""

    print("compute_similarity()\n")

    n_users = user_profiles.shape[0]
    n_items = item_matrix.shape[0]

    # Résultat en float32 pour limiter la mémoire
    scores = np.empty((n_users, n_items), dtype=np.float32)
    
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        scores[start:end] = cosine_similarity(
            user_profiles[start:end], item_matrix
        )

    return scores




def top_n_sim(scores, n: int = TOP_N) -> Any:
    """Tri pour la recommandation Top-N"""
    top_n_indices = np.argsort(-scores, axis=1)[:, :n]
    return top_n_indices



def compute_top_n_memory_safe(
    user_profiles,
    item_matrix,
    seen_by_user_row: dict[int, np.ndarray],
    top_n: int = 10,
    batch_size: int = 512,
) -> np.ndarray:
    n_users = user_profiles.shape[0]
    top_idx_all = np.empty((n_users, top_n), dtype=np.int32)

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)

        # Dense bloc temporaire seulement
        block_scores = cosine_similarity(user_profiles[start:end], item_matrix).astype(np.float32, copy=False)

        # Masquage "déjà vus"
        for local_r, global_r in enumerate(range(start, end)):
            seen_cols = seen_by_user_row.get(global_r)
            if seen_cols is not None and len(seen_cols) > 0:
                block_scores[local_r, seen_cols] = -np.inf

        # Top-N sans trier toute la ligne
        part = np.argpartition(-block_scores, kth=top_n - 1, axis=1)[:, :top_n]
        rr = np.arange(block_scores.shape[0])[:, None]
        ord_ = np.argsort(-block_scores[rr, part], axis=1)
        top_idx_all[start:end] = part[rr, ord_]

        del block_scores

    return top_idx_all





def load_item_tables(data_dir: Path):

    item_ids_file_path = Path(data_dir / "item_ids.npy")
    item_titles_file_path = Path(data_dir / "item_titles.npy")

    if item_ids_file_path.exists() and \
        item_titles_file_path.exists():
        print("item_ids_file_path.exists() et item_titles_file_path.exists(): True")
        item_ids = np.load(item_ids_file_path, allow_pickle=True)
        item_titles = np.load(item_titles_file_path, allow_pickle=True)
        
    else:
        # 1) Chemin déterministe de la source alignée avec item_representation
        clean_path = data_dir.parent / f"{data_dir.name}_clean_joined.parquet"
        src_path = clean_path if clean_path.exists() else (data_dir / "train_interactions.parquet")

        df = pd.read_parquet(src_path, columns=["parent_asin", "title"])

        # 2) Déduplication en gardant le premier (même logique que item_representation)
        mask = ~df["parent_asin"].duplicated(keep="first")
        items = df.loc[mask, ["parent_asin", "title"]].reset_index(drop=True)

        # 3) Structures efficaces
        item_ids = items["parent_asin"].to_numpy()              # index i -> asin
        item_titles = items.drop_duplicates("parent_asin").set_index("parent_asin")["title"].to_dict()

    return item_ids, item_titles




def load_item_matrix(user_profiles_path: Path):
    
    clean_src = user_profiles_path.parent / "books_representation_sparse.npz"
    item_matrix = load_npz(clean_src)

    return item_matrix




def get_recommendations(top_n_indices, item_ids, item_titles):
    recommendations = []

    if isinstance(item_titles, dict):
        # item_titles maps asin -> title
        for user_indices in top_n_indices:
            user_recommendations = [item_titles[item_ids[i]] for i in user_indices]
            recommendations.append(user_recommendations)
    elif isinstance(item_titles, (list, tuple, np.ndarray)):
        # item_titles is array-like aligned with item index
        for user_indices in top_n_indices:
            user_recommendations = [item_titles[i] for i in user_indices]
            recommendations.append(user_recommendations)
    else:
        raise TypeError(f"Unsupported item_titles type: {type(item_titles)}")

    return recommendations





def save_recommendations(recommendations, user_profiles_path: Path, top_n: int = TOP_N):
    """Sauvegarde la liste des titre resultat de la selection top_n"""

    recommendations_array = np.array(recommendations, dtype=str)
    
    dest_path = Path(user_profiles_path.parent / f"recommendations_top_{top_n}.npy")
    np.save(dest_path, recommendations_array)

    return print(f"Fichier de recommendations dense: {_fmt_size(os.path.getsize(dest_path))}")





def estimate_scores_memory(user_profiles, item_matrix, dtype=np.float32):
    n_users = user_profiles.shape[0]
    n_items = item_matrix.shape[0]
    bytes_ = n_users * n_items * np.dtype(dtype).itemsize
    print("bytes:", bytes_)
    print("MiB:", bytes_ / (1024**2))
    print("GiB:", bytes_ / (1024**3))
    avail = psutil.virtual_memory().available
    can_use_dense = bytes_ < (avail * 0.5)
    return can_use_dense




    
def main() -> None:
    
    top_n = TOP_N
    t0 = time.perf_counter()
    for data_dir in DATA_DIR:
        
        t1 = time.perf_counter()
        print(f"Path: {data_dir}\n")

        user_profiles_paths = sorted(Path(data_dir).glob("user_profiles_tfidf*"))

        item_ids, item_titles = load_item_tables(data_dir=data_dir)
        seen_rows, seen_cols = build_seen_indices(data_dir=data_dir, item_ids=item_ids)

        for user_profiles_path in user_profiles_paths:
            if user_profiles_path.suffix == ".npz":
                user_profiles = load_npz(user_profiles_path)
            elif user_profiles_path.suffix == ".npy":
                user_profiles = np.load(user_profiles_path, allow_pickle=False)  # dense
            else:
                continue

            item_matrix = load_item_matrix(user_profiles_path=user_profiles_path)

            print("Dimension: \n"
                f"user_profiles: {user_profiles.shape}\n"
                f"item_matrix: {item_matrix.shape}\n")

            if estimate_scores_memory(
                user_profiles=user_profiles,
                item_matrix=item_matrix):
                
                scores = compute_similarity(
                    user_profiles=user_profiles,
                    item_matrix=item_matrix,
                    batch_size=BATCH_SIZE
                )
                print(f"Calcule la similarité cosinus profil-item: {scores}\n")

                mask_seen_items(scores, seen_rows, seen_cols)
                top_n_indices = top_n_sim(scores, n=top_n)
                print(f"Calcule la similarité cosinus profil-item du Top {top_n}: {top_n_indices}\n")

            else:
                seen_by_user_row = build_seen_by_user_row(seen_rows=seen_rows,seen_cols=seen_cols)
                top_n_indices = compute_top_n_memory_safe(
                    user_profiles=user_profiles,
                    item_matrix=item_matrix,
                    seen_by_user_row=seen_by_user_row,
                    top_n=top_n,
                    batch_size=512,
                )
                print(f"Calcule la similarité cosinus profil-item du Top {top_n}: {top_n_indices}\n")

            recommendations = get_recommendations(
                top_n_indices=top_n_indices,
                item_ids=item_ids,
                item_titles=item_titles
            )              
            print(f"Exemple de recommandations pour le premier utilisateur: {recommendations[0]}\n")

            save_recommendations(recommendations=recommendations, user_profiles_path=user_profiles_path, top_n=top_n)
                        
            print(f"Temps écoulé pour ce lot: {(time.perf_counter() - t1):.1f}s\n")

    print(f"Elapse total: {(time.perf_counter() - t0):.1f}s\n")


if __name__ == "__main__":
    main()