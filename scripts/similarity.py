from typing import Any, Tuple
import time
import gc
import os

from pathlib import Path

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd

BATCH_SIZE = 5_000
DATA_DIR = sorted(Path("data/joining").glob("*_pre_split"))
TOP_N = 10




def _fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"




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
    """Retourne les IDs ou titres des livres recommandés."""
    print("get_recommendations()\n")

    recommendations = []

    for user_indices in top_n_indices:
        user_recommendations = [item_titles[i] for i in user_indices]
        recommendations.append(user_recommendations)

    return recommendations




def save_recommendations(recommendations, user_profiles_path: Path, top_n: int = TOP_N):
    """Sauvegarde la liste des titre resultat de la selection top_n"""

    recommendations_array = np.array(recommendations, dtype=str)
    
    dest_path = Path(user_profiles_path.parent / f"recommendations_top_{top_n}.npy")
    np.save(dest_path, recommendations_array)

    return print(f"Fichier de recommendations dense: {_fmt_size(os.path.getsize(dest_path))}")




def main() -> None:
    
    top_n = TOP_N
    t0 = time.perf_counter()
    for data_dir in DATA_DIR:
        
        t1 = time.perf_counter()
        print(f"Path: {data_dir}\n")

        user_profiles_paths = sorted(Path(data_dir).glob("user_profiles_tfidf*"))

        item_ids, item_titles = load_item_tables(data_dir=data_dir)

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

            scores = compute_similarity(
                user_profiles=user_profiles,
                item_matrix=item_matrix,
                batch_size=10_000
            )
            print(f"Calcule la similarité cosinus profil-item: {scores}\n")

            score_top_n = top_n_sim(scores)
            print(f"Calcule la similarité cosinus profil-item du Top {top_n}: {scores}\n")

            recommendations = get_recommendations(top_n_indices=score_top_n, item_ids=item_ids, item_titles=item_titles)
            
            print(f"Exemple de recommandations pour le premier utilisateur: {recommendations[0]}\n")

            save_recommendations(recommendations=recommendations, user_profiles_path=user_profiles_path, top_n=top_n)
                        
            print(f"Temps écoulé pour ce lot: {(time.perf_counter() - t1):.1f}s\n")

    print(f"Elapse total: {(time.perf_counter() - t0):.1f}s\n")


if __name__ == "__main__":
    main()