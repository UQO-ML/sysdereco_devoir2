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




def compute_similarity(user_profiles, item_matrix, batch_size=500, n=10) -> Any:
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




def top_n_sim(n: int, scores) -> Any:
    """Tri pour la recommandation Top-N"""
    return np.argsort(-scores, axis=1)[:, :n]




def load_dataset(data_dir: Path) -> Tuple:

        books_data = pd.read_parquet(f"{data_dir}/train_interactions.parquet", columns=["parent_asin", "title"])
        
        # Dictionnaire {indice: parent_asin}
        indice_to_id = {i: parent_asin for i, parent_asin in enumerate(books_data["parent_asin"].tolist())}
        
        # Dictionnaire {parent_asin: titre}
        id_to_title = dict(zip(books_data["parent_asin"], books_data["title"]))
        
        del books_data
        gc.collect()

        return indice_to_id, id_to_title




def load_item_matrix(user_profiles_path: Path):
    
    clean_src = user_profiles_path.parent / "books_representation_sparse.npz"
    item_matrix = load_npz(clean_src)

    return item_matrix




def get_recommendations(top_n_indices, item_ids, item_titles):
    """Retourne les IDs ou titres des livres recommandés."""
    print("get_recommendations()\n")

    recommendations = []

    for user_indices in top_n_indices:
        user_recommendations = [item_titles[item_ids[i]] for i in user_indices]
        recommendations.append(user_recommendations)

    return recommendations




def save_recommendations(recommendations, user_profiles_path: Path, top_n: int = TOP_N):
    """Sauvegarde la liste des titre resultat de la selection top_n"""

    dest_path : Path(user_profiles_path.parent / f"recommendations_top_{top_n}.npy")
    np.save(dest_path, recommendations)

    return print(f"Fichier de recommendations dense: {_fmt_size(os.path.getsize(dest_path))}")




def main() -> None:

    t0 = time.perf_counter()
    for data_dir in DATA_DIR:
        
        top_n = TOP_N
        t1 = time.perf_counter()
        print(f"Path: {data_dir}\n")

        user_profiles_paths = sorted(Path(data_dir).glob("user_profiles_tfidf*"))

        indice_to_id, id_to_title = load_dataset(data_dir=data_dir)

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

            score_top_n = compute_similarity(
                user_profiles=user_profiles,
                item_matrix=item_matrix,
                batch_size=10_000,
                n=top_n
            )

            print(f"Calcule la similarité cosinus profil-item: {score_top_n}\n")
            print(f"Elapse partiel: {(time.perf_counter() - t1):.1f}s\n")

            recommendations = get_recommendations(score_top_n, indice_to_id, id_to_title)

            save_recommendations(recommendations=recommendations, user_profiles_path=user_profiles_path, top_n=top_n)
                        
            print(f"Exemple de recommandations pour le premier utilisateur: {recommendations[0]}\n")
            print(f"Temps écoulé pour ce lot: {(time.perf_counter() - t1):.1f}s\n")

    print(f"Elapse total: {(time.perf_counter() - t0):.1f}s\n")


if __name__ == "__main__":
    main()