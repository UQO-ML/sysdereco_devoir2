from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from pathlib import Path

BATCH_SIZE = 5_000
DATA_DIR = sorted(Path("data/joining").glob("*_pre_split"))


def compute_similarity(user_profiles, item_matrix, batch_size=500):
    """Calcule la similarité cosinus profil-item par batch."""
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


def main() -> None:
    t0 = time.perf_counter()
    for data_dir in DATA_DIR:
        t1 = time.perf_counter()
        print(f"Path: \n{data_dir}\n")
        user_profiles_paths = sorted(Path(data_dir).glob("user_profiles_tfidf*"))
        for user_profiles_path in user_profiles_paths:
            if ".npz" in user_profiles_path:
                user_profiles = load_npz(user_profiles_path)
            elif ".npy" in user_profiles_path:
                

            item_matrix = Path(dir / "books_representation_sparse.npz")
            score = compute_similarity(
                user_profiles=user_profiles,
                item_matrix=item_matrix,
                batch_size=10_000
            )
            print(f"Calcule la similarité cosinus profil-item: {score}\n")
            print(f"Elapse total: {(time.perf_counter() - t1):.1f}s\n")
    print(f"Elapse total: {(time.perf_counter() - t0):.1f}s\n")


if __name__ == "__main__":
    main()