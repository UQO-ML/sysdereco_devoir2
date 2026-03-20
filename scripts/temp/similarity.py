from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from pathlib import Path
import pandas as pd

BATCH_SIZE = 5_000
DATA_DIR = sorted(Path("data/joining").glob("*_pre_split"))

def compute_similarity(user_profiles, item_matrix, batch_size=500, n=10):
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

    # Tri pour la recommandation Top-N
    top_n_indices = np.argsort(-scores, axis=1)[:, :n]
    
    return top_n_indices



def get_recommendations(top_n_indices, item_ids, item_titles):
    """Retourne les IDs ou titres des livres recommandés."""
    print("get_recommendations()\n")
    recommendations = []
    for user_indices in top_n_indices:
        user_recommendations = [item_titles[item_ids[i]] for i in user_indices]
        recommendations.append(user_recommendations)
    return recommendations

def main() -> None:
    t0 = time.perf_counter()
    for data_dir in DATA_DIR:

        t1 = time.perf_counter()
        print(f"Path: {data_dir}\n")
        user_profiles_paths = sorted(Path(data_dir).glob("user_profiles_tfidf*"))

        clean_path = data_dir.parent / f"{data_dir.name}_clean_joined.parquet"
        src_path = clean_path if clean_path.exists() else (data_dir / "train_interactions.parquet")

        books_data = pd.read_parquet(src_path, columns=["parent_asin", "title"])

        # 2) Déduplication en gardant le premier (même logique que item_representation)
        mask = ~books_data["parent_asin"].duplicated(keep="first")
        items = books_data.loc[mask, ["parent_asin", "title"]].reset_index(drop=True)

        # 3) Structures efficaces
        indice_to_id = items["parent_asin"].to_numpy()

        # Dictionnaire {parent_asin: titre}
        id_to_title = dict(zip(books_data["parent_asin"], books_data["title"]))
        top_n = 10

        for user_profiles_path in user_profiles_paths:

            if user_profiles_path.suffix == ".npz":
                user_profiles = load_npz(user_profiles_path)
            elif user_profiles_path.suffix == ".npy":
                user_profiles = np.load(user_profiles_path, allow_pickle=False)  # dense
            else:
                continue

            clean_src = user_profiles_path.parent / "books_representation_sparse.npz"
            item_matrix = load_npz(clean_src)

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

            np.save(user_profiles_path.parent / f"recommendations_top{top_n}.npy", recommendations)

            print(f"Exemple de recommandations pour le premier utilisateur: {recommendations[0]}\n")
            print(f"Temps écoulé pour ce lot: {(time.perf_counter() - t1):.1f}s\n")

    print(f"Elapse total: {(time.perf_counter() - t0):.1f}s\n")


if __name__ == "__main__":
    main()