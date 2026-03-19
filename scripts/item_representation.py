from __future__ import annotations

import pandas as pd

import re
import time
import glob
import os
import gc

from typing import Any, Dict, List, Optional, Tuple

from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix, hstack, save_npz
from pathlib import Path


import numpy as np


BOOKS_DATA_PATH = "data/joining/active_pre_split_clean_joined.parquet"

GLOB_PATTERN = "*_clean_joined.parquet"
GLOB_SUFFIX = GLOB_PATTERN.replace("*", "")  # → "_clean_joined.parquet"

CLEAN_DATASETS_PATHS = sorted(Path("data/joining").glob(GLOB_PATTERN))

# Gestion des stop words
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

# Gestion de la ponctuation
PUNCTUATION_TABLE = str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

TFIDF_PARAMS = {
    "max_features": None,   # Plafond du vocabulaire : garde les termes les plus fréquents globalement au-delà des filtres min_df/max_df.
    "max_df": 0.95, # Ignore les termes présents dans > NN % des items (quasi stopwords “métiers”).
    "min_df": 5,    # Un mot doit apparaître dans au moins N documents pour entrer dans le vocab → réduit bruit et taille.
    "stop_words": "english",    # Retire une liste fixe de mots vides anglais (en plus du filtrage par fréquence).
    "lowercase": True,
    "token_pattern": r"(?u)\b[a-zA-Z]{2,}\b",
    "dtype": np.float32, # Matrice plus légère qu'en float64.
}

def _fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"


def _disk_size(path: str) -> str:
    try:
        return _fmt_size(os.path.getsize(path))
    except OSError:
        return "N/A"


def _df_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def load_dataset(path: str) -> pd.DataFrame :
    print("load_dataset()")
    return pd.read_parquet(path)


def category_formating(books_data = pd.DataFrame) -> pd.DataFrame:
    print("category_formating()")

    # Dédupliquer au niveau item pour éviter plusieurs lignes par parent_asin
    if isinstance(books_data, pd.DataFrame) and "parent_asin" in books_data.columns:
        before_dedup = len(books_data)

        books_data = books_data.drop_duplicates(subset=["parent_asin"]).reset_index(drop=True)

        after_dedup = len(books_data)

        if before_dedup != after_dedup:
            print(f"category_formating(): deduplicated by parent_asin: {before_dedup} -> {after_dedup} rows")

    # Formattage éventuel des catégories
    if isinstance(books_data.at[0, "categories"], str):
        books_data["categories"] = books_data["categories"].apply(lambda x: x.split(", "))

    # Combinaison des colonnes 'title', 'description' et 'categories'
    books_data["combined_infos"] = books_data.apply(
        lambda row: f"{row['title']}. {row['description']}. {' '.join(row['categories'])}",
        axis=1
    )

    # Vérification
    print(books_data["combined_infos"].iloc[0])
    return books_data





def clean_text(text) -> str:
    # Suppression des URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Suppression de la ponctuation
    text = text.translate(PUNCTUATION_TABLE)
    # Suppression des chiffres
    text = re.sub(r'\d+', '', text)
    # Conversion en minuscules et enlèvement des accents
    text = unidecode(text.lower())
    # Tokenisation et filtrage des stopwords
    tokens = text.split()
    tokens = [
        token for token in tokens 
        if token not in STOPWORDS 
        and len(token) > 2
        and token.isalpha()
        and not re.search(r'(.)\1{2,}', token)
    ]
    return " ".join(tokens)



def info_cleaning(books_data: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    t_start = time.perf_counter()
    print("info_cleaning()")
   
    batch_size = 10000
    cleaned_texts = []

    batch_number = 0

    # for i in range(0, len(books_data), batch_size):
    #     batch = books_data["combined_infos"].iloc[i:i+batch_size]
    #     cleaned_batch = batch.apply(clean_text)
    #     cleaned_texts.extend(cleaned_batch)
    #     batch_number += 1
    #     print(f"batch number: {batch_number}")

    cleaned_texts = books_data["combined_infos"].apply(clean_text)

    books_data["cleaned_infos"] = cleaned_texts

    # Vérification
    print(f"Elapsed time: {(time.perf_counter() - t_start):.1f}")
    print(books_data["cleaned_infos"].iloc[0])

    return books_data, cleaned_texts


def vectorization(cleaned_texts: str ) -> Tuple:
    print("\nvectorization()\n")

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    x_tfidf = vectorizer.fit_transform(cleaned_texts)

    # Vérifications
    print(f"Dimension de la matrice TF-IDF : {x_tfidf.shape}")
    print(f"Vocabulaire (10 premiers mots) : {vectorizer.get_feature_names_out()[:10]}")
    
    return x_tfidf, vectorizer

def struct_attr(books_data: pd.DataFrame) -> pd.DataFrame:
    print("\nstruct_attr")

    # Extraction des attributs
    attributes = books_data[["average_rating", "price"]].values

    # Normalisation des attributs
    scaler = MinMaxScaler()
    normalized_attributes = scaler.fit_transform(attributes)

    # Vérification (5 premières lignes)
    print(f"{normalized_attributes[:5]}\n")

    return normalized_attributes


def fuze_save(x_tfidf: Tuple, 
    normalized_attributes: pd.DataFrame, 
    out_dir: Path,
    vectorizer: TfidfVectorizer,
):
    # Conversion des attributs normalisés en matrice creuse
    attributs_sparse = csr_matrix(normalized_attributes)

    # Fusion des matrices creuses
    final_representation = hstack([x_tfidf, attributs_sparse])

    # Sauvegarde
    save_npz(out_dir / "books_representation_sparse.npz", final_representation)

    # Sauvegarde du vocabulaire TF-IDF
    with open(out_dir / "vocabulary_tfidf.txt", "w") as f:
        f.write("\n".join(vectorizer.get_feature_names_out()))
    
    return final_representation

def attr_checks(vectorizer: TfidfVectorizer, final_representation: csr_matrix) -> None:

    print("\nattr_checks()\n")
    feature_names = vectorizer.get_feature_names_out()

    # Affichage des 10 mots-clés dominants pour les 5 premiers livres
    for i in range(5):
        # Indices des 10 plus grands poids TF-IDF
        top_indices = final_representation[i, :-2].toarray().argsort()[0][-10:][::-1]
        # Noms des mots-clés correspondants
        top_features = [feature_names[idx] for idx in top_indices]
        print(f"Livre {i+1} - Mots-clés dominants : {top_features}")
    # Attributs structurés pour les 5 premiers livres
    for i in range(5):
        attributes = final_representation[i, -2:].toarray()[0]
        print(f"Livre {i+1} - Attributs structurés : {attributes}")


def main() -> None:
    t0 = time.perf_counter()

    for path in CLEAN_DATASETS_PATHS:

        t1 = time.perf_counter()
        folder_name = path.name.removesuffix(GLOB_SUFFIX)   
        out_dir = path.parent / folder_name                  
        out_dir.mkdir(parents=True, exist_ok=True)

        books_data = load_dataset(path)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")
        
        books_data = category_formating(books_data)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")

        books_data, cleaned_texts = info_cleaning(books_data)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")
        
        x_tfidf, vectorizer = vectorization(cleaned_texts)
                
        print(books_data[["average_rating", "price"]].isnull().sum())
        normalized_attributes = struct_attr(books_data)

        del books_data, cleaned_texts
        gc.collect()

        final_representation = fuze_save(x_tfidf=x_tfidf, normalized_attributes=normalized_attributes, out_dir=out_dir, vectorizer=vectorizer)
        del x_tfidf, normalized_attributes
        gc.collect()

        # Dimension finale
        print(f"Dimension de la représentation finale : {final_representation.shape}")

        # Densité
        density = final_representation.nnz / (final_representation.shape[0] * final_representation.shape[1])
        print(f"Densité de la matrice\ndensity = final_representation.nnz / (final_representation.shape[0] * final_representation.shape[1])\n = {density:.4f} ({density:.2%})")

        attr_checks(vectorizer=vectorizer, final_representation=final_representation)
        del vectorizer, final_representation
        gc.collect()
        print(f"item_representation elapse for \n{path}\n: {(time.perf_counter() - t1):.1f} s")

    print(f"item_representation total elapse: {(time.perf_counter() - t0):.1f} s")


if __name__ == "__main__":
    main()