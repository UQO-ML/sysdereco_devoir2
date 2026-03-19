from __future__ import annotations

import pandas as pd

import re
from unidecode import unidecode
import time
import glob
import os

books_data_PATH = "data/joining/active_pre_split_clean_joined.parquet"
CLEAN_DATASETS_PATHS = sorted(glob.glob("data/joining/*clean_joined.parquet"))

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


def load_dataset(path = str) -> pd.DataFrame :
    print("load_dataset()")
    return pd.read_parquet(path)


def category_formating(books_data = pd.DataFrame) -> pd.DataFrame:
    print("category_formating()")

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



t_start = time.time()


def clean_text(text):
    # print("clean_text()")

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

def info_cleaning(books_data = pd.DataFrame):
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
    print(f"Elapsed time: {(time.time() - t_start):.1f}")
    print(books_data["cleaned_infos"].iloc[0])

    return books_data


def main() -> None:

    for path in CLEAN_DATASETS_PATHS:

        books_data = load_dataset(path)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")
        
        books_data = category_formating(books_data)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")

        books_data = info_cleaning(books_data)
        print(f"\n{path},\ndisk: {_disk_size(path)}, \nmemory (loaded): {_df_memory_mb(books_data):.1f} MiB, \n{books_data.shape}, \n{books_data.columns.tolist()}\n, \n{books_data}\n")
        
        del books_data


if __name__ == "__main__":
    main()