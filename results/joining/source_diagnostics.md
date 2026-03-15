# Source Diagnostics

- generated_at: 2026-03-14T22:36:51

## active_pre_split
- stage: `pre_split`
- variant: `active`
- role: `interactions`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `508878`
- cols: `10`
- size_bytes: `318642272`
- paths: `['data/processed/sample-active-users/active_users_filtered.parquet']`

## active_post_split_union
- stage: `post_split`
- variant: `active`
- role: `interactions`
- kind: `union`
- exists: `True`
- format: `parquet_union`
- rows: `497931`
- cols: `10`
- size_bytes: `319440598`
- paths: `['data/processed/sample-active-users/splits/train.parquet', 'data/processed/sample-active-users/splits/test.parquet']`

## temporal_pre_split
- stage: `pre_split`
- variant: `temporal`
- role: `interactions`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `289949`
- cols: `9`
- size_bytes: `138036503`
- paths: `['data/processed/sample-temporal/temporal_filtered.parquet']`

## temporal_post_split_union
- stage: `post_split`
- variant: `temporal`
- role: `interactions`
- kind: `union`
- exists: `True`
- format: `parquet_union`
- rows: `285521`
- cols: `9`
- size_bytes: `138832248`
- paths: `['data/processed/sample-temporal/splits/train.parquet', 'data/processed/sample-temporal/splits/test.parquet']`

## metadata
- stage: `raw`
- variant: `meta_books`
- role: `metadata`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `4448181`
- cols: `16`
- size_bytes: `4696897350`
- paths: `['data/raw/parquet/meta_Books.parquet']`

## books_reviews_raw
- stage: `raw`
- variant: `books`
- role: `optional_debug`
- kind: `single`
- exists: `True`
- format: `parquet`
- rows: `29475453`
- cols: `10`
- size_bytes: `5736672372`
- paths: `['data/raw/parquet/Books.parquet']`

# Vérifications schéma et clés

## active_pre_split
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## active_post_split_union
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## temporal_pre_split
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## temporal_post_split_union
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## metadata
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## books_reviews_raw
- ok: `True`
- colonnes_manquantes: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`
