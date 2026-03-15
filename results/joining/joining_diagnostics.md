# Diagnostic Task 0 — Préparation des données (P2)

- generated_at: 2026-03-15T15:28:42

## A. Réutilisation du sous-ensemble de travail
- note: `P2 réutilise les sous-ensembles P1 (active/temporal, filtered + splits).`
- methodological_note: `Aucun nouvel échantillonnage massif du corpus complet n'est effectué; les jeux issus de P1 sont réexploités pour la fusion avec meta_Books.`

## B. Documentation des sources

### active_pre_split
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
- columns names: '['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']

### active_post_split_union
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
- columns names: '['asin', 'helpful_vote', 'images', 'parent_asin', 'rating', 'text', 'timestamp', 'title', 'user_id', 'verified_purchase']

### temporal_pre_split
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
- columns names: '['rating', 'title', 'text', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']

### temporal_post_split_union
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
- columns names: '['asin', 'helpful_vote', 'parent_asin', 'rating', 'text', 'timestamp', 'title', 'user_id', 'verified_purchase']

### metadata
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
- columns names: '['main_category', 'title', 'subtitle', 'author', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'videos', 'store', 'categories', 'details', 'parent_asin', 'bought_together']

## C. Vérifications schéma et clés (`parent_asin`)

### metadata
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

### active_pre_split
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

### active_post_split_union
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

### temporal_pre_split
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

### temporal_post_split_union
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## D. Qualité de jointure via `parent_asin`

### active_pre_split
- nb_parent_asin_communs: `45073`
- nb_interactions_jointes / nb_interactions_totales: `508878 / 508878`
- ratio_interactions_jointes: `1.0`
- nb_items_avec_meta / nb_items_totaux: `45073 / 45073`
- ratio_items_avec_meta: `1.0`
- interactions_non_jointes_si_inner_join: `0`
- items_sans_meta: `0`

### active_post_split_union
- nb_parent_asin_communs: `45073`
- nb_interactions_jointes / nb_interactions_totales: `497931 / 497931`
- ratio_interactions_jointes: `1.0`
- nb_items_avec_meta / nb_items_totaux: `45073 / 45073`
- ratio_items_avec_meta: `1.0`
- interactions_non_jointes_si_inner_join: `0`
- items_sans_meta: `0`

### temporal_pre_split
- nb_parent_asin_communs: `22007`
- nb_interactions_jointes / nb_interactions_totales: `289949 / 289949`
- ratio_interactions_jointes: `1.0`
- nb_items_avec_meta / nb_items_totaux: `22007 / 22007`
- ratio_items_avec_meta: `1.0`
- interactions_non_jointes_si_inner_join: `0`
- items_sans_meta: `0`

### temporal_post_split_union
- nb_parent_asin_communs: `22007`
- nb_interactions_jointes / nb_interactions_totales: `285521 / 285521`
- ratio_interactions_jointes: `1.0`
- nb_items_avec_meta / nb_items_totaux: `22007 / 22007`
- ratio_items_avec_meta: `1.0`
- interactions_non_jointes_si_inner_join: `0`
- items_sans_meta: `0`

## E. Attributs exploitables

### active_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `[]`
- ignored_metadata_cols: `[]`

### active_post_split_union
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `[]`
- ignored_metadata_cols: `[]`

### temporal_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `[]`
- ignored_metadata_cols: `[]`

### temporal_post_split_union
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `[]`
- ignored_metadata_cols: `[]`

## F. Valeurs manquantes et stratégie

### active_pre_split
- parent_asin: missing_count=`0`, missing_pct=`0.0`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: missing_count=`0`, missing_pct=`0.0`, strategy=`remplacer NaN par chaîne vide`
- features: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- description: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- categories: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- author: missing_count=`1580830`, missing_pct=`35.5388`, strategy=`aplatir struct (extraire champ clé en string)`
- details: missing_count=`0`, missing_pct=`0.0`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: missing_count=`1063181`, missing_pct=`23.9015`, strategy=`imputation médiane (ou exclusion si trop manquant)`

### active_post_split_union
- parent_asin: missing_count=`0`, missing_pct=`0.0`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: missing_count=`0`, missing_pct=`0.0`, strategy=`remplacer NaN par chaîne vide`
- features: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- description: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- categories: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- author: missing_count=`1580830`, missing_pct=`35.5388`, strategy=`aplatir struct (extraire champ clé en string)`
- details: missing_count=`0`, missing_pct=`0.0`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: missing_count=`1063181`, missing_pct=`23.9015`, strategy=`imputation médiane (ou exclusion si trop manquant)`

### temporal_pre_split
- parent_asin: missing_count=`0`, missing_pct=`0.0`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: missing_count=`0`, missing_pct=`0.0`, strategy=`remplacer NaN par chaîne vide`
- features: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- description: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- categories: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- author: missing_count=`1580830`, missing_pct=`35.5388`, strategy=`aplatir struct (extraire champ clé en string)`
- details: missing_count=`0`, missing_pct=`0.0`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: missing_count=`1063181`, missing_pct=`23.9015`, strategy=`imputation médiane (ou exclusion si trop manquant)`

### temporal_post_split_union
- parent_asin: missing_count=`0`, missing_pct=`0.0`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: missing_count=`0`, missing_pct=`0.0`, strategy=`remplacer NaN par chaîne vide`
- features: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- description: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- categories: missing_count=`0`, missing_pct=`0.0`, strategy=`joindre éléments en string, vide si absent`
- author: missing_count=`1580830`, missing_pct=`35.5388`, strategy=`aplatir struct (extraire champ clé en string)`
- details: missing_count=`0`, missing_pct=`0.0`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: missing_count=`0`, missing_pct=`0.0`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: missing_count=`1063181`, missing_pct=`23.9015`, strategy=`imputation médiane (ou exclusion si trop manquant)`

## G. Jeux de données finaux

### active_pre_split
- path: `data/joining/active_pre_split_joined.parquet`
- rows: `508878`
- cols: `14`

### active_post_split_union
- path: `data/joining/active_post_split_union_joined.parquet`
- rows: `497931`
- cols: `14`

### temporal_pre_split
- path: `data/joining/temporal_pre_split_joined.parquet`
- rows: `289949`
- cols: `14`

### temporal_post_split_union
- path: `data/joining/temporal_post_split_union_joined.parquet`
- rows: `285521`
- cols: `14`
