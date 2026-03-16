# Diagnostic Task 0 — Préparation des données (P2)

- generated_at: 2026-03-15T17:02:24

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
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['title', 'text', 'images', 'asin', 'helpful_vote', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

### active_post_split_union
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['asin', 'helpful_vote', 'images', 'text', 'title', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

### temporal_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['title', 'text', 'asin', 'helpful_vote', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

### temporal_post_split_union
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['asin', 'helpful_vote', 'text', 'title', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

## F. Valeurs manquantes et stratégie

### active_pre_split
#### Meta global (4.4M)
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`10.7274%`, vide=`0.0%`, effectif=`10.7274%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`11.6973%`, effectif=`11.6973%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`58.6279%`, effectif=`58.6279%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`11.8852%`, effectif=`11.8852%`, strategy=`joindre éléments en string, vide si absent`
- author: NaN=`35.5388%`, vide=`0.0%`, effectif=`35.5388%`, strategy=`aplatir struct (extraire champ clé en string)`
- details: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`23.9015%`, vide=`0.0%`, effectif=`23.9015%`, strategy=`imputation médiane (ou exclusion si trop manquant)`

#### Sous-ensemble joint
- user_id: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- timestamp: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`0.0%`, vide=`8.5512%`, effectif=`8.5512%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`0.4473%`, effectif=`0.4473%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`11.9889%`, effectif=`11.9889%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`0.1497%`, effectif=`0.1497%`, strategy=`joindre éléments en string, vide si absent`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- author_name: NaN=`0.0%`, vide=`4.3382%`, effectif=`4.3382%`, strategy=`au cas par cas / hors périmètre`
- details_publisher: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- details_language: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`

### active_post_split_union
#### Meta global (4.4M)
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`10.7274%`, vide=`0.0%`, effectif=`10.7274%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`11.6973%`, effectif=`11.6973%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`58.6279%`, effectif=`58.6279%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`11.8852%`, effectif=`11.8852%`, strategy=`joindre éléments en string, vide si absent`
- author: NaN=`35.5388%`, vide=`0.0%`, effectif=`35.5388%`, strategy=`aplatir struct (extraire champ clé en string)`
- details: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`23.9015%`, vide=`0.0%`, effectif=`23.9015%`, strategy=`imputation médiane (ou exclusion si trop manquant)`

#### Sous-ensemble joint
- user_id: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- timestamp: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`0.0%`, vide=`8.552%`, effectif=`8.552%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`0.4314%`, effectif=`0.4314%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`11.9097%`, effectif=`11.9097%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`0.1219%`, effectif=`0.1219%`, strategy=`joindre éléments en string, vide si absent`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- author_name: NaN=`0.0%`, vide=`4.2657%`, effectif=`4.2657%`, strategy=`au cas par cas / hors périmètre`
- details_publisher: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- details_language: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`

### temporal_pre_split
#### Meta global (4.4M)
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`10.7274%`, vide=`0.0%`, effectif=`10.7274%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`11.6973%`, effectif=`11.6973%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`58.6279%`, effectif=`58.6279%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`11.8852%`, effectif=`11.8852%`, strategy=`joindre éléments en string, vide si absent`
- author: NaN=`35.5388%`, vide=`0.0%`, effectif=`35.5388%`, strategy=`aplatir struct (extraire champ clé en string)`
- details: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`23.9015%`, vide=`0.0%`, effectif=`23.9015%`, strategy=`imputation médiane (ou exclusion si trop manquant)`

#### Sous-ensemble joint
- user_id: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- timestamp: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`0.0%`, vide=`10.3925%`, effectif=`10.3925%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`1.423%`, effectif=`1.423%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`33.3517%`, effectif=`33.3517%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`0.3887%`, effectif=`0.3887%`, strategy=`joindre éléments en string, vide si absent`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- author_name: NaN=`0.0%`, vide=`5.9166%`, effectif=`5.9166%`, strategy=`au cas par cas / hors périmètre`
- details_publisher: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- details_language: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`

### temporal_post_split_union
#### Meta global (4.4M)
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`10.7274%`, vide=`0.0%`, effectif=`10.7274%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`11.6973%`, effectif=`11.6973%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`58.6279%`, effectif=`58.6279%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`11.8852%`, effectif=`11.8852%`, strategy=`joindre éléments en string, vide si absent`
- author: NaN=`35.5388%`, vide=`0.0%`, effectif=`35.5388%`, strategy=`aplatir struct (extraire champ clé en string)`
- details: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`aplatir struct (extraire champ clé en string)`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`23.9015%`, vide=`0.0%`, effectif=`23.9015%`, strategy=`imputation médiane (ou exclusion si trop manquant)`

#### Sous-ensemble joint
- user_id: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- parent_asin: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- timestamp: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`supprimer lignes incomplètes (clé obligatoire)`
- title: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`remplacer NaN par chaîne vide`
- subtitle: NaN=`0.0%`, vide=`10.3863%`, effectif=`10.3863%`, strategy=`remplacer NaN par chaîne vide`
- features: NaN=`0.0%`, vide=`1.4192%`, effectif=`1.4192%`, strategy=`joindre éléments en string, vide si absent`
- description: NaN=`0.0%`, vide=`33.3072%`, effectif=`33.3072%`, strategy=`joindre éléments en string, vide si absent`
- categories: NaN=`0.0%`, vide=`0.3797%`, effectif=`0.3797%`, strategy=`joindre éléments en string, vide si absent`
- average_rating: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- rating_number: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- price: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`imputation médiane (ou exclusion si trop manquant)`
- author_name: NaN=`0.0%`, vide=`5.8864%`, effectif=`5.8864%`, strategy=`au cas par cas / hors périmètre`
- details_publisher: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`
- details_language: NaN=`0.0%`, vide=`0.0%`, effectif=`0.0%`, strategy=`au cas par cas / hors périmètre`

## G. Jeux de données finaux

### active_pre_split
- path: `data/joining/active_pre_split_joined.parquet`
- rows: `508878`
- cols: `15`

### active_post_split_union
- path: `data/joining/active_post_split_union_joined.parquet`
- rows: `497931`
- cols: `15`

### temporal_pre_split
- path: `data/joining/temporal_pre_split_joined.parquet`
- rows: `289949`
- cols: `15`

### temporal_post_split_union
- path: `data/joining/temporal_post_split_union_joined.parquet`
- rows: `285521`
- cols: `15`
