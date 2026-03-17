# Diagnostic Task 0 — Préparation des données (P2)

- generated_at: 2026-03-17T13:18:30

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
- columns names: `['rating', 'title', 'text', 'images', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']`

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
- columns names: `['rating', 'title', 'text', 'asin', 'parent_asin', 'user_id', 'timestamp', 'helpful_vote', 'verified_purchase']`

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
- columns names: `['main_category', 'title', 'subtitle', 'author', 'average_rating', 'rating_number', 'features', 'description', 'price', 'images', 'videos', 'store', 'categories', 'details', 'parent_asin', 'bought_together']`


### active_pre_split
- chemin de sauvegarde: `data/joining/active_pre_split_clean_joined.parquet`

### temporal_pre_split
- chemin de sauvegarde: `data/joining/temporal_pre_split_clean_joined.parquet`


### active_pre_split
- chemin de sauvegarde: `data/joining/active_pre_split_clean_joined.parquet`

### temporal_pre_split
- chemin de sauvegarde: `data/joining/temporal_pre_split_clean_joined.parquet`

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

### temporal_pre_split
- ok: `True`
- missing_required: `[]`
- missing_parent_asin_count: `0`
- missing_parent_asin_pct: `0.0`
- coercion_warning: `None`
- warnings: `[]`

## C2. Détection de doublons

### metadata
- n_rows: `4448181`
- doublons exacts: `0` (0.0%)
- doublons parent_asin: `0` (0.0%)

### active_pre_split
- n_rows: `508878`
- doublons exacts: `10756` (2.1137%)
- doublons (user_id, parent_asin): `10947` (2.1512%)

### temporal_pre_split
- n_rows: `289949`
- doublons exacts: `4428` (1.5272%)
- doublons (user_id, parent_asin): `4428` (1.5272%)

## C3. Validation des valeurs (rating, timestamp)

### active_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.2477`, median=`5.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`int64`, min=`1996-12-17 04:37:25`, max=`2023-09-03 20:10:49.845000029`, non convertibles=`0`, ok=`False`
  -   569 timestamps avant 2000

### temporal_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.4423`, median=`5.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`datetime64[ms]`, min=`2020-01-01 00:17:29.935000`, max=`2023-09-03 20:10:49.845000`, non convertibles=`0`, ok=`True`

## C3. Validation des valeurs (rating, timestamp)

### active_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.2477`, median=`5.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`int64`, min=`1996-12-17 04:37:25`, max=`2023-09-03 20:10:49.845000029`, non convertibles=`0`, ok=`False`
  -   569 timestamps avant 2000

### temporal_pre_split
- rating: min=`1.0`, max=`5.0`, mean=`4.4423`, median=`5.0`, hors intervalle=`0` (0.0%), ok=`True`
- timestamp: dtype=`datetime64[ms]`, min=`2020-01-01 00:17:29.935000`, max=`2023-09-03 20:10:49.845000`, non convertibles=`0`, ok=`True`

## D. Qualité de jointure via `parent_asin`

### active_pre_split
- nb_parent_asin_communs: `45073`
- nb_interactions_jointes / nb_interactions_totales: `508878 / 508878`
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

## E. Attributs exploitables

### active_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp', 'text']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['title', 'images', 'asin', 'helpful_vote', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

### temporal_pre_split
- interactions_kept: `['user_id', 'parent_asin', 'rating', 'timestamp', 'text']`
- metadata_text_kept: `['title', 'subtitle', 'features', 'description', 'categories', 'author', 'details']`
- metadata_struct_kept: `['average_rating', 'rating_number', 'price']`
- ignored_interactions_cols: `['title', 'asin', 'helpful_vote', 'verified_purchase']`
- ignored_metadata_cols: `['main_category', 'images', 'videos', 'store', 'bought_together']`

## F. Valeurs manquantes et stratégie

### active_pre_split

#### Interactions brutes

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0006% | 0.0006% | au cas par cas / hors périmètre | — |

#### Métadonnées globales

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 10.7274% | 0.0% | 10.7274% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 11.6973% | 11.6973% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 58.6279% | 58.6279% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 11.8852% | 11.8852% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| author | catégorielle (struct imbriqué) | 35.5388% | 0.0% | 35.5388% | aplatir struct (extraire champ clé en string) | Structure imbriquée → extraction de author_name ; vide si auteur inconnu, impact limité. |
| details | catégorielle (struct imbriqué) | 0.0% | 0.0% | 0.0% | aplatir struct (extraire champ clé en string) | Structure imbriquée → publisher/language ; vide tolérable, attributs secondaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 23.9015% | 0.0% | 23.9015% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |

#### Sous-ensemble joint

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0004% | 0.0004% | au cas par cas / hors périmètre | — |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 0.0% | 8.552% | 8.552% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 0.4314% | 0.4314% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 11.9097% | 11.9097% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 0.1219% | 0.1219% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |
| author_name | catégorielle (extraite de struct) | 0.0% | 4.2657% | 4.2657% | au cas par cas / hors périmètre | Extrait de struct author ; vide si auteur inconnu, impact limité. |
| details_publisher | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, attribut catégoriel secondaire. |
| details_language | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, quasi-constant ('English'). |

### temporal_pre_split

#### Interactions brutes

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0017% | 0.0017% | au cas par cas / hors périmètre | — |

#### Métadonnées globales

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 10.7274% | 0.0% | 10.7274% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 11.6973% | 11.6973% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 58.6279% | 58.6279% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 11.8852% | 11.8852% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| author | catégorielle (struct imbriqué) | 35.5388% | 0.0% | 35.5388% | aplatir struct (extraire champ clé en string) | Structure imbriquée → extraction de author_name ; vide si auteur inconnu, impact limité. |
| details | catégorielle (struct imbriqué) | 0.0% | 0.0% | 0.0% | aplatir struct (extraire champ clé en string) | Structure imbriquée → publisher/language ; vide tolérable, attributs secondaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 23.9015% | 0.0% | 23.9015% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |

#### Sous-ensemble joint

| colonne | type | % NaN | % vide | % effectif | stratégie | justification |
|---------|------|-------|--------|------------|-----------|---------------|
| user_id | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé primaire interactions — ligne non identifiable sans user_id. |
| parent_asin | clé (identifiant) | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Clé de jointure obligatoire — toute ligne sans parent_asin est inutilisable. |
| rating | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (variable cible) | Variable cible du système de recommandation — ligne sans note exclue. |
| timestamp | numérique | 0.0% | 0.0% | 0.0% | supprimer lignes incomplètes (clé obligatoire) | Nécessaire au split temporel train/test — ligne inutilisable sans date. |
| text | clé (identifiant) | 0.0% | 0.0018% | 0.0018% | au cas par cas / hors périmètre | — |
| title | textuelle (scalaire) | 0.0% | 0.0% | 0.0% | remplacer NaN par chaîne vide | Contenu textuel principal pour TF-IDF/embeddings ; chaîne vide tolérable car concaténé avec description. |
| subtitle | textuelle (scalaire) | 0.0% | 10.3863% | 10.3863% | remplacer NaN par chaîne vide | Complément textuel mineur ; chaîne vide acceptable, faible impact sur la représentation. |
| features | textuelle (liste) | 0.0% | 1.4192% | 1.4192% | joindre éléments en string, vide si absent | Points clés marketing ; vide tolérable, données complémentaires au content-based. |
| description | textuelle (liste) | 0.0% | 33.3072% | 33.3072% | joindre éléments en string, vide si absent | Contenu sémantique riche pour content-based filtering ; vide tolérable car compensé par title+features. |
| categories | textuelle (liste) | 0.0% | 0.3797% | 0.3797% | joindre éléments en string, vide si absent | Taxonomie Amazon pour filtrage par genre ; vide tolérable, données complémentaires. |
| average_rating | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Signal de popularité agrégé ; NaN rare, imputation non nécessaire sauf si >5%. |
| rating_number | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Volume de notes ; NaN rare, indicateur de confiance secondaire. |
| price | numérique | 0.0% | 0.0% | 0.0% | imputation médiane (ou exclusion si trop manquant) | Distribution asymétrique, ~24% manquant → médiane plus robuste que la moyenne. |
| author_name | catégorielle (extraite de struct) | 0.0% | 5.8864% | 5.8864% | au cas par cas / hors périmètre | Extrait de struct author ; vide si auteur inconnu, impact limité. |
| details_publisher | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, attribut catégoriel secondaire. |
| details_language | catégorielle (extraite de struct) | 0.0% | 0.0% | 0.0% | au cas par cas / hors périmètre | Extrait de struct details ; vide tolérable, quasi-constant ('English'). |


## F2. Qualité des champs textuels

### active_pre_split
- title: avg_len=`39.7`, median_len=`36`, vides=`0` (0.0%), HTML=`0` (0.0%)
- subtitle: avg_len=`21.8`, median_len=`24`, vides=`43515` (8.5512%), HTML=`0` (0.0%)
- description: avg_len=`8662.2`, median_len=`3503`, vides=`61009` (11.9889%), HTML=`3248` (0.6383%)
- categories: avg_len=`50.5`, median_len=`49`, vides=`762` (0.1497%), HTML=`0` (0.0%)
- features: avg_len=`1272.9`, median_len=`1145`, vides=`2276` (0.4473%), HTML=`287` (0.0564%)
- author_name: avg_len=`12.7`, median_len=`13`, vides=`22076` (4.3382%), HTML=`0` (0.0%)
- details_publisher: avg_len=`38.3`, median_len=`39`, vides=`0` (0.0%), HTML=`10` (0.002%)
- details_language: avg_len=`6.9`, median_len=`7`, vides=`0` (0.0%), HTML=`0` (0.0%)

### temporal_pre_split
- title: avg_len=`46.2`, median_len=`39`, vides=`0` (0.0%), HTML=`0` (0.0%)
- subtitle: avg_len=`18.7`, median_len=`14`, vides=`30133` (10.3925%), HTML=`0` (0.0%)
- description: avg_len=`3824.1`, median_len=`1347`, vides=`96703` (33.3517%), HTML=`445` (0.1535%)
- categories: avg_len=`50.4`, median_len=`49`, vides=`1127` (0.3887%), HTML=`0` (0.0%)
- features: avg_len=`1361.6`, median_len=`1245`, vides=`4126` (1.423%), HTML=`178` (0.0614%)
- author_name: avg_len=`12.3`, median_len=`12`, vides=`17155` (5.9166%), HTML=`0` (0.0%)
- details_publisher: avg_len=`31.8`, median_len=`34`, vides=`0` (0.0%), HTML=`5` (0.0017%)
- details_language: avg_len=`6.9`, median_len=`7`, vides=`0` (0.0%), HTML=`0` (0.0%)

## F3. Nettoyage appliqué (avant / après)

### active_pre_split

| métrique | avant | après | delta |
|----------|-------|-------|-------|
| lignes | 508878 | 497931 | −10947 |
| items | 45073 | 45073 | −0 |
| users | 10847 | 10847 | −0 |

**Raisons de suppression :**
- `missing_key_cols`: 0 lignes
- `interaction_duplicates`: 10947 lignes

### temporal_pre_split

| métrique | avant | après | delta |
|----------|-------|-------|-------|
| lignes | 289949 | 285521 | −4428 |
| items | 22007 | 22007 | −0 |
| users | 6739 | 6739 | −0 |

**Raisons de suppression :**
- `missing_key_cols`: 0 lignes
- `interaction_duplicates`: 4428 lignes

## F4. Vérifications post-nettoyage

### active_pre_split

- Doublons résiduels `(user_id, parent_asin)`: **0** — OK
- Distribution rating post-nettoyage: min=1.0, max=5.0, mean=4.247 — OK
- Intégrité parent_asin: 45073 → 45073 items — OK
- NaN résiduel sur clés: {'user_id': 0, 'parent_asin': 0, 'rating': 0, 'timestamp': 0} — OK

### temporal_pre_split

- Doublons résiduels `(user_id, parent_asin)`: **0** — OK
- Distribution rating post-nettoyage: min=1.0, max=5.0, mean=4.4439 — OK
- Intégrité parent_asin: 22007 → 22007 items — OK
- NaN résiduel sur clés: {'user_id': 0, 'parent_asin': 0, 'rating': 0, 'timestamp': 0} — OK

## G. Jeux de données finaux

### active_pre_split
- path: `data/joining/active_pre_split_clean_joined.parquet`
- rows: `497931`
- cols: `16`

### temporal_pre_split
- path: `data/joining/temporal_pre_split_clean_joined.parquet`
- rows: `285521`
- cols: `16`

## H. Usage des colonnes par tâche

### Représentation de contenu (Tâches 0-2)
- `title`: TF-IDF / embeddings — titre du livre
- `description`: TF-IDF / embeddings — description éditoriale
- `categories`: Encodage catégoriel / multi-hot — taxonomie Amazon
- `features`: TF-IDF — points clés marketing
- `author_name`: Encodage catégoriel — filtrage par auteur

### Variables explicatives (Tâche 3)
- `average_rating`: Variable continue — popularité agrégée de l'item
- `rating_number`: Variable continue — volume de notes (confiance)
- `price`: Variable continue — prix (imputé médiane)
- `details_publisher`: Variable catégorielle — éditeur
- `details_language`: Variable catégorielle — langue
- `author_name`: Variable catégorielle — auteur (partagé avec contenu)

## I. Split temporel train / test

### active_pre_split

**Méthode** : `temporal_per_user`

**Règle** : n_test = max(1, floor(n_total × 0.2)), borné à n_total − 1. Utilisateurs avec <3 interactions → train uniquement.

**Justification** : Split temporel : on entraîne sur le passé, on évalue sur le futur. Simule un scénario de déploiement réaliste. Utilisateurs avec <3 interactions → train only (pas assez d'historique pour construire un profil ET tester).

| métrique | train | test |
|----------|-------|------|
| interactions | 402,327 | 95,604 |
| utilisateurs | 10,847 | 10,844 |
| items | 44,402 | 31,466 |
| ratio effectif | 80.80% | 19.20% |

- Users train-only (< 3 interactions) : **3**

- Chaque user test ∈ train : **OK** (violateurs : 0)
- Items test-only : **671** (2.1325%)
  - Items test-only ont une représentation metadata (TF-IDF sur title/description) même sans interaction train — acceptable pour un content-based system.

- `data/joining/active_pre_split/train_interactions.parquet`
- `data/joining/active_pre_split/test_interactions.parquet`

### temporal_pre_split

**Méthode** : `temporal_per_user`

**Règle** : n_test = max(1, floor(n_total × 0.2)), borné à n_total − 1. Utilisateurs avec <3 interactions → train uniquement.

**Justification** : Split temporel : on entraîne sur le passé, on évalue sur le futur. Simule un scénario de déploiement réaliste. Utilisateurs avec <3 interactions → train only (pas assez d'historique pour construire un profil ET tester).

| métrique | train | test |
|----------|-------|------|
| interactions | 230,938 | 54,583 |
| utilisateurs | 6,739 | 6,737 |
| items | 21,224 | 13,145 |
| ratio effectif | 80.88% | 19.12% |

- Users train-only (< 3 interactions) : **2**

- Chaque user test ∈ train : **OK** (violateurs : 0)
- Items test-only : **783** (5.9566%)
  - Items test-only ont une représentation metadata (TF-IDF sur title/description) même sans interaction train — acceptable pour un content-based system.

- `data/joining/temporal_pre_split/train_interactions.parquet`
- `data/joining/temporal_pre_split/test_interactions.parquet`
