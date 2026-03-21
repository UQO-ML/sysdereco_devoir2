# Feature Engineering Report — active_pre_split

## Vue d'ensemble

- **Paires totales** : 1,609,308
- **Positives** : 402,327
- **Négatives** : 1,206,981
- **Users** : 10,847
- **Items** : 45,073
- **Features** : 28
- **Matrice** : [1609308, 28] (171.9 MiB)
- **Temps** : 133.9s

## Distribution des labels

- Seuil de pertinence : rating >= 4.0
- Pertinents (1) : 327,510
- Non pertinents (0) : 1,281,798

### Ratings

- rating_0 : 1,206,981
- rating_1 : 8,012
- rating_2 : 16,843
- rating_3 : 49,962
- rating_4 : 120,778
- rating_5 : 206,732

## Groupes LTR

- Groupes : 10,847
- Taille moyenne : 148.4
- Min / Max : 8 / 3856

## Statistiques des features

| Feature | Mean | Std | Min | Max | %Zero | %NaN |
|---------|------|-----|-----|-----|-------|------|
| average_rating | 4.3091 | 0.5292 | 0.0000 | 5.0000 | 1.1 | 0.0 |
| log_rating_number | 7.2456 | 1.9519 | 0.0000 | 13.3244 | 1.1 | 0.0 |
| price | 11.2188 | 13.4919 | 0.0000 | 2000.0000 | 14.3 | 0.0 |
| item_nb_interactions_train | 12.3800 | 20.5630 | 0.0000 | 439.0000 | 1.1 | 0.0 |
| item_mean_rating_train | 4.1834 | 0.6735 | 0.0000 | 5.0000 | 1.1 | 0.0 |
| user_nb_interactions | 81.4014 | 105.9197 | 2.0000 | 964.0000 | 0.0 | 0.0 |
| user_mean_rating | 4.2462 | 0.5023 | 1.1111 | 5.0000 | 0.0 | 0.0 |
| user_std_rating | 0.7595 | 0.3370 | 0.0000 | 2.8284 | 4.9 | 0.0 |
| user_mean_helpful | 2.4533 | 5.7598 | 0.0000 | 215.2273 | 2.7 | 0.0 |
| user_verified_ratio | 0.3653 | 0.3740 | 0.0000 | 1.0000 | 16.2 | 0.0 |
| user_genre_diversity | 6.5090 | 4.7868 | 1.0000 | 30.0000 | 0.0 | 0.0 |
| item_mean_helpful | 2.5745 | 7.9156 | 0.0000 | 677.2000 | 13.7 | 0.0 |
| nb_pages | 338.2404 | 164.3861 | 0.0000 | 5216.0000 | 5.2 | 0.0 |
| page_deviation | -27.1840 | 169.3378 | -935.0000 | 5145.9258 | 0.0 | 0.0 |
| pub_year | 1916.9728 | 428.1548 | 0.0000 | 2023.0000 | 4.8 | 0.0 |
| book_age | 106.0276 | 428.1548 | 0.0000 | 2023.0000 | 0.2 | 0.0 |
| era_match | 100.8030 | 427.1631 | 0.0000 | 2023.0000 | 8.8 | 0.0 |
| fmt_ebook | 0.2755 | 0.4468 | 0.0000 | 1.0000 | 72.5 | 0.0 |
| fmt_hardcover | 0.3158 | 0.4648 | 0.0000 | 1.0000 | 68.4 | 0.0 |
| fmt_paperback | 0.2701 | 0.4440 | 0.0000 | 1.0000 | 73.0 | 0.0 |
| fmt_mass_market | 0.0859 | 0.2802 | 0.0000 | 1.0000 | 91.4 | 0.0 |
| cosine_similarity | 0.6433 | 0.0440 | 0.0062 | 0.9880 | 0.0 | 0.0 |
| cosine_ecart | -0.0000 | 0.0431 | -0.6445 | 0.4491 | 0.0 | 0.0 |
| already_read_author | 0.2494 | 0.4327 | 0.0000 | 1.0000 | 75.1 | 0.0 |
| nb_books_same_author | 0.6322 | 2.1900 | 0.0000 | 66.0000 | 75.1 | 0.0 |
| user_avg_rating_author | 1.0571 | 1.8897 | 0.0000 | 5.0000 | 75.1 | 0.0 |
| genre_match | 0.7643 | 0.4244 | 0.0000 | 1.0000 | 23.6 | 0.0 |
| user_nb_genres_explored | 6.5090 | 4.7868 | 1.0000 | 30.0000 | 0.0 | 0.0 |

## Sampling négatif

- N négatifs par positif : 3
- Seed : 42