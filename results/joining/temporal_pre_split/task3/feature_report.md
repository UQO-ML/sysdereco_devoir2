# Feature Engineering Report — temporal_pre_split

## Vue d'ensemble

- **Paires totales** : 923,752
- **Positives** : 230,938
- **Négatives** : 692,814
- **Users** : 6,739
- **Items** : 22,007
- **Features** : 28
- **Matrice** : [923752, 28] (98.7 MiB)
- **Temps** : 33.9s

## Distribution des labels

- Seuil de pertinence : rating >= 4.0
- Pertinents (1) : 205,081
- Non pertinents (0) : 718,671

### Ratings

- rating_0 : 692,814
- rating_1 : 1,768
- rating_2 : 4,605
- rating_3 : 19,484
- rating_4 : 68,185
- rating_5 : 136,896

## Groupes LTR

- Groupes : 6,739
- Taille moyenne : 137.1
- Min / Max : 8 / 1632

## Statistiques des features

| Feature | Mean | Std | Min | Max | %Zero | %NaN |
|---------|------|-----|-----|-----|-------|------|
| average_rating | 4.2998 | 0.7585 | 0.0000 | 5.0000 | 2.7 | 0.0 |
| log_rating_number | 6.5233 | 2.3031 | 0.0000 | 13.3211 | 2.7 | 0.0 |
| price | 8.4786 | 7.2822 | 0.0000 | 296.0100 | 27.6 | 0.0 |
| item_nb_interactions_train | 14.8822 | 22.8109 | 0.0000 | 302.0000 | 2.7 | 0.0 |
| item_mean_rating_train | 4.3350 | 0.8378 | 0.0000 | 5.0000 | 2.7 | 0.0 |
| user_nb_interactions | 54.2983 | 49.0822 | 2.0000 | 408.0000 | 0.0 | 0.0 |
| user_mean_rating | 4.4456 | 0.4768 | 1.8571 | 5.0000 | 0.0 | 0.0 |
| user_std_rating | 0.5557 | 0.3216 | 0.0000 | 2.3094 | 11.4 | 0.0 |
| user_mean_helpful | 0.8121 | 2.6463 | 0.0000 | 101.5833 | 2.8 | 0.0 |
| user_verified_ratio | 0.2759 | 0.3518 | 0.0000 | 1.0000 | 27.9 | 0.0 |
| user_genre_diversity | 4.9130 | 3.7510 | 1.0000 | 28.0000 | 0.0 | 0.0 |
| item_mean_helpful | 0.7445 | 2.2980 | 0.0000 | 224.0000 | 29.5 | 0.0 |
| nb_pages | 280.6172 | 154.2802 | 0.0000 | 2744.0000 | 8.5 | 0.0 |
| page_deviation | -36.7300 | 162.5547 | -567.4706 | 2610.4092 | 0.0 | 0.0 |
| pub_year | 1861.3722 | 543.6139 | 0.0000 | 2023.0000 | 7.9 | 0.0 |
| book_age | 161.6278 | 543.6139 | 0.0000 | 2023.0000 | 2.5 | 0.0 |
| era_match | 160.2417 | 543.2858 | 0.0000 | 2023.0000 | 26.7 | 0.0 |
| fmt_ebook | 0.4270 | 0.4946 | 0.0000 | 1.0000 | 57.3 | 0.0 |
| fmt_hardcover | 0.1930 | 0.3947 | 0.0000 | 1.0000 | 80.7 | 0.0 |
| fmt_paperback | 0.2799 | 0.4489 | 0.0000 | 1.0000 | 72.0 | 0.0 |
| fmt_mass_market | 0.0161 | 0.1257 | 0.0000 | 1.0000 | 98.4 | 0.0 |
| cosine_similarity | 0.5731 | 0.0798 | 0.0001 | 0.9995 | 0.0 | 0.0 |
| cosine_ecart | 0.0000 | 0.0789 | -0.5851 | 0.5044 | 0.0 | 0.0 |
| already_read_author | 0.2396 | 0.4268 | 0.0000 | 1.0000 | 76.0 | 0.0 |
| nb_books_same_author | 0.6991 | 2.9494 | 0.0000 | 81.0000 | 76.0 | 0.0 |
| user_avg_rating_author | 1.0652 | 1.9340 | 0.0000 | 5.0000 | 76.0 | 0.0 |
| genre_match | 0.7374 | 0.4400 | 0.0000 | 1.0000 | 26.3 | 0.0 |
| user_nb_genres_explored | 4.9130 | 3.7510 | 1.0000 | 28.0000 | 0.0 | 0.0 |

## Sampling négatif

- N négatifs par positif : 3
- Seed : 42