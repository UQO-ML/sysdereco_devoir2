"""
Module : metrics
Rôle : Calcul des métriques d’évaluation des recommandations (précision, rappel, NDCG, etc.).

Structure prévue :
--------------------
1. Imports (numpy, éventuellement sklearn pour des helpers).
2. Fonctions par métrique :
   - precision_at_k(recommended, relevant, k) : précision@k (nombre de pertinents dans les k premiers / k).
   - recall_at_k(recommended, relevant, k) : rappel@k (nombre de pertinents dans les k premiers / total pertinent).
   - ndcg_at_k(recommended, relevant, k) : NDCG@k (qualité du classement avec décroissance).
   - mean_metric(metric_fn, recommendations, ground_truth, k) : moyenne de la métrique sur tous les utilisateurs.
3. evaluate_all(recommendations, ground_truth, k_list) : retourne un dict { "precision@5": ..., "recall@5": ..., "ndcg@10": ..., … }.
4. Entrées : recommendations = dict user_id -> list of item_id ; ground_truth = dict user_id -> set of item_id (pertinents).
5. Sortie : dict de métriques (à sauvegarder ou afficher dans run_pipeline).
"""

# TODO: Implémenter les métriques selon l’énoncé et les jeux de données.
