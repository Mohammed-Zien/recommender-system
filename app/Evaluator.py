import numpy as np

class Evaluator:
    def __init__(self, recommender, user_clicks, k=10):
        self.recommender = recommender
        self.user_clicks = user_clicks
        self.k = k

    def precision_at_k(self, actual, predicted):
        if not actual:
            return 0
        
        predicted = predicted[:self.k]
        hits = len(set(actual) & set(predicted))
        return hits/self.k
    
    def recall_at_k(self, actual, predicted):
        if not actual:
            return 0
        
        predicted = predicted[:self.k]
        hits = len(set(actual) & set(predicted))
        return hits / len(actual)
    
    def ndcg_at_k(self, actual, predicted):
        predicted = predicted[:self.k]
        dcg = 0.0
        idcg = 0.0

        for idx, p in enumerate(predicted):
            if p in actual:
                dcg += 1 / np.log2(idx + 2)  # +2 because idx starts from 0

        for idx in range(min(len(actual), self.k)):
            idcg += 1 / np.log2(idx + 2)

        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_user(self, user_id):
        """
        Evaluate recommendations for a single user.
        """
        actual_clicks = self.user_clicks.get(user_id, [])
        
        if not actual_clicks:
            return None  # No data for this user

        # Assume last click is the "query" article
        last_clicked = actual_clicks[-1]

        # All earlier clicks are the "ground truth" to predict
        ground_truth = actual_clicks[:-1]

        if not ground_truth:
            return None

        # Get recommendations
        recs = self.recommender.recommend(last_clicked)
        recommended_news_ids = [r["News ID"] for r in recs]

        # Calculate metrics
        precision = round(self.precision_at_k(ground_truth, recommended_news_ids), 4)
        recall = round(self.recall_at_k(ground_truth, recommended_news_ids), 4)
        ndcg = round(self.ndcg_at_k(ground_truth, recommended_news_ids), 4)

        return {"precision": precision, "recall": recall, "ndcg": ndcg}
    
    def evaluate_all(self):
        """
        Evaluate across all users.
        Returns average metrics.
        """
        all_metrics = []

        for user_id in self.user_clicks.keys():
            metrics = self.evaluate_user(user_id)
            if metrics:
                all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Average the results
        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_ndcg = np.mean([m["ndcg"] for m in all_metrics])

        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_ndcg": avg_ndcg
        }