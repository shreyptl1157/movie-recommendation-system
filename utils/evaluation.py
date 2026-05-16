import numpy as np
from typing import List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(actuals, predictions))

def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean Absolute Error."""
    return mean_absolute_error(actuals, predictions)

def precision_at_k(predictions: List[int], actuals: List[int], k: int = 10) -> float:
    """Precision@K metric."""
    if not predictions or k <= 0:
        return 0.0
    
    predictions_k = predictions[:k]
    common = len(set(predictions_k) & set(actuals))
    return common / k

def recall_at_k(predictions: List[int], actuals: List[int], k: int = 10) -> float:
    """Recall@K metric."""
    if not actuals:
        return 0.0
    
    predictions_k = predictions[:k]
    common = len(set(predictions_k) & set(actuals))
    return common / len(actuals)

def ndcg_at_k(predictions: List[Tuple[int, float]], actuals: List[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain."""
    if not predictions or not actuals:
        return 0.0
    
    dcg = 0.0
    for i, (pred_id, _) in enumerate(predictions[:k], 1):
        if pred_id in actuals:
            dcg += 1 / np.log2(i + 1)
    
    # Ideal DCG
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(actuals), k) + 1))
    
    return dcg / idcg if idcg > 0 else 0.0

def coverage(recommendations: dict, total_items: int) -> float:
    """Coverage metric - percentage of catalog recommended."""
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update([item_id for item_id, _ in recs])
    
    return len(recommended_items) / total_items if total_items > 0 else 0.0

def diversity(recommendations: List[Tuple[int, float]], item_similarity_matrix) -> float:
    """Diversity metric - average dissimilarity in recommendations."""
    if len(recommendations) < 2:
        return 0.0
    
    dissimilarities = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item_i = recommendations[i][0]
            item_j = recommendations[j][0]
            # Assume similarity between 0 and 1
            similarity = item_similarity_matrix.get((item_i, item_j), 0.5)
            dissimilarities.append(1 - similarity)
    
    return np.mean(dissimilarities) if dissimilarities else 0.0
