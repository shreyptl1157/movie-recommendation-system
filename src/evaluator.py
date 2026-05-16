import numpy as np
import logging
from typing import Tuple, List, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluation metrics for recommendation systems."""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            RMSE score
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error.
        
        Args:
            y_true: True ratings
            y_pred: Predicted ratings
            
        Returns:
            MAE score
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def precision_at_k(y_true: np.ndarray, y_pred: List[int], k: int = 10) -> float:
        """Calculate Precision@K.
        
        Args:
            y_true: Relevant items
            y_pred: Predicted items ranked by score
            k: Number of top items
            
        Returns:
            Precision@K score
        """
        if len(y_pred) == 0:
            return 0.0
        
        k = min(k, len(y_pred))
        top_k = set(y_pred[:k])
        relevant = set(y_true)
        
        if len(top_k) == 0:
            return 0.0
        
        return float(len(top_k & relevant) / len(top_k))
    
    @staticmethod
    def recall_at_k(y_true: np.ndarray, y_pred: List[int], k: int = 10) -> float:
        """Calculate Recall@K.
        
        Args:
            y_true: Relevant items
            y_pred: Predicted items ranked by score
            k: Number of top items
            
        Returns:
            Recall@K score
        """
        if len(y_true) == 0:
            return 0.0
        
        k = min(k, len(y_pred))
        top_k = set(y_pred[:k])
        relevant = set(y_true)
        
        if len(relevant) == 0:
            return 0.0
        
        return float(len(top_k & relevant) / len(relevant))
    
    @staticmethod
    def ndcg_at_k(y_true: np.ndarray, y_pred: List[Tuple[int, float]], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            y_true: Relevant items
            y_pred: Predicted items with scores [(item_id, score), ...]
            k: Number of top items
            
        Returns:
            NDCG@K score
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        k = min(k, len(y_pred))
        relevant_set = set(y_true)
        
        # Calculate DCG
        dcg = 0.0
        for i, (item_id, _) in enumerate(y_pred[:k]):
            if item_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because ranking starts at 1
        
        # Calculate IDCG
        idcg = 0.0
        for i in range(min(k, len(relevant_set))):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return float(dcg / idcg)
    
    @staticmethod
    def map_at_k(y_true: np.ndarray, y_pred: List[int], k: int = 10) -> float:
        """Calculate Mean Average Precision@K.
        
        Args:
            y_true: Relevant items
            y_pred: Predicted items ranked by score
            k: Number of top items
            
        Returns:
            MAP@K score
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return 0.0
        
        k = min(k, len(y_pred))
        relevant_set = set(y_true)
        
        score = 0.0
        num_hits = 0.0
        
        for i, item_id in enumerate(y_pred[:k]):
            if item_id in relevant_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        if num_hits == 0:
            return 0.0
        
        return float(score / min(k, len(relevant_set)))
    
    @staticmethod
    def coverage(predictions: Dict[int, List[int]], num_items: int) -> float:
        """Calculate catalog coverage.
        
        Args:
            predictions: Dictionary mapping user_id to predicted item_ids
            num_items: Total number of items in catalog
            
        Returns:
            Coverage ratio (0-1)
        """
        all_predicted = set()
        for items in predictions.values():
            all_predicted.update(items)
        
        return float(len(all_predicted) / num_items)
    
    @staticmethod
    def diversity(
        predictions: Dict[int, List[Tuple[int, float]]],
        similarity_matrix: np.ndarray
    ) -> float:
        """Calculate diversity of recommendations.
        
        Args:
            predictions: Dictionary mapping user_id to recommended items with scores
            similarity_matrix: Item similarity matrix
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        diversities = []
        
        for items_with_scores in predictions.values():
            if len(items_with_scores) < 2:
                continue
            
            items = [item_id for item_id, _ in items_with_scores]
            
            # Calculate average pairwise dissimilarity
            dissimilarities = []
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    if items[i] < similarity_matrix.shape[0] and items[j] < similarity_matrix.shape[1]:
                        sim = similarity_matrix[items[i], items[j]]
                        dissim = 1 - sim
                        dissimilarities.append(dissim)
            
            if dissimilarities:
                diversities.append(np.mean(dissimilarities))
        
        if not diversities:
            return 0.0
        
        return float(np.mean(diversities))
    
    @staticmethod
    def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate model with multiple metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: List of metric names to compute
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['rmse', 'mae']
        
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = Evaluator.rmse(y_true, y_pred)
        if 'mae' in metrics:
            results['mae'] = Evaluator.mae(y_true, y_pred)
        
        return results
