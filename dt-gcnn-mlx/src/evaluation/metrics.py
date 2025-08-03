"""Evaluation metrics for classification and embedding tasks."""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


def accuracy(y_true: mx.array, y_pred: mx.array) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score between 0 and 1
    """
    correct = mx.sum(y_true == y_pred)
    total = len(y_true)
    return float(correct) / total


def precision_recall_f1(y_true: mx.array, y_pred: mx.array, 
                       num_classes: Optional[int] = None,
                       average: str = 'macro') -> Dict[str, float]:
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (auto-detected if None)
        average: 'micro', 'macro', or 'weighted'
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if num_classes is None:
        num_classes = max(mx.max(y_true).item(), mx.max(y_pred).item()) + 1
        
    # Convert to numpy for easier computation
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    precisions = []
    recalls = []
    f1_scores = []
    support = []
    
    for class_id in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true_np == class_id) & (y_pred_np == class_id))
        fp = np.sum((y_true_np != class_id) & (y_pred_np == class_id))
        fn = np.sum((y_true_np == class_id) & (y_pred_np != class_id))
        
        # Precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        support.append(np.sum(y_true_np == class_id))
        
    # Compute average
    if average == 'micro':
        # Global calculation
        tp_total = sum((y_true_np == y_pred_np))
        precision = recall = f1 = tp_total / len(y_true_np)
    elif average == 'macro':
        # Unweighted mean
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1_scores)
    elif average == 'weighted':
        # Weighted by support
        total_support = sum(support)
        precision = sum(p * s for p, s in zip(precisions, support)) / total_support
        recall = sum(r * s for r, s in zip(recalls, support)) / total_support
        f1 = sum(f * s for f, s in zip(f1_scores, support)) / total_support
    else:
        # Return per-class metrics
        return {
            'precision': precisions,
            'recall': recalls,
            'f1': f1_scores,
            'support': support
        }
        
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def confusion_matrix(y_true: mx.array, y_pred: mx.array, 
                    num_classes: Optional[int] = None) -> mx.array:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if num_classes is None:
        num_classes = max(mx.max(y_true).item(), mx.max(y_pred).item()) + 1
        
    # Initialize confusion matrix
    cm = mx.zeros((num_classes, num_classes), dtype=mx.int32)
    
    # Convert to numpy for easier indexing
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    # Count occurrences
    for true_label, pred_label in zip(y_true_np, y_pred_np):
        cm_np = np.array(cm)
        cm_np[true_label, pred_label] += 1
        cm = mx.array(cm_np)
        
    return cm


def classification_report(y_true: mx.array, y_pred: mx.array,
                         class_names: Optional[List[str]] = None,
                         digits: int = 4) -> str:
    """
    Generate a text classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional names for each class
        digits: Number of decimal places
        
    Returns:
        Text report as string
    """
    num_classes = max(mx.max(y_true).item(), mx.max(y_pred).item()) + 1
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
        
    # Get per-class metrics
    metrics = precision_recall_f1(y_true, y_pred, num_classes, average=None)
    
    # Build report
    headers = ['precision', 'recall', 'f1-score', 'support']
    rows = []
    
    # Header
    header_fmt = '{:>15}' + '{:>12}' * len(headers)
    rows.append(header_fmt.format('', *headers))
    rows.append('')
    
    # Per-class metrics
    row_fmt = '{:>15}' + '{:>12.{digits}f}' * 3 + '{:>12}'
    
    for i, class_name in enumerate(class_names):
        row = row_fmt.format(
            class_name,
            metrics['precision'][i],
            metrics['recall'][i],
            metrics['f1'][i],
            metrics['support'][i],
            digits=digits
        )
        rows.append(row)
        
    rows.append('')
    
    # Overall metrics
    acc = accuracy(y_true, y_pred)
    macro_avg = precision_recall_f1(y_true, y_pred, num_classes, average='macro')
    weighted_avg = precision_recall_f1(y_true, y_pred, num_classes, average='weighted')
    
    total_support = len(y_true)
    
    rows.append(row_fmt.format(
        'accuracy', acc, acc, acc, total_support, digits=digits
    ).replace(f'{acc:.{digits}f}', '', 2))  # Remove duplicate accuracy values
    
    rows.append(row_fmt.format(
        'macro avg',
        macro_avg['precision'],
        macro_avg['recall'],
        macro_avg['f1'],
        total_support,
        digits=digits
    ))
    
    rows.append(row_fmt.format(
        'weighted avg',
        weighted_avg['precision'],
        weighted_avg['recall'],
        weighted_avg['f1'],
        total_support,
        digits=digits
    ))
    
    return '\n'.join(rows)


def embedding_metrics(embeddings: mx.array, labels: mx.array) -> Dict[str, float]:
    """
    Compute metrics for embedding quality.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        
    Returns:
        Dictionary with embedding quality metrics
    """
    embeddings_np = np.array(embeddings)
    labels_np = np.array(labels)
    
    unique_labels = np.unique(labels_np)
    
    # Compute intra-class and inter-class distances
    intra_distances = []
    inter_distances = []
    
    for label in unique_labels:
        class_embeddings = embeddings_np[labels_np == label]
        
        # Intra-class distances
        for i in range(len(class_embeddings)):
            for j in range(i + 1, len(class_embeddings)):
                dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                intra_distances.append(dist)
                
    # Inter-class distances
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            class1_embeddings = embeddings_np[labels_np == label1]
            class2_embeddings = embeddings_np[labels_np == label2]
            
            # Sample for efficiency
            n_samples = min(20, len(class1_embeddings), len(class2_embeddings))
            
            for e1 in class1_embeddings[:n_samples]:
                for e2 in class2_embeddings[:n_samples]:
                    dist = np.linalg.norm(e1 - e2)
                    inter_distances.append(dist)
                    
    # Compute statistics
    metrics = {
        'intra_class_mean': np.mean(intra_distances) if intra_distances else 0.0,
        'intra_class_std': np.std(intra_distances) if intra_distances else 0.0,
        'inter_class_mean': np.mean(inter_distances) if inter_distances else 0.0,
        'inter_class_std': np.std(inter_distances) if inter_distances else 0.0,
    }
    
    # Separation ratio (higher is better)
    if metrics['intra_class_mean'] > 0:
        metrics['separation_ratio'] = metrics['inter_class_mean'] / metrics['intra_class_mean']
    else:
        metrics['separation_ratio'] = float('inf')
        
    # Silhouette coefficient approximation
    if intra_distances and inter_distances:
        a = np.mean(intra_distances)  # Mean intra-cluster distance
        b = np.mean(inter_distances)  # Mean nearest-cluster distance
        metrics['silhouette_score'] = (b - a) / max(a, b)
    else:
        metrics['silhouette_score'] = 0.0
        
    return metrics


def top_k_accuracy(logits: mx.array, labels: mx.array, k: int = 5) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits
        labels: True labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy
    """
    # Get top-k predictions
    top_k_preds = mx.argsort(logits, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = 0
    for i, label in enumerate(labels):
        if label in top_k_preds[i]:
            correct += 1
            
    return correct / len(labels)


def mean_average_precision(embeddings: mx.array, labels: mx.array, k: int = 10) -> float:
    """
    Compute mean average precision for retrieval tasks.
    
    Args:
        embeddings: Embedding vectors
        labels: Class labels
        k: Number of retrieved items to consider
        
    Returns:
        Mean average precision score
    """
    n_samples = len(embeddings)
    ap_scores = []
    
    for i in range(n_samples):
        query_embedding = embeddings[i:i+1]
        query_label = labels[i]
        
        # Compute distances to all other samples
        distances = mx.sqrt(mx.sum((embeddings - query_embedding) ** 2, axis=1))
        
        # Sort by distance (excluding self)
        sorted_indices = mx.argsort(distances)[1:k+1]
        retrieved_labels = labels[sorted_indices]
        
        # Compute average precision
        relevant = (retrieved_labels == query_label)
        relevant_np = np.array(relevant)
        
        if np.sum(relevant_np) == 0:
            ap_scores.append(0.0)
            continue
            
        precisions = []
        for j in range(k):
            if relevant_np[j]:
                precision_at_j = np.sum(relevant_np[:j+1]) / (j + 1)
                precisions.append(precision_at_j)
                
        ap = np.mean(precisions) if precisions else 0.0
        ap_scores.append(ap)
        
    return np.mean(ap_scores)