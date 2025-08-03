"""Model analysis and interpretation utilities."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings


def analyze_model_predictions(model: nn.Module,
                            data_loader,
                            adjacency_matrix: mx.array,
                            num_samples: int = 100) -> Dict[str, np.ndarray]:
    """
    Analyze model predictions in detail.
    
    Args:
        model: Trained DT-GCNN model
        data_loader: Data loader with samples
        adjacency_matrix: Graph adjacency matrix
        num_samples: Number of samples to analyze
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    predictions = []
    true_labels = []
    confidences = []
    embeddings = []
    
    sample_count = 0
    
    for batch_features, batch_labels in data_loader:
        if sample_count >= num_samples:
            break
            
        # Get predictions
        logits = model(batch_features, adjacency_matrix, training=False)
        probs = mx.softmax(logits, axis=1)
        preds = mx.argmax(logits, axis=1)
        
        # Get embeddings if available
        if hasattr(model, 'get_embeddings'):
            emb = model.get_embeddings(batch_features, adjacency_matrix)
            embeddings.append(np.array(emb))
            
        predictions.extend(np.array(preds))
        true_labels.extend(np.array(batch_labels))
        confidences.extend(np.array(mx.max(probs, axis=1)))
        
        sample_count += len(batch_features)
        
    predictions = np.array(predictions[:num_samples])
    true_labels = np.array(true_labels[:num_samples])
    confidences = np.array(confidences[:num_samples])
    
    # Analyze results
    correct_mask = predictions == true_labels
    incorrect_mask = ~correct_mask
    
    analysis = {
        'predictions': predictions,
        'true_labels': true_labels,
        'confidences': confidences,
        'accuracy': np.mean(correct_mask),
        'avg_confidence_correct': np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0.0,
        'avg_confidence_incorrect': np.mean(confidences[incorrect_mask]) if np.any(incorrect_mask) else 0.0,
        'high_confidence_errors': np.sum((confidences > 0.9) & incorrect_mask),
        'low_confidence_correct': np.sum((confidences < 0.5) & correct_mask),
    }
    
    if embeddings:
        analysis['embeddings'] = np.concatenate(embeddings, axis=0)[:num_samples]
        
    # Per-class analysis
    unique_classes = np.unique(true_labels)
    per_class_accuracy = {}
    per_class_confidence = {}
    
    for class_id in unique_classes:
        class_mask = true_labels == class_id
        if np.any(class_mask):
            per_class_accuracy[class_id] = np.mean(correct_mask[class_mask])
            per_class_confidence[class_id] = np.mean(confidences[class_mask])
            
    analysis['per_class_accuracy'] = per_class_accuracy
    analysis['per_class_confidence'] = per_class_confidence
    
    return analysis


def compute_feature_importance(model: nn.Module,
                             features: mx.array,
                             adjacency_matrix: mx.array,
                             labels: mx.array,
                             method: str = 'gradient') -> Dict[str, mx.array]:
    """
    Compute feature importance scores.
    
    Args:
        model: Trained model
        features: Input features [batch, time, nodes, features]
        adjacency_matrix: Graph adjacency matrix
        labels: True labels
        method: 'gradient', 'permutation', or 'integrated_gradients'
        
    Returns:
        Dictionary with importance scores
    """
    model.eval()
    
    if method == 'gradient':
        # Gradient-based importance
        def loss_fn(features):
            logits = model(features, adjacency_matrix, training=False)
            loss = mx.mean(nn.losses.cross_entropy(logits, labels))
            return loss
            
        # Compute gradients
        grad_fn = mx.grad(loss_fn)
        grads = grad_fn(features)
        
        # Aggregate gradients
        # Average over batch and time dimensions
        importance = mx.mean(mx.abs(grads), axis=(0, 1))  # [nodes, features]
        
        return {
            'node_importance': mx.mean(importance, axis=1),  # [nodes]
            'feature_importance': mx.mean(importance, axis=0),  # [features]
            'node_feature_importance': importance  # [nodes, features]
        }
        
    elif method == 'permutation':
        # Permutation-based importance
        baseline_logits = model(features, adjacency_matrix, training=False)
        baseline_loss = mx.mean(nn.losses.cross_entropy(baseline_logits, labels))
        
        node_importance = []
        feature_importance = []
        
        # Permute each node's features
        for node_idx in range(features.shape[2]):
            permuted_features = mx.array(np.array(features))
            # Shuffle node features across time
            node_data = np.array(permuted_features[:, :, node_idx, :])
            np.random.shuffle(node_data.reshape(-1, node_data.shape[-1]))
            permuted_features = mx.array(permuted_features)
            
            permuted_logits = model(permuted_features, adjacency_matrix, training=False)
            permuted_loss = mx.mean(nn.losses.cross_entropy(permuted_logits, labels))
            
            importance = float(permuted_loss - baseline_loss)
            node_importance.append(importance)
            
        # Permute each feature dimension
        for feat_idx in range(features.shape[3]):
            permuted_features = mx.array(np.array(features))
            # Shuffle feature across all nodes and time
            feat_data = np.array(permuted_features[:, :, :, feat_idx])
            np.random.shuffle(feat_data.flatten())
            permuted_features = mx.array(permuted_features)
            
            permuted_logits = model(permuted_features, adjacency_matrix, training=False)
            permuted_loss = mx.mean(nn.losses.cross_entropy(permuted_logits, labels))
            
            importance = float(permuted_loss - baseline_loss)
            feature_importance.append(importance)
            
        return {
            'node_importance': mx.array(node_importance),
            'feature_importance': mx.array(feature_importance)
        }
        
    elif method == 'integrated_gradients':
        # Simplified integrated gradients
        steps = 20
        baseline = mx.zeros_like(features)
        
        integrated_grads = mx.zeros_like(features)
        
        for i in range(steps):
            alpha = i / steps
            interpolated = baseline + alpha * (features - baseline)
            
            def loss_fn(x):
                logits = model(x, adjacency_matrix, training=False)
                loss = mx.mean(nn.losses.cross_entropy(logits, labels))
                return loss
                
            grads = mx.grad(loss_fn)(interpolated)
            integrated_grads = integrated_grads + grads / steps
            
        # Multiply by input difference
        integrated_grads = integrated_grads * (features - baseline)
        
        # Aggregate
        importance = mx.mean(mx.abs(integrated_grads), axis=(0, 1))
        
        return {
            'node_importance': mx.mean(importance, axis=1),
            'feature_importance': mx.mean(importance, axis=0),
            'node_feature_importance': importance
        }
        
    else:
        raise ValueError(f"Unknown importance method: {method}")


def analyze_temporal_patterns(model: nn.Module,
                            features: mx.array,
                            adjacency_matrix: mx.array,
                            window_size: int = 5) -> Dict[str, np.ndarray]:
    """
    Analyze temporal patterns in model predictions.
    
    Args:
        model: Trained model
        features: Input features [batch, time, nodes, features]
        adjacency_matrix: Graph adjacency matrix
        window_size: Size of temporal window for analysis
        
    Returns:
        Dictionary with temporal analysis results
    """
    model.eval()
    
    batch_size, seq_length, num_nodes, num_features = features.shape
    
    # Get predictions for full sequence
    full_logits = model(features, adjacency_matrix, training=False)
    full_probs = mx.softmax(full_logits, axis=1)
    
    # Analyze predictions with masked time windows
    temporal_importance = []
    
    for t in range(0, seq_length - window_size + 1):
        # Mask out temporal window
        masked_features = mx.array(np.array(features))
        masked_features[:, t:t+window_size, :, :] = 0
        
        # Get predictions with masked window
        masked_logits = model(masked_features, adjacency_matrix, training=False)
        masked_probs = mx.softmax(masked_logits, axis=1)
        
        # Compute KL divergence
        kl_div = mx.sum(full_probs * mx.log(full_probs / (masked_probs + 1e-8)), axis=1)
        temporal_importance.append(float(mx.mean(kl_div)))
        
    # Analyze temporal receptive field
    receptive_field_analysis = []
    
    for t in range(seq_length):
        # Create impulse at time t
        impulse_features = mx.zeros_like(features)
        impulse_features[:, t, :, :] = 1.0
        
        # Get model response
        response = model(impulse_features, adjacency_matrix, training=False)
        response_magnitude = float(mx.mean(mx.abs(response)))
        receptive_field_analysis.append(response_magnitude)
        
    return {
        'temporal_importance': np.array(temporal_importance),
        'receptive_field_response': np.array(receptive_field_analysis),
        'most_important_window': int(np.argmax(temporal_importance)),
        'temporal_attention_span': np.sum(np.array(receptive_field_analysis) > 0.1)
    }


def evaluate_graph_influence(model: nn.Module,
                           features: mx.array,
                           adjacency_matrix: mx.array,
                           target_nodes: Optional[List[int]] = None) -> Dict[str, np.ndarray]:
    """
    Evaluate influence of graph structure on predictions.
    
    Args:
        model: Trained model
        features: Input features
        adjacency_matrix: Graph adjacency matrix
        target_nodes: Specific nodes to analyze (default: all)
        
    Returns:
        Dictionary with graph influence analysis
    """
    model.eval()
    
    num_nodes = adjacency_matrix.shape[0]
    if target_nodes is None:
        target_nodes = list(range(num_nodes))
        
    # Baseline predictions with full graph
    baseline_logits = model(features, adjacency_matrix, training=False)
    baseline_probs = mx.softmax(baseline_logits, axis=1)
    
    # Analyze edge influence
    edge_influence = np.zeros((num_nodes, num_nodes))
    
    adj_array = np.array(adjacency_matrix)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_array[i, j] > 0:
                # Remove edge
                modified_adj = adj_array.copy()
                modified_adj[i, j] = 0
                modified_adj[j, i] = 0  # Symmetric
                
                # Renormalize
                row_sums = modified_adj.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1
                modified_adj = modified_adj / row_sums
                
                # Get predictions with modified graph
                modified_logits = model(features, mx.array(modified_adj), training=False)
                modified_probs = mx.softmax(modified_logits, axis=1)
                
                # Compute influence as KL divergence
                kl_div = mx.sum(baseline_probs * mx.log(baseline_probs / (modified_probs + 1e-8)), axis=1)
                edge_influence[i, j] = float(mx.mean(kl_div))
                
    # Analyze node influence
    node_influence = []
    
    for node in target_nodes:
        # Isolate node (remove all edges)
        isolated_adj = adj_array.copy()
        isolated_adj[node, :] = 0
        isolated_adj[:, node] = 0
        isolated_adj[node, node] = 1  # Self-loop
        
        # Renormalize
        row_sums = isolated_adj.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        isolated_adj = isolated_adj / row_sums
        
        # Get predictions
        isolated_logits = model(features, mx.array(isolated_adj), training=False)
        isolated_probs = mx.softmax(isolated_logits, axis=1)
        
        # Compute influence
        kl_div = mx.sum(baseline_probs * mx.log(baseline_probs / (isolated_probs + 1e-8)), axis=1)
        node_influence.append(float(mx.mean(kl_div)))
        
    # Analyze graph properties influence
    # Test with different graph structures
    graph_structures = {
        'fully_connected': np.ones((num_nodes, num_nodes)) / num_nodes,
        'identity': np.eye(num_nodes),
        'random': np.random.rand(num_nodes, num_nodes)
    }
    
    structure_influence = {}
    
    for name, structure in graph_structures.items():
        # Normalize
        structure = structure / structure.sum(axis=1, keepdims=True)
        
        struct_logits = model(features, mx.array(structure), training=False)
        struct_probs = mx.softmax(struct_logits, axis=1)
        
        kl_div = mx.sum(baseline_probs * mx.log(baseline_probs / (struct_probs + 1e-8)), axis=1)
        structure_influence[name] = float(mx.mean(kl_div))
        
    return {
        'edge_influence': edge_influence,
        'node_influence': np.array(node_influence),
        'most_influential_edges': np.unravel_index(np.argmax(edge_influence), edge_influence.shape),
        'most_influential_node': target_nodes[np.argmax(node_influence)] if node_influence else -1,
        'structure_influence': structure_influence,
        'graph_dependency_score': np.mean(edge_influence[edge_influence > 0])
    }


def analyze_failure_cases(model: nn.Module,
                         data_loader,
                         adjacency_matrix: mx.array,
                         num_failures: int = 50) -> Dict[str, Union[np.ndarray, List]]:
    """
    Detailed analysis of model failure cases.
    
    Args:
        model: Trained model
        data_loader: Data loader
        adjacency_matrix: Graph adjacency matrix
        num_failures: Number of failures to analyze
        
    Returns:
        Dictionary with failure analysis
    """
    model.eval()
    
    failures = []
    failure_count = 0
    
    for batch_features, batch_labels in data_loader:
        if failure_count >= num_failures:
            break
            
        # Get predictions
        logits = model(batch_features, adjacency_matrix, training=False)
        probs = mx.softmax(logits, axis=1)
        preds = mx.argmax(logits, axis=1)
        
        # Find failures
        incorrect_mask = preds != batch_labels
        incorrect_indices = np.where(np.array(incorrect_mask))[0]
        
        for idx in incorrect_indices:
            if failure_count >= num_failures:
                break
                
            failure_info = {
                'true_label': int(batch_labels[idx]),
                'predicted_label': int(preds[idx]),
                'confidence': float(mx.max(probs[idx])),
                'true_class_prob': float(probs[idx, int(batch_labels[idx])]),
                'confusion': (int(batch_labels[idx]), int(preds[idx])),
                'features_summary': {
                    'mean': float(mx.mean(batch_features[idx])),
                    'std': float(mx.std(batch_features[idx])),
                    'max': float(mx.max(batch_features[idx])),
                    'min': float(mx.min(batch_features[idx]))
                }
            }
            
            # Get top-3 predictions
            top3_indices = mx.argsort(probs[idx])[-3:][::-1]
            top3_probs = probs[idx, top3_indices]
            failure_info['top3_predictions'] = [(int(i), float(p)) for i, p in zip(top3_indices, top3_probs)]
            
            failures.append(failure_info)
            failure_count += 1
            
    # Analyze failure patterns
    confusion_pairs = [(f['true_label'], f['predicted_label']) for f in failures]
    unique_pairs, counts = np.unique(confusion_pairs, axis=0, return_counts=True)
    
    most_common_confusions = []
    for pair, count in zip(unique_pairs, counts):
        most_common_confusions.append({
            'true_label': int(pair[0]),
            'predicted_label': int(pair[1]),
            'count': int(count),
            'percentage': float(count / len(failures) * 100)
        })
        
    most_common_confusions.sort(key=lambda x: x['count'], reverse=True)
    
    # Average confidence analysis
    avg_confidence_failures = np.mean([f['confidence'] for f in failures])
    avg_true_class_prob = np.mean([f['true_class_prob'] for f in failures])
    
    return {
        'failures': failures,
        'num_failures_analyzed': len(failures),
        'most_common_confusions': most_common_confusions[:5],
        'avg_confidence_on_failures': avg_confidence_failures,
        'avg_true_class_probability': avg_true_class_prob,
        'confidence_distribution': np.histogram([f['confidence'] for f in failures], bins=10)[0],
        'high_confidence_failures': sum(1 for f in failures if f['confidence'] > 0.8)
    }