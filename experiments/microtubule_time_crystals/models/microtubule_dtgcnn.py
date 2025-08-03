"""
Adapted DT-GCNN for Microtubule Time Crystal Modeling

Extends the original DT-GCNN architecture for:
- MHz oscillation signal processing
- Temporal coherence learning
- Predictive coding of microtubule dynamics
- EEG frequency down-conversion
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Dict, List
import math
import sys
import os

# Import original DT-GCNN components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dt-gcnn-mlx', 'src'))
from models.dt_gcnn import GRUCell, DTGCNN


class MicrotubuleGCNLayer(nn.Module):
    """
    Graph Convolutional Layer adapted for microtubule networks.
    
    Processes spatial relationships and oscillation propagation
    between tubulin dimers in microtubule structure.
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 activation: str = "relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear transformation for node features
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        # Normalization for stability
        self.norm = nn.LayerNorm(output_dim)
        
    def __call__(self, 
                 node_features: mx.array,
                 adjacency_matrix: mx.array,
                 edge_weights: Optional[mx.array] = None) -> mx.array:
        """
        Forward pass of microtubule GCN layer.
        
        Args:
            node_features: Node features (num_nodes, input_dim)
            adjacency_matrix: Graph adjacency matrix (num_nodes, num_nodes)
            edge_weights: Optional edge weights for weighted aggregation
            
        Returns:
            Updated node features (num_nodes, output_dim)
        """
        num_nodes = node_features.shape[0]
        
        # Linear transformation
        transformed = self.linear(node_features)
        
        # Graph convolution: aggregate from neighbors
        if edge_weights is not None:
            # Weighted adjacency
            weighted_adj = adjacency_matrix * edge_weights
        else:
            weighted_adj = adjacency_matrix
        
        # Normalize adjacency matrix (add self-loops and degree normalization)
        # Add self-loops
        identity = mx.eye(num_nodes)
        adj_with_self = weighted_adj + identity
        
        # Degree normalization: D^(-1/2) * A * D^(-1/2)
        degree = mx.sum(adj_with_self, axis=1)
        degree_inv_sqrt = mx.power(degree + 1e-8, -0.5)
        degree_matrix_inv_sqrt = mx.diag(degree_inv_sqrt)
        
        normalized_adj = degree_matrix_inv_sqrt @ adj_with_self @ degree_matrix_inv_sqrt
        
        # Message passing: aggregate features from neighbors
        aggregated = normalized_adj @ transformed
        
        # Apply activation and normalization
        output = self.activation(aggregated)
        output = self.norm(output)
        
        return output


class TemporalCoherenceModule(nn.Module):
    """
    Models temporal coherence in microtubule oscillations.
    
    Uses recurrent connections to maintain phase relationships
    and model quantum coherence effects.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 hidden_dim: int = 128,
                 coherence_steps: int = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.coherence_steps = coherence_steps
        
        # GRU for temporal dynamics
        self.temporal_gru = GRUCell(feature_dim, hidden_dim)
        
        # Phase coherence modeling
        self.phase_projection = nn.Linear(hidden_dim, 2)  # [cos(φ), sin(φ)]
        self.amplitude_projection = nn.Linear(hidden_dim, 1)
        
        # Coherence interaction
        self.coherence_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def __call__(self, 
                 node_features: mx.array,
                 adjacency_matrix: mx.array,
                 num_time_steps: int = 10) -> Tuple[mx.array, mx.array]:
        """
        Model temporal coherence evolution.
        
        Args:
            node_features: Initial node features
            adjacency_matrix: Graph structure
            num_time_steps: Number of coherence evolution steps
            
        Returns:
            Tuple of (evolved_features, coherence_measures)
        """
        num_nodes = node_features.shape[0]
        
        # Initialize hidden states
        hidden_state = mx.zeros((num_nodes, self.hidden_dim))
        
        # Track coherence evolution
        coherence_history = []
        
        for step in range(num_time_steps):
            # Update temporal state
            hidden_state = self.temporal_gru(node_features, hidden_state)
            
            # Compute local phase and amplitude
            phase_components = self.phase_projection(hidden_state)
            amplitudes = self.amplitude_projection(hidden_state)
            
            # Normalize phase components to unit circle
            phase_norm = mx.sqrt(mx.sum(phase_components**2, axis=1, keepdims=True) + 1e-8)
            phase_normalized = phase_components / phase_norm
            
            # Coherence interaction between neighbors
            for i in range(num_nodes):
                neighbor_indices = mx.where(adjacency_matrix[i] > 0)[0]
                
                if len(neighbor_indices) > 0:
                    # Get neighbor states
                    neighbor_states = hidden_state[neighbor_indices]
                    current_state = hidden_state[i:i+1]
                    
                    # Mean-field coherence interaction
                    mean_neighbor = mx.mean(neighbor_states, axis=0, keepdims=True)
                    interaction_input = mx.concatenate([current_state, mean_neighbor], axis=1)
                    
                    # Update through coherence MLP
                    coherence_update = self.coherence_mlp(interaction_input)
                    hidden_state = hidden_state.at[i].set(coherence_update.squeeze(0))
            
            # Measure global coherence (order parameter)
            phases = mx.arctan2(phase_normalized[:, 1], phase_normalized[:, 0])
            complex_phases = mx.exp(1j * phases)
            order_parameter = mx.abs(mx.mean(complex_phases))
            
            coherence_history.append(float(order_parameter))
        
        # Final evolved features
        evolved_features = mx.concatenate([
            hidden_state,
            phase_components,
            amplitudes
        ], axis=1)
        
        return evolved_features, mx.array(coherence_history)


class PredictiveCodingModule(nn.Module):
    """
    Implements predictive coding for microtubule dynamics.
    
    Based on free energy principle:
    - Predicts next oscillation state
    - Minimizes prediction error
    - Updates internal model
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int = 64,
                 prediction_steps: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.prediction_steps = prediction_steps
        
        # Encoder (recognition model)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )
        
        # Decoder (generative model)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, input_dim)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def encode(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Encode input to latent space (VAE-style)."""
        encoded = self.encoder(x)
        mu = encoded[:, :self.latent_dim]
        log_var = encoded[:, self.latent_dim:]
        return mu, log_var
    
    def reparameterize(self, mu: mx.array, log_var: mx.array) -> mx.array:
        """Reparameterization trick for VAE."""
        std = mx.exp(0.5 * log_var)
        eps = mx.random.normal(mu.shape)
        return mu + std * eps
    
    def decode(self, z: mx.array) -> mx.array:
        """Decode latent representation to observation space."""
        return self.decoder(z)
    
    def predict_next_state(self, z: mx.array) -> mx.array:
        """Predict next latent state."""
        return self.predictor(z)
    
    def __call__(self, 
                 current_state: mx.array,
                 target_state: Optional[mx.array] = None) -> Dict[str, mx.array]:
        """
        Predictive coding forward pass.
        
        Args:
            current_state: Current oscillation state
            target_state: Target state for supervised learning
            
        Returns:
            Dictionary with predictions, reconstructions, and losses
        """
        # Encode current state
        mu, log_var = self.encode(current_state)
        z = self.reparameterize(mu, log_var)
        
        # Reconstruct current state
        reconstruction = self.decode(z)
        
        # Predict next state in latent space
        z_next_pred = self.predict_next_state(z)
        next_state_pred = self.decode(z_next_pred)
        
        # Compute losses
        results = {
            'reconstruction': reconstruction,
            'next_prediction': next_state_pred,
            'latent_mu': mu,
            'latent_log_var': log_var,
            'latent_z': z
        }
        
        # Reconstruction loss
        recon_loss = mx.mean((current_state - reconstruction) ** 2)
        results['recon_loss'] = recon_loss
        
        # KL divergence loss (regularization)
        kl_loss = -0.5 * mx.mean(1 + log_var - mu**2 - mx.exp(log_var))
        results['kl_loss'] = kl_loss
        
        # Prediction loss (if target available)
        if target_state is not None:
            pred_loss = mx.mean((target_state - next_state_pred) ** 2)
            results['pred_loss'] = pred_loss
        
        return results


class MicrotubuleDTGCNN(nn.Module):
    """
    Complete DT-GCNN model adapted for microtubule time crystal simulation.
    
    Combines:
    - Spatial graph convolution for microtubule structure
    - Temporal coherence modeling
    - Predictive coding for dynamics
    - Multi-scale frequency analysis
    """
    
    def __init__(self,
                 node_feature_dim: int = 32,
                 hidden_dims: List[int] = [128, 64, 32],
                 coherence_dim: int = 64,
                 latent_dim: int = 32,
                 num_gcn_layers: int = 3,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dims = hidden_dims
        self.num_gcn_layers = num_gcn_layers
        
        # Input projection
        self.input_projection = nn.Linear(node_feature_dim, hidden_dims[0])
        
        # Graph convolutional layers
        self.gcn_layers = []
        for i in range(num_gcn_layers):
            input_dim = hidden_dims[min(i, len(hidden_dims)-1)]
            output_dim = hidden_dims[min(i+1, len(hidden_dims)-1)]
            
            gcn_layer = MicrotubuleGCNLayer(input_dim, output_dim)
            self.gcn_layers.append(gcn_layer)
        
        # Temporal coherence module
        final_dim = hidden_dims[-1]
        self.temporal_module = TemporalCoherenceModule(
            feature_dim=final_dim,
            hidden_dim=coherence_dim,
            coherence_steps=10
        )
        
        # Predictive coding module
        coherence_output_dim = coherence_dim + 2 + 1  # hidden + phase + amplitude
        self.predictive_module = PredictiveCodingModule(
            input_dim=coherence_output_dim,
            latent_dim=latent_dim,
            prediction_steps=5
        )
        
        # Output heads
        self.oscillation_head = nn.Linear(latent_dim, 3)  # freq, phase, amplitude
        self.coherence_head = nn.Linear(coherence_output_dim, 1)  # coherence score
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def __call__(self, 
                 node_features: mx.array,
                 adjacency_matrix: mx.array,
                 oscillation_states: Optional[mx.array] = None,
                 return_intermediate: bool = False) -> Dict[str, mx.array]:
        """
        Forward pass of microtubule DT-GCNN.
        
        Args:
            node_features: Initial node features
            adjacency_matrix: Graph structure
            oscillation_states: Current oscillation states for prediction
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Dictionary with model outputs and intermediate results
        """
        results = {}
        
        # Input projection
        x = self.input_projection(node_features)
        x = self.dropout(x)
        
        # Graph convolution layers
        gcn_outputs = [x]
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adjacency_matrix)
            x = self.dropout(x)
            gcn_outputs.append(x)
        
        if return_intermediate:
            results['gcn_outputs'] = gcn_outputs
        
        # Temporal coherence modeling
        coherence_features, coherence_history = self.temporal_module(
            x, adjacency_matrix, num_time_steps=10
        )
        
        results['coherence_features'] = coherence_features
        results['coherence_history'] = coherence_history
        
        # Predictive coding
        if oscillation_states is not None:
            # Use oscillation states as target for prediction
            pred_results = self.predictive_module(
                coherence_features, 
                target_state=oscillation_states
            )
            results.update(pred_results)
        else:
            # Unsupervised mode
            pred_results = self.predictive_module(coherence_features)
            results.update(pred_results)
        
        # Output predictions
        latent_z = results['latent_z']
        
        # Oscillation parameters
        oscillation_output = self.oscillation_head(latent_z)
        results['predicted_oscillations'] = oscillation_output
        
        # Coherence score
        coherence_score = self.coherence_head(coherence_features)
        results['coherence_score'] = coherence_score
        
        return results
    
    def compute_total_loss(self, 
                          outputs: Dict[str, mx.array],
                          target_oscillations: Optional[mx.array] = None,
                          coherence_weight: float = 0.1,
                          prediction_weight: float = 1.0) -> mx.array:
        """
        Compute total training loss.
        
        Args:
            outputs: Model outputs
            target_oscillations: Ground truth oscillations
            coherence_weight: Weight for coherence regularization
            prediction_weight: Weight for prediction loss
            
        Returns:
            Total loss
        """
        losses = []
        
        # Reconstruction loss (always present)
        if 'recon_loss' in outputs:
            losses.append(outputs['recon_loss'])
        
        # KL divergence loss (VAE regularization)
        if 'kl_loss' in outputs:
            losses.append(0.1 * outputs['kl_loss'])
        
        # Prediction loss
        if 'pred_loss' in outputs:
            losses.append(prediction_weight * outputs['pred_loss'])
        
        # Oscillation prediction loss
        if target_oscillations is not None and 'predicted_oscillations' in outputs:
            osc_loss = mx.mean((target_oscillations - outputs['predicted_oscillations']) ** 2)
            losses.append(osc_loss)
        
        # Coherence regularization (encourage high coherence)
        if 'coherence_score' in outputs:
            # Penalize low coherence (want high coherence)
            coherence_penalty = mx.mean((1.0 - outputs['coherence_score']) ** 2)
            losses.append(coherence_weight * coherence_penalty)
        
        # Total loss
        if losses:
            total_loss = sum(losses)
        else:
            total_loss = mx.array(0.0)
        
        return total_loss


def create_microtubule_model(config: Optional[Dict] = None) -> MicrotubuleDTGCNN:
    """
    Create a microtubule DT-GCNN model with default or custom configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured MicrotubuleDTGCNN model
    """
    default_config = {
        'node_feature_dim': 32,
        'hidden_dims': [128, 64, 32],
        'coherence_dim': 64,
        'latent_dim': 32,
        'num_gcn_layers': 3,
        'dropout_rate': 0.1
    }
    
    if config:
        default_config.update(config)
    
    return MicrotubuleDTGCNN(**default_config)


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing Microtubule DT-GCNN model...")
    
    # Create test data
    num_nodes = 50
    node_feature_dim = 32
    
    node_features = mx.random.normal((num_nodes, node_feature_dim))
    adjacency_matrix = mx.random.uniform(0, 1, (num_nodes, num_nodes))
    adjacency_matrix = (adjacency_matrix > 0.8).astype(mx.float32)  # Sparse adjacency
    
    # Create model
    model = create_microtubule_model()
    
    print(f"Model created with {sum(p.size for p in model.parameters().values())} parameters")
    
    # Forward pass
    outputs = model(node_features, adjacency_matrix, return_intermediate=True)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, mx.array):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of {len(value)} items")
        else:
            print(f"  {key}: {type(value)}")
    
    # Test loss computation
    target_oscillations = mx.random.normal((num_nodes, 3))
    loss = model.compute_total_loss(outputs, target_oscillations)
    print(f"Total loss: {loss.item():.4f}")
    
    print("Microtubule DT-GCNN model test completed successfully!")