"""
Microtubule Graph Structure for Time Crystal Simulation

Creates graph representations of microtubules with tubulin dimers as nodes
and spatial/chemical interactions as edges.
"""

import mlx.core as mx
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class TubulinDimer:
    """Represents a single tubulin dimer in the microtubule."""
    
    def __init__(self, x: float, y: float, z: float, protofilament: int, position: int):
        self.x = x
        self.y = y 
        self.z = z
        self.protofilament = protofilament  # Which protofilament (0-12 for typical MT)
        self.position = position  # Position along protofilament
        self.oscillation_freq = 0.0  # MHz frequency
        self.phase = 0.0  # Oscillation phase
        

class MicrotubuleGraph:
    """
    Creates a graph representation of microtubules for time crystal simulation.
    
    Based on biological structure:
    - 13 protofilaments in typical microtubule
    - ~8nm spacing between dimers
    - ~25nm microtubule diameter
    - Helical structure with 3-start helix
    """
    
    def __init__(self, 
                 length_um: float = 10.0,
                 num_protofilaments: int = 13,
                 dimer_spacing_nm: float = 8.0):
        """
        Initialize microtubule structure.
        
        Args:
            length_um: Length of microtubule in micrometers
            num_protofilaments: Number of protofilaments (typically 13)
            dimer_spacing_nm: Spacing between dimers in nanometers
        """
        self.length_um = length_um
        self.num_protofilaments = num_protofilaments
        self.dimer_spacing_nm = dimer_spacing_nm
        self.mt_diameter_nm = 25.0  # Typical microtubule diameter
        
        # Calculate structure
        self.length_nm = length_um * 1000  # Convert to nm
        self.dimers_per_protofilament = int(self.length_nm / dimer_spacing_nm)
        self.total_dimers = self.num_protofilaments * self.dimers_per_protofilament
        
        # Create dimers and adjacency
        self.dimers = self._create_dimers()
        self.adjacency_matrix = self._create_adjacency_matrix()
        self.edge_index = self._create_edge_index()
        
    def _create_dimers(self) -> List[TubulinDimer]:
        """Create all tubulin dimers with 3D coordinates."""
        dimers = []
        
        for pf in range(self.num_protofilaments):
            # Protofilament angle around microtubule
            angle = 2 * math.pi * pf / self.num_protofilaments
            
            for pos in range(self.dimers_per_protofilament):
                # Z-coordinate along microtubule axis
                z = pos * self.dimer_spacing_nm
                
                # Helical twist (3-start helix)
                helical_angle = angle + (3 * 2 * math.pi * pos / self.dimers_per_protofilament)
                
                # X,Y coordinates on cylinder surface
                radius = self.mt_diameter_nm / 2
                x = radius * math.cos(helical_angle)
                y = radius * math.sin(helical_angle)
                
                dimer = TubulinDimer(x, y, z, pf, pos)
                dimers.append(dimer)
                
        return dimers
    
    def _create_adjacency_matrix(self) -> mx.array:
        """Create adjacency matrix based on spatial proximity and bonds."""
        n = self.total_dimers
        adj = np.zeros((n, n), dtype=np.float32)
        
        for i, dimer_i in enumerate(self.dimers):
            for j, dimer_j in enumerate(self.dimers):
                if i == j:
                    continue
                    
                # Calculate 3D distance
                dx = dimer_i.x - dimer_j.x
                dy = dimer_i.y - dimer_j.y  
                dz = dimer_i.z - dimer_j.z
                distance = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                # Connection criteria
                connected = False
                weight = 0.0
                
                # Longitudinal bonds (along protofilament)
                if (dimer_i.protofilament == dimer_j.protofilament and
                    abs(dimer_i.position - dimer_j.position) == 1):
                    connected = True
                    weight = 1.0  # Strong longitudinal bond
                
                # Lateral bonds (between protofilaments)
                elif (abs(dimer_i.protofilament - dimer_j.protofilament) == 1 or
                      abs(dimer_i.protofilament - dimer_j.protofilament) == self.num_protofilaments - 1):
                    if abs(dimer_i.position - dimer_j.position) <= 1:
                        connected = True
                        weight = 0.7  # Weaker lateral bond
                
                # Long-range interactions (exponential decay)
                elif distance < 50.0:  # Within 50nm
                    weight = 0.1 * math.exp(-distance / 20.0)
                    if weight > 0.05:
                        connected = True
                
                if connected:
                    adj[i, j] = weight
                    
        return mx.array(adj)
    
    def _create_edge_index(self) -> mx.array:
        """Create edge index in COO format for efficient processing."""
        adj_np = np.array(self.adjacency_matrix)
        edges = np.where(adj_np > 0)
        edge_index = np.stack([edges[0], edges[1]], axis=0)
        return mx.array(edge_index)
    
    def get_coordinates(self) -> mx.array:
        """Get 3D coordinates of all dimers."""
        coords = np.zeros((self.total_dimers, 3), dtype=np.float32)
        for i, dimer in enumerate(self.dimers):
            coords[i] = [dimer.x, dimer.y, dimer.z]
        return mx.array(coords)
    
    def get_node_features(self, include_spatial: bool = True) -> mx.array:
        """
        Get node features for each dimer.
        
        Args:
            include_spatial: Whether to include 3D coordinates as features
            
        Returns:
            Node features array of shape (num_nodes, feature_dim)
        """
        features = []
        
        for dimer in self.dimers:
            node_feat = []
            
            if include_spatial:
                # Normalized coordinates
                node_feat.extend([
                    dimer.x / self.mt_diameter_nm,  # Normalized X
                    dimer.y / self.mt_diameter_nm,  # Normalized Y  
                    dimer.z / self.length_nm       # Normalized Z
                ])
            
            # Protofilament one-hot encoding
            pf_onehot = [0.0] * self.num_protofilaments
            pf_onehot[dimer.protofilament] = 1.0
            node_feat.extend(pf_onehot)
            
            # Position along protofilament (normalized)
            node_feat.append(dimer.position / self.dimers_per_protofilament)
            
            # Radial position (for cylindrical coordinates)
            angle = 2 * math.pi * dimer.protofilament / self.num_protofilaments
            node_feat.extend([math.cos(angle), math.sin(angle)])
            
            features.append(node_feat)
            
        return mx.array(features)
    
    def initialize_oscillations(self, 
                              freq_range_mhz: Tuple[float, float] = (1.0, 100.0),
                              coherent_fraction: float = 0.3) -> mx.array:
        """
        Initialize MHz oscillations for each dimer.
        
        Args:
            freq_range_mhz: Frequency range for oscillations
            coherent_fraction: Fraction of dimers that start coherently
            
        Returns:
            Initial oscillation states [frequency, phase, amplitude]
        """
        np.random.seed(42)  # Reproducible results
        
        states = []
        coherent_count = int(self.total_dimers * coherent_fraction)
        
        # Coherent oscillators (similar frequency and phase)
        base_freq = np.random.uniform(freq_range_mhz[0], freq_range_mhz[1])
        base_phase = np.random.uniform(0, 2*math.pi)
        
        for i in range(self.total_dimers):
            if i < coherent_count:
                # Coherent group - small variations around base
                freq = base_freq + np.random.normal(0, base_freq * 0.1)
                phase = base_phase + np.random.normal(0, 0.2)
            else:
                # Random oscillators
                freq = np.random.uniform(freq_range_mhz[0], freq_range_mhz[1])
                phase = np.random.uniform(0, 2*math.pi)
            
            # Amplitude based on local connectivity
            local_connections = mx.sum(self.adjacency_matrix[i] > 0).item()
            amplitude = 0.5 + 0.5 * (local_connections / 10.0)  # 0.5 to 1.0
            
            states.append([freq, phase, amplitude])
            
        return mx.array(states)
        
    def add_thermal_noise(self, 
                         oscillation_states: mx.array,
                         temperature_k: float = 310.0,
                         noise_strength: float = 0.1) -> mx.array:
        """
        Add thermal noise to oscillation states.
        
        Args:
            oscillation_states: Current oscillation states
            temperature_k: Temperature in Kelvin (body temperature ~310K)
            noise_strength: Strength of thermal fluctuations
            
        Returns:
            Noisy oscillation states
        """
        # Thermal energy at given temperature
        k_b = 1.38e-23  # Boltzmann constant
        thermal_energy = k_b * temperature_k
        
        # Scale noise by thermal energy (normalized)
        thermal_scale = noise_strength * math.sqrt(thermal_energy / (k_b * 300))
        
        # Add Gaussian noise to each component
        noise = mx.random.normal(oscillation_states.shape) * thermal_scale
        
        # Ensure frequencies stay positive
        noisy_states = oscillation_states + noise
        noisy_states = mx.where(noisy_states[:, 0:1] < 0.1, 
                               mx.concatenate([mx.ones((len(noisy_states), 1)) * 0.1, 
                                             noisy_states[:, 1:]], axis=1),
                               noisy_states)
        
        return noisy_states
    
    def compute_graph_statistics(self) -> Dict[str, float]:
        """Compute graph topology statistics."""
        adj_np = np.array(self.adjacency_matrix)
        
        # Basic statistics
        num_edges = np.sum(adj_np > 0) // 2  # Undirected graph
        avg_degree = np.mean(np.sum(adj_np > 0, axis=1))
        max_degree = np.max(np.sum(adj_np > 0, axis=1))
        
        # Clustering coefficient (approximate)
        degrees = np.sum(adj_np > 0, axis=1)
        clustering = 0.0
        for i in range(len(adj_np)):
            if degrees[i] > 1:
                neighbors = np.where(adj_np[i] > 0)[0]
                possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
                actual_edges = 0
                for j in range(len(neighbors)):
                    for k in range(j+1, len(neighbors)):
                        if adj_np[neighbors[j], neighbors[k]] > 0:
                            actual_edges += 1
                if possible_edges > 0:
                    clustering += actual_edges / possible_edges
        clustering /= len(adj_np)
        
        return {
            'num_nodes': self.total_dimers,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'clustering_coeff': clustering,
            'length_um': self.length_um,
            'protofilaments': self.num_protofilaments
        }


def create_microtubule_network(num_microtubules: int = 1,
                              length_um: float = 10.0,
                              spacing_um: float = 1.0) -> MicrotubuleGraph:
    """
    Create a network of multiple microtubules.
    
    For now, returns single microtubule. Can be extended for dendritic networks.
    """
    # Start with single microtubule
    mt = MicrotubuleGraph(length_um=length_um)
    
    # TODO: Extend for multiple MTs with cross-linking
    # This would involve:
    # 1. Multiple parallel microtubules
    # 2. Cross-linking proteins (MAP2, tau)
    # 3. Inter-MT interactions
    
    return mt


if __name__ == "__main__":
    # Test microtubule graph creation
    print("Creating microtubule graph...")
    mt = MicrotubuleGraph(length_um=5.0)
    
    stats = mt.compute_graph_statistics()
    print("\nMicrotubule Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test oscillation initialization
    print("\nInitializing oscillations...")
    oscillations = mt.initialize_oscillations()
    print(f"Oscillation states shape: {oscillations.shape}")
    print(f"Frequency range: {mx.min(oscillations[:, 0]):.2f} - {mx.max(oscillations[:, 0]):.2f} MHz")
    
    # Test thermal noise
    print("\nAdding thermal noise...")
    noisy_oscillations = mt.add_thermal_noise(oscillations)
    print(f"Noisy frequency range: {mx.min(noisy_oscillations[:, 0]):.2f} - {mx.max(noisy_oscillations[:, 0]):.2f} MHz")
    
    print("\nMicrotubule graph creation successful!")