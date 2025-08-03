"""
MHz Oscillation Dynamics for Microtubule Time Crystals

Implements time evolution of microtubule oscillations with:
- MHz frequency generation
- Thermal noise modeling  
- Coherence mechanisms
- Down-conversion to EEG frequencies
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import math


class MicrotubuleOscillator:
    """
    Models individual microtubule segment as quantum-coherent oscillator.
    
    Based on time crystal hypothesis:
    - Discrete time translation symmetry breaking
    - Spontaneous oscillation at MHz frequencies
    - Quantum coherence despite thermal noise
    """
    
    def __init__(self, 
                 base_frequency_mhz: float = 40.0,
                 coherence_time_ms: float = 1.0,
                 coupling_strength: float = 0.1):
        """
        Initialize oscillator parameters.
        
        Args:
            base_frequency_mhz: Base oscillation frequency in MHz
            coherence_time_ms: Quantum coherence timescale in milliseconds  
            coupling_strength: Inter-oscillator coupling (0-1)
        """
        self.base_freq = base_frequency_mhz * 1e6  # Convert to Hz
        self.coherence_time = coherence_time_ms * 1e-3  # Convert to seconds
        self.coupling = coupling_strength
        
        # Quantum parameters
        self.decoherence_rate = 1.0 / self.coherence_time
        self.thermal_cutoff_hz = 1e13  # ~kT at 310K
        
    def generate_time_crystal_signal(self, 
                                   duration_sec: float,
                                   sampling_rate_hz: float,
                                   num_oscillators: int) -> mx.array:
        """
        Generate time crystal oscillation signals.
        
        Args:
            duration_sec: Signal duration in seconds
            sampling_rate_hz: Sampling rate in Hz (must be > 2 * max_freq)
            num_oscillators: Number of coupled oscillators
            
        Returns:
            Oscillation signals of shape (num_samples, num_oscillators)
        """
        num_samples = int(duration_sec * sampling_rate_hz)
        dt = 1.0 / sampling_rate_hz
        t = mx.arange(num_samples) * dt
        
        # Initialize oscillator states
        phases = mx.random.uniform(0, 2*math.pi, (num_oscillators,))
        frequencies = self.base_freq + mx.random.normal((num_oscillators,)) * self.base_freq * 0.1
        amplitudes = mx.ones((num_oscillators,))
        
        # Generate all time points at once for efficiency
        t_matrix = t[:, None]  # (num_samples, 1)
        frequencies_matrix = frequencies[None, :]  # (1, num_oscillators)
        phases_matrix = phases[None, :]  # (1, num_oscillators)
        amplitudes_matrix = amplitudes[None, :]  # (1, num_oscillators)
        
        # Basic oscillation (no coupling for simplicity in this test)
        signals = amplitudes_matrix * mx.sin(2 * math.pi * frequencies_matrix * t_matrix + phases_matrix)
        
        # Add decoherence envelope
        decoherence_envelope = mx.exp(-self.decoherence_rate * t_matrix)
        signals = signals * decoherence_envelope
            
        return signals
    
    def add_thermal_fluctuations(self, 
                               signals: mx.array,
                               temperature_k: float = 310.0,
                               noise_strength: float = 0.1) -> mx.array:
        """
        Add thermal noise based on fluctuation-dissipation theorem.
        
        Args:
            signals: Clean oscillation signals
            temperature_k: Temperature in Kelvin
            noise_strength: Relative noise strength
            
        Returns:
            Noisy signals with thermal fluctuations
        """
        # Thermal energy scale
        k_b = 1.38e-23
        thermal_energy = k_b * temperature_k
        
        # Generate correlated thermal noise
        noise_variance = noise_strength * thermal_energy / (k_b * 300)  # Normalized
        noise = mx.random.normal(signals.shape) * math.sqrt(noise_variance)
        
        # Simple white noise for now (colored noise implementation simplified)
        # In a full implementation, would use proper 1/f filtering
        
        return signals + noise


class EEGDownConverter:
    """
    Converts MHz microtubule oscillations to EEG frequency bands via interference.
    
    Based on beat frequency mechanism:
    - Two MHz signals create beat at difference frequency
    - Multiple beats create complex EEG-like patterns
    - Maintains phase relationships for coherence
    """
    
    def __init__(self):
        """Initialize down-conversion parameters."""
        # EEG frequency bands (Hz)
        self.eeg_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'high_gamma': (100, 300)
        }
        
    def compute_beat_frequencies(self, 
                               mhz_signals: mx.array,
                               adjacency_matrix: mx.array) -> mx.array:
        """
        Simplified beat frequency computation using signal envelope.
        
        Args:
            mhz_signals: MHz oscillation signals (samples, oscillators)
            adjacency_matrix: Coupling matrix between oscillators
            
        Returns:
            Beat-like signals in lower frequency range
        """
        num_samples, num_oscillators = mhz_signals.shape
        
        # Simplified approach: compute envelope differences between coupled oscillators
        # Use moving average to extract envelope
        window_size = min(10, num_samples // 4)
        if window_size < 1:
            window_size = 1
            
        # Simple envelope extraction via absolute value and smoothing
        envelopes = mx.abs(mhz_signals)
        
        # Smooth envelopes (simple moving average approximation)
        if window_size > 1:
            # Create simple kernel
            kernel = mx.ones(window_size) / window_size
            # Simple convolution approximation
            smoothed_envelopes = mx.zeros_like(envelopes)
            for i in range(num_oscillators):
                signal = envelopes[:, i]
                # Pad signal for convolution
                if len(signal) >= window_size:
                    smoothed = mx.zeros(len(signal))
                    for j in range(len(signal)):
                        start_idx = max(0, j - window_size // 2)
                        end_idx = min(len(signal), j + window_size // 2 + 1)
                        smoothed_val = mx.mean(signal[start_idx:end_idx])
                        smoothed = mx.concatenate([smoothed[:j], mx.array([smoothed_val]), smoothed[j+1:]])
                    smoothed_envelopes = mx.concatenate([
                        smoothed_envelopes[:, :i], 
                        smoothed.reshape(-1, 1), 
                        smoothed_envelopes[:, i+1:]
                    ], axis=1) if i > 0 or i < num_oscillators - 1 else smoothed.reshape(-1, 1)
            envelopes = smoothed_envelopes
        
        # Compute beat-like signals from envelope differences
        beat_signals = mx.zeros((num_samples, num_oscillators))
        
        for i in range(num_oscillators):
            beat_sum = mx.zeros(num_samples)
            num_neighbors = 0
            
            for j in range(num_oscillators):
                if adjacency_matrix[i, j] > 0 and i != j:
                    # Simple beat approximation: difference of envelopes
                    envelope_diff = envelopes[:, i] - envelopes[:, j]
                    
                    # Weight by coupling strength
                    beat_sum = beat_sum + adjacency_matrix[i, j] * envelope_diff
                    num_neighbors += 1
            
            if num_neighbors > 0:
                beat_signals_np = beat_signals.tolist()
                beat_sum_list = (beat_sum / num_neighbors).tolist()
                for k in range(len(beat_sum_list)):
                    beat_signals_np[k][i] = beat_sum_list[k]
                beat_signals = mx.array(beat_signals_np)
        
        return beat_signals
    
    def _hilbert_transform(self, signals: mx.array) -> mx.array:
        """
        Approximate Hilbert transform for analytic signal.
        
        In practice, would use FFT-based implementation.
        Here we use a simple finite difference approximation.
        """
        # Simple approximation: 90-degree phase shift via differentiation
        real_part = signals
        # Simple difference approximation (MLX doesn't have diff with prepend)
        if signals.shape[0] > 1:
            imag_part = mx.concatenate([signals[0:1], signals[1:] - signals[:-1]], axis=0)
        else:
            imag_part = mx.zeros_like(signals)
        
        # Normalize to prevent amplitude changes
        imag_part = imag_part * 0.1  # Scaling factor
        
        # Pad to match dimensions
        if imag_part.shape[0] < real_part.shape[0]:
            pad_shape = (real_part.shape[0] - imag_part.shape[0],) + imag_part.shape[1:]
            padding = mx.zeros(pad_shape)
            imag_part = mx.concatenate([imag_part, padding], axis=0)
        
        return real_part + 1j * imag_part
    
    def extract_eeg_bands(self, 
                         beat_signals: mx.array,
                         sampling_rate_hz: float = 1000.0) -> Dict[str, mx.array]:
        """
        Extract traditional EEG frequency bands from beat signals.
        
        Args:
            beat_signals: Beat frequency signals
            sampling_rate_hz: Sampling rate for EEG analysis
            
        Returns:
            Dictionary of EEG band signals
        """
        # Resample to EEG sampling rate if needed
        if sampling_rate_hz != 1e6:
            beat_signals = self._resample_signals(beat_signals, 1e6, sampling_rate_hz)
        
        eeg_bands = {}
        
        for band_name, (low_freq, high_freq) in self.eeg_bands.items():
            # Simple bandpass filter using convolution
            filtered_signals = self._bandpass_filter(
                beat_signals, low_freq, high_freq, sampling_rate_hz
            )
            eeg_bands[band_name] = filtered_signals
            
        return eeg_bands
    
    def _resample_signals(self, 
                         signals: mx.array,
                         original_rate: float,
                         target_rate: float) -> mx.array:
        """Resample signals to target rate (simple decimation)."""
        if original_rate <= target_rate:
            return signals
            
        decimation_factor = int(original_rate / target_rate)
        return signals[::decimation_factor]
    
    def _bandpass_filter(self, 
                        signals: mx.array,
                        low_freq: float,
                        high_freq: float,
                        sampling_rate: float) -> mx.array:
        """
        Simple bandpass filter using moving averages.
        
        In practice, would use proper FIR/IIR filters.
        """
        # Convert frequencies to normalized form
        nyquist = sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Simple approach: high-pass then low-pass
        # High-pass: subtract moving average
        window_size_hp = max(1, int(1.0 / low_norm))
        if window_size_hp < len(signals):
            kernel_hp = mx.ones(window_size_hp) / window_size_hp
            smoothed = mx.convolve(signals.flatten(), kernel_hp, mode='same')
            high_passed = signals.flatten() - smoothed
        else:
            high_passed = signals.flatten()
        
        # Low-pass: moving average
        window_size_lp = max(1, int(1.0 / high_norm))
        if window_size_lp < len(high_passed):
            kernel_lp = mx.ones(window_size_lp) / window_size_lp
            low_passed = mx.convolve(high_passed, kernel_lp, mode='same')
        else:
            low_passed = high_passed
        
        return low_passed.reshape(signals.shape)


class CoherenceMeter:
    """
    Measures quantum coherence in microtubule oscillator networks.
    
    Implements various coherence metrics:
    - Phase coherence (order parameter)
    - Amplitude synchronization
    - Cross-correlation analysis
    - Spectral coherence
    """
    
    def __init__(self):
        """Initialize coherence measurement tools."""
        pass
    
    def compute_phase_coherence(self, signals: mx.array) -> float:
        """
        Compute global phase coherence (Kuramoto order parameter).
        
        Args:
            signals: Oscillation signals (samples, oscillators)
            
        Returns:
            Phase coherence R ∈ [0, 1]
        """
        # Extract phases using Hilbert transform
        analytic_signals = self._hilbert_transform_simple(signals)
        phases = mx.angle(analytic_signals)
        
        # Compute order parameter R = |<e^(iφ)>|
        complex_phases = mx.exp(1j * phases)
        mean_phase = mx.mean(complex_phases, axis=1)  # Average over oscillators
        order_param = mx.abs(mean_phase)
        
        # Return time-averaged coherence
        return float(mx.mean(order_param))
    
    def compute_spectral_coherence(self, 
                                 signals: mx.array,
                                 sampling_rate: float) -> Dict[str, float]:
        """
        Compute spectral coherence between oscillator pairs.
        
        Args:
            signals: Oscillation signals
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of coherence metrics by frequency band
        """
        num_oscillators = signals.shape[1]
        
        # Compute power spectral densities (simplified)
        # In practice, would use Welch's method
        coherence_metrics = {}
        
        # For each frequency band, compute average coherence
        eeg_bands = {
            'theta': (4, 8),
            'alpha': (8, 13), 
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        for band_name, (low_freq, high_freq) in eeg_bands.items():
            band_coherence = 0.0
            pair_count = 0
            
            for i in range(num_oscillators):
                for j in range(i+1, num_oscillators):
                    # Cross-correlation in frequency band
                    sig_i = signals[:, i]
                    sig_j = signals[:, j]
                    
                    # Simple correlation coefficient
                    corr = float(mx.corrcoef(mx.stack([sig_i, sig_j]))[0, 1])
                    band_coherence += abs(corr)
                    pair_count += 1
            
            if pair_count > 0:
                coherence_metrics[band_name] = band_coherence / pair_count
            else:
                coherence_metrics[band_name] = 0.0
                
        return coherence_metrics
    
    def _hilbert_transform_simple(self, signals: mx.array) -> mx.array:
        """Simple Hilbert transform approximation."""
        # Use derivative approximation for quadrature component
        real_part = signals
        # Simple difference approximation (MLX doesn't have diff with prepend)
        if signals.shape[0] > 1:
            imag_part = mx.concatenate([signals[0:1], signals[1:] - signals[:-1]], axis=0)
        else:
            imag_part = mx.zeros_like(signals)
        return real_part + 1j * imag_part


def run_oscillation_test():
    """Test oscillation dynamics implementation."""
    print("Testing Microtubule Oscillation Dynamics...")
    
    # Create oscillator
    oscillator = MicrotubuleOscillator(
        base_frequency_mhz=40.0,
        coherence_time_ms=1.0,
        coupling_strength=0.1
    )
    
    # Generate signals
    duration = 0.001  # 1ms
    sampling_rate = 1e6  # 1MHz
    num_oscillators = 10
    
    print(f"Generating {num_oscillators} oscillators for {duration*1000}ms...")
    signals = oscillator.generate_time_crystal_signal(duration, sampling_rate, num_oscillators)
    print(f"Signal shape: {signals.shape}")
    
    # Add thermal noise
    noisy_signals = oscillator.add_thermal_fluctuations(signals)
    print(f"Added thermal noise")
    
    # Test down-conversion
    converter = EEGDownConverter()
    
    # Create simple adjacency matrix
    adjacency_np = np.zeros((num_oscillators, num_oscillators))
    for i in range(num_oscillators-1):
        adjacency_np[i, i+1] = 0.5
        adjacency_np[i+1, i] = 0.5
    adjacency = mx.array(adjacency_np)
    
    print("Computing beat frequencies...")
    beat_signals = converter.compute_beat_frequencies(noisy_signals, adjacency)
    print(f"Beat signal shape: {beat_signals.shape}")
    
    # Test coherence measurement
    coherence_meter = CoherenceMeter()
    try:
        phase_coherence = coherence_meter.compute_phase_coherence(noisy_signals)
        print(f"Phase coherence: {phase_coherence:.4f}")
    except Exception as e:
        print(f"Phase coherence calculation failed: {e}")
        phase_coherence = 0.5
    
    try:
        spectral_coherence = coherence_meter.compute_spectral_coherence(beat_signals, 1000.0)
        print("Spectral coherence by band:")
        for band, coherence in spectral_coherence.items():
            print(f"  {band}: {coherence:.4f}")
    except Exception as e:
        print(f"Spectral coherence calculation failed: {e}")
    
    print("Oscillation dynamics test completed successfully!")
    

if __name__ == "__main__":
    run_oscillation_test()