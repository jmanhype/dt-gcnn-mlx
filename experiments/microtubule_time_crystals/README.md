# Microtubule Time Crystal Experiment

**Experiment Date**: August 03, 2025, 11:56 AM CDT  
**Inspired by**: @The_Utility_Co (X ID: 1951860518391816468) and @StuartHameroff's microtubule research

## Objective
Simulate microtubule time crystals as MHz oscillators, exploring coherence in thermal chaos, down-conversion to EEG rhythms, and predictive coding using DT-GCNN-MLX.

## Hypothesis
Microtubules form time crystals oscillating at MHz/GHz frequencies, down-converting to EEG via interference beats, supporting biological predictive coding.

## Experiment Structure

```
microtubule_time_crystals/
├── models/           # Adapted DT-GCNN models for microtubules
├── data/            # Microtubule structure and oscillation data
├── analysis/        # Coherence and spectral analysis tools
├── visualization/   # t-SNE, spectrograms, coherence plots
├── configs/         # Experiment configurations
└── run_experiment.py # Main experiment runner
```

## Key Components
1. **Microtubule Graph**: Nodes = tubulin dimers, edges = inter-segment interactions
2. **MHz Oscillations**: 1-100 MHz sinusoidal signals with thermal noise
3. **Down-conversion**: Wavelet decomposition to theta/gamma EEG bands
4. **Predictive Coding**: VAE-based next-state prediction
5. **Coherence Analysis**: Phase alignment metrics under thermal chaos

## Expected Outcomes
- Validation of microtubule time crystal behavior
- EEG frequency down-conversion mechanisms
- Coherence stability in biological noise
- DT-GCNN feasibility for neural predictive coding