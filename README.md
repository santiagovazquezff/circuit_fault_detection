# ML Circuit Fault Detection

Detects circuit faults (Healthy, Short R1, Open R1) from SPICE simulation waveforms using feature engineering + Random Forest.

## Features
Feature extraction: gain, phase, THD, harmonics, noise
RandomForest model (300 trees)
100% accuracy on train/test, ~94% accuracy on unseen frequencies
Validated with cross-validation, shuffle test, and duplicate check

## Repo Layout
"circuit_fault.ipynb" this is the Jupyter/Colab training notebook
"LICENSE" this is the MIT License (free to use)

## Quickstart
Clone this repo:
```bash
git clone https://github.com/YOUR-USERNAME/circuit_fault_detection.git
cd circuit_fault_detection
