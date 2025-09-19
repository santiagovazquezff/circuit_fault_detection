# Circuit Fault Detection with Machine Learning  
**Refer to main_report: for a more comprehensive report explaining the model in depth and a user guide for the demo.**

This repository implements a machine learning pipeline to automatically classify faults in a simple non-inverting amplifier circuit. The model distinguishes between three circuit states:  

- Healthy  
- Open fault  
- Short fault  

The classification is performed using simulated oscilloscope waveforms generated in **Altium Designer** and a **Random Forest** model implemented with **scikit-learn**.  

The repository includes dataset generation, feature extraction, model training and evaluation, and an interactive demo that can be run directly in Google Colab.  

---

## Contents  

- Dataset generation using Altium Designer (LM324N amplifier, Monte Carlo analysis).  
- Feature extraction from time-domain waveforms (harmonic amplitudes, phase, THD, residual noise).  
- Model training with Random Forest.  
- Evaluation including train/test accuracy, cross-validation, frequency-held-out validation, label shuffle, and duplicate checks.  
- Interactive Colab demo for classifying new oscilloscope-like CSV files.  

---

## Dataset  

- Circuit: LM324N non-inverting amplifier.  
  - Healthy configuration (Rf = 10 kΩ, R1 = 1 kΩ). See schematic below:  
  *** Insert Healthy schematic image here ***  
  - Short fault configuration (R1 removed). See schematic below:  
  *** Insert Short schematic image here ***  
  - Open fault configuration (R1 = 1e9 Ω). See schematic below:  
  *** Insert Open schematic image here ***  

- Simulation setup:  
  - Input: sinusoidal source, amplitude = 0.283 V.  
  - Frequencies: 200, 500, 800, 1000, 1500, 2000 Hz.  
  - Transient analysis: 0.12 s total, 2.5 µs time step.  
  - Monte Carlo: 50 runs per frequency with Gaussian 5% resistor tolerance.  
  - Example transient analysis setup is shown below:  
  *** Insert Altium transient analysis screenshot here ***  

- Total dataset: 900 runs (3 classes × 6 frequencies × 50 runs).  
- Storage format: CSV files with one time column (“s”) and 50 voltage runs (“vout1 … vout50”).  
- Example folder structure:  
  *** Insert CSV folder structure screenshot here ***  

---

## Pipeline  

### 1. Data Loading  
CSV files are read into a time vector and a matrix of Monte Carlo runs using:  

```python
def read_multirun_vout_csv(path: str, n_keep=50):
    df = pd.read_csv(path)
    t = df["s"].to_numpy(float)
    V = df.iloc[:, 1:1+n_keep].to_numpy(float)
    return t, V
````

Each column corresponds to one waveform, ensuring consistent structure across all classes.

### 2. Standardisation 
Waveforms are cropped to the first 0.1 s and resampled to 16,384 points on a uniform grid:
```python
def standardise(t, v, time_window=0.1, n_points=16384):
    t0 = t[0]
    mask = (t - t0) < time_window
    t2, v2 = t[mask], v[mask]
    t_fit = np.linspace(0.0, time_window, n_points, endpoint=False)
    v_fit = np.interp(t_fit, t2 - t0, v2)
    return t_fit, v_fit
```
This ensures that all signals are directly comparable regardless of simulation sampling density.
An illustration of raw vs resampled waveform is shown below:

### 3. Feature Extraction

From each waveform, six physically interpretable features are extracted:
- A1 (fundamental amplitude at f0)
- Phase at f0
- THD (based on 2nd and 3rd harmonics)
- Residual noise RMS
- A2 and A3 (2nd and 3rd harmonic amplitudes)

This is implemented using:
```python
def lockin_features(v_fit, t_fit, f0):
    x, tt = v_fit.astype(float), t_fit.astype(float)
    dc = float(np.mean(x)); xz = x - dc; N = x.size

    def ap(freq):
        w = 2*np.pi*freq
        c = np.cos(w*tt); s = np.sin(w*tt)
        a = (2.0/N)*np.dot(xz, c)
        b = (2.0/N)*np.dot(xz, s)
        A = float(np.hypot(a, b))
        ph = float(np.arctan2(-b, a))
        return A, ph

    A1, ph = ap(f0); A2,_ = ap(2*f0); A3,_ = ap(3*f0)
    thd = (np.sqrt(A2**2 + A3**2)/A1) if A1>0 else 0.0

    ms_total = float(np.mean(x**2)); ms_dc = dc**2
    ms_tones = (A1**2 + A2**2 + A3**2)/2
    noise_rms = float(np.sqrt(max(ms_total - ms_dc - ms_tones, 0.0)))

    return A1, ph, thd, noise_rms, A2, A3
```
A schematic of harmonic extraction is shown below:

### 4. Dataset Assembly

Features are normalised by input amplitude, and optionally the excitation frequency (f0) is added:
```python
X, y, feat_names = build_dataset()
```
This produces X (features), y (labels), and feat_names.

### 5. Model Training

A Random Forest classifier is trained with 300 estimators and balanced class weights:
```python 
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
rf.fit(Xtr, ytr)
```
The dataset is split 80/20 (train/test) with stratification. Trained models and metadata are saved in model_artifacts/.

---
## Results
### Train and Test (with f0 included)

Perfect separation between classes was achieved. Both training and test sets reached 100% accuracy.
Insert table

Confusion matrices for train and test sets are shown below:

### Cross-Validation

5-fold cross-validation confirmed stability:
Insert table

### Frequency-Held-Out Validation (without f0)

When excitation frequency was excluded, GroupKFold validation across frequencies showed high but imperfect generalisation:

Insert table 

### Label Shuffle Control

When labels were randomly permuted, accuracy dropped to chance level (≈0.25).

Insert table

### Duplicate Check

No exact duplicates were found between training and test sets.

---

## Limitations

- Dataset scope: only three fault classes on one amplifier topology.
- Synthetic data: Altium simulations do not fully capture real-world noise and component variability.
- Feature set: limited to first three harmonics; severe nonlinearities not represented.
- Perfect scores reflect dataset separability more than real-world robustness.

---

## Interactive Demo

**An interactive demo is provided in demo/demo.ipynb: https://github.com/santiagovazquezff/circuit_fault_detection/blob/main/demo/demo.ipynb.**

Steps to run:
1. Download one of the example files (demo_file_1.csv, demo_file_2.csv, demo_file_3.csv) from the demo
 folder.
2. Open demo.ipynb in Colab using the “Open in Colab” button.
Screenshot of the button below:
3. Run all cells.
4. Upload the chosen CSV via the provided file chooser.

Screenshot of the file upload box below:

5. Enter the corresponding amplitude and frequency values.
6. View the predicted class, probabilities, and waveform plots.
Example output is shown below:

Each Colab session runs as a temporary copy; no changes are saved to the repository.

---

## Installation

Clone the repository:
```python
git clone https://github.com/santiagovazquezff/circuit_fault_detection.git
cd circuit_fault_detection
```

Install dependencies (Python 3.9+, NumPy, Pandas, scikit-learn, Matplotlib required):
```python 
pip install -r requirements.txt
```
