# Circuit Fault Detection with Machine Learning  
**Refer to main_report for a more comprehensive report explaining the model in depth and a user guide for the demo.** https://github.com/santiagovazquezff/circuit_fault_detection/blob/main/main_report/Classification%20Machine%20Learning%20Model%20for%20Faults%20in%20a%20Non-Inverting%20Amplifier.pdf

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
<img width="572" height="441" alt="image" src="https://github.com/user-attachments/assets/18b5304b-fb7f-43aa-8d02-6b66389ba3f4" />

  - Short fault configuration (R1 removed). See schematic below:  
<img width="548" height="435" alt="image" src="https://github.com/user-attachments/assets/d29fe8c8-3593-4d50-9d3b-160dd0a9518a" />

  - Open fault configuration (R1 = 1e9 Ω). See schematic below:  
<img width="556" height="427" alt="image" src="https://github.com/user-attachments/assets/c88b8697-1683-4513-aa2f-091a81bb8908" />

- Simulation setup:  
  - Input: sinusoidal source, amplitude = 0.283 V.  
  - Frequencies: 200, 500, 800, 1000, 1500, 2000 Hz.  
  - Transient analysis: 0.12 s total, 2.5 µs time step.  
  - Monte Carlo: 50 runs per frequency with Gaussian 5% resistor tolerance.  
  - Example transient analysis setup is shown below:  
 <img width="947" height="582" alt="image" src="https://github.com/user-attachments/assets/372970d8-572e-47f4-b910-ce0c01e22187" />
 
- Total dataset: 900 runs (3 classes × 6 frequencies × 50 runs).  
- Storage format: CSV files with one time column (“s”) and 50 voltage runs (“vout1 … vout50”).  
- Example folder structure:  
<img width="914" height="268" alt="image" src="https://github.com/user-attachments/assets/f0fc885c-a12a-44d4-aa69-dd2a214381c8" />

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
The dataset is split 80/20 (train/test) with stratification. Trained models and metadata are saved in model_artifacts/: https://github.com/santiagovazquezff/circuit_fault_detection/tree/main/model_artifacts

---
## Results
### Train and Test (with f0 included)

Perfect separation between classes was achieved. Both training and test sets reached 100% accuracy.
<img width="783" height="213" alt="image" src="https://github.com/user-attachments/assets/aaee9fc4-4073-4833-b82a-824a5b308d1d" />

Confusion matrices for train and test sets are shown below:
<img width="957" height="439" alt="image" src="https://github.com/user-attachments/assets/887fbe17-25be-41dc-a012-39ca4087ec70" />

### Cross-Validation

5-fold cross-validation confirmed stability:
<img width="813" height="169" alt="image" src="https://github.com/user-attachments/assets/93aa614a-881a-4e04-84b3-7815318d2a4c" />

### Frequency-Held-Out Validation (without f0)

When excitation frequency was excluded, GroupKFold validation across frequencies showed high but imperfect generalisation:
<img width="802" height="163" alt="image" src="https://github.com/user-attachments/assets/686f2944-89ff-4732-8a63-b86882bf547c" />

### Label Shuffle Control

When labels were randomly permuted, accuracy dropped to chance level (≈0.25).
<img width="824" height="159" alt="image" src="https://github.com/user-attachments/assets/5168225f-7c30-489d-89dd-471a51515d46" />

### Duplicate Check

No exact duplicates were found between training and test sets.
<img width="821" height="163" alt="image" src="https://github.com/user-attachments/assets/18919c79-f9cd-4e20-930b-80382e21ffdb" />

---

## Limitations

- Dataset scope: only three fault classes on one amplifier topology.
- Synthetic data: Altium simulations do not fully capture real-world noise and component variability.
- Feature set: limited to first three harmonics; severe nonlinearities not represented.
- Perfect scores reflect dataset separability more than real-world robustness.

---

## Interactive Demo

**An interactive demo is provided in demo/demo.ipynb.**
https://github.com/santiagovazquezff/circuit_fault_detection/blob/main/demo/demo.ipynb.

Steps to run:
1. Download one of the example files (demo_file_1.csv, demo_file_2.csv, demo_file_3.csv) from the demo
 folder.
<img width="685" height="337" alt="image" src="https://github.com/user-attachments/assets/ea765fa5-5882-45c3-b3c3-2d1dcc8b0b7b" />

2. Open demo.ipynb in Colab using the “Open in Colab” button.
<img width="380" height="107" alt="image" src="https://github.com/user-attachments/assets/ed334e88-e89c-4ed3-b655-4f399fcc048f" />

3. Run all cells.
<img width="312" height="158" alt="image" src="https://github.com/user-attachments/assets/39ca18ad-6819-42ad-b0e8-60fe08b74107" />

4. Upload the chosen CSV via the provided file chooser.
<img width="1089" height="207" alt="image" src="https://github.com/user-attachments/assets/d894f64d-4f12-4e14-a0b1-2370ce969965" />

5. Enter the corresponding amplitude and frequency values.
<img width="1021" height="85" alt="image" src="https://github.com/user-attachments/assets/55b59d5a-33ba-458f-a72e-a77878da390d" />

7. View the predicted class, probabilities, and waveform plots.
<img width="656" height="458" alt="image" src="https://github.com/user-attachments/assets/f08a9676-4815-4b36-9b39-81fe4e433845" />

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
