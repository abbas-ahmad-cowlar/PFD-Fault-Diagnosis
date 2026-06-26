# PFD Fault Diagnosis — Hydrodynamic Bearing Fault Classification

A MATLAB machine-learning pipeline for diagnosing faults in hydrodynamic
(fluid-film) bearings — *paliers fluide dynamique*, hence **PFD** — from their
vibration signatures. The project covers the full chain: physics-based synthetic
signal generation, multi-domain feature extraction, model training and
selection, robustness evaluation, and a standalone inference function for
deployment.

On the held-out test set the selected Random Forest model reaches **95.33%
accuracy** across 11 classes (macro F1 95.37%, mean AUC 0.997).

---

## What the pipeline does

- **Generates** synthetic vibration data using a Sommerfeld hydrodynamic-bearing
  model with configurable operating conditions, an 8-layer noise model,
  severity control, and transient behaviour.
- **Extracts** 36 time-, frequency-, and envelope-domain features (52 with the
  optional advanced wavelet/non-linear set).
- **Trains** and compares SVM, Random Forest, and a neural network with Bayesian
  hyperparameter optimisation and MRMR feature selection.
- **Evaluates** each model with confusion matrices, one-vs-rest ROC curves,
  per-class metrics, and robustness tests (sensor noise, missing features,
  temporal drift).
- **Exports** the best model as a standalone `predictPFDFault` inference
  function plus a full text report and figure set.

### Fault classes (11)

**Single faults (7):** misalignment (*désalignement*), imbalance
(*déséquilibre*), bearing clearance (*jeu*), lubrication fault
(*lubrification*), cavitation, wear (*usure*), oil whirl.

**Mixed faults (3):** misalignment + imbalance, wear + lubrication,
cavitation + clearance.

**Baseline (1):** healthy bearing (*sain*).

---

## Requirements

- MATLAB R2020b or later (R2024b recommended for the Simulink model).
- Statistics and Machine Learning Toolbox
- Signal Processing Toolbox
- Wavelet Toolbox
- Parallel Computing Toolbox (optional, speeds up training)

Recommended hardware: 16 GB RAM, 8+ CPU cores. A full run uses 2–5 GB of disk
for the generated dataset and results.

---

## Quick start

```matlab
% 1. Generate the training data (~3-5 min, ~1,100-1,500 signals)
run('generator.m')

% 2. Run the full pipeline: feature extraction, training,
%    evaluation, and deployment (~30-40 min with hyperparameter search)
run('pipeline.m')
```

A complete run takes roughly 30–45 minutes. All artefacts land in
`PFD_SVM_Results_Production/`.

### Using the trained model

```matlab
data = load('path/to/new_signal.mat');
[predictedFault, scores] = predictPFDFault_Production(data.x, data.fs);
fprintf('Predicted fault: %s (%.1f%% confidence)\n', ...
        predictedFault, max(scores) * 100);
```

---

## Key configuration

Both `generator.m` and `pipeline.m` are driven by a `CONFIG` struct at the top
of the file. The most useful knobs:

**Dataset size** (`generator.m`)
```matlab
CONFIG.num_signals_per_fault = 100;   % 50-200 recommended
CONFIG.augmentation.ratio    = 0.30;  % 30% extra augmented samples
```

**Model selection** (`pipeline.m`)
```matlab
CONFIG.models.trainSVM          = true;
CONFIG.models.trainRandomForest = true;   % usually the best model
CONFIG.models.trainNeuralNetwork = true;
```

**Feature engineering** (`pipeline.m`)
```matlab
CONFIG.includeAdvancedFeatures = false;  % false: 36 features, fast
                                         % true:  52 features, slower
CONFIG.useFeatureSelection   = true;     % MRMR selection
CONFIG.numFeaturesToSelect   = 15;
```

**Data split and tuning** (`pipeline.m`)
```matlab
CONFIG.trainRatio = 0.70;  CONFIG.valRatio = 0.15;  CONFIG.testRatio = 0.15;
CONFIG.hyperOptIterations = 50;   % Bayesian optimisation iterations
CONFIG.developmentMode    = false; % true = fast iteration, reduced search
```

Disabling fault categories or reducing `num_signals_per_fault` shortens runtime
proportionally. Accuracy on a given configuration depends on dataset size and
the optimisation budget; expect the mid-90s for the default settings.

---

## Outputs

After a full run, `PFD_SVM_Results_Production/` contains:

- `Best_PFD_Model_Production.mat` — the trained, deployable model package.
- `predictPFDFault_Production.m` — standalone inference function.
- `features_pfd_production.csv`, `dataset_metadata.mat` — feature matrix and
  metadata.
- `Fig*.png` — feature analysis, confusion matrices, ROC curves, model
  comparison.
- `PFD_Analysis_Report_Production.txt` — full performance report.
- `pipeline_run_production.log` — execution trace.

---

## Repository layout

```
generator.m                  Physics-based signal generator
pipeline.m                   Feature extraction, training, evaluation, deployment
compute_thd_all.m            THD analysis across all fault types
compute_diagnostics_all.m    Kurtosis / THD / band-energy diagnostics
Technical_report.tex         LaTeX technical report
LiveScripts/                 Signal-analysis scripts (healthy, single, mixed, comparative)
Figures/                     Per-fault time/frequency/envelope figures
PFD_SVM_Results_Production/   Pre-computed results, model, and reports
Simulink/                    Simulink block-diagram version of generator.m
data_signaux_sep_production/ Generated signal dataset
```

The Simulink model is a visual, block-diagram equivalent of `generator.m`: same
physics, same outputs, presented as a flowchart. See `Simulink/README.md`.

---

## Notes

- Synthetic training data limits direct transfer to physical machines; the
  robustness tests (noise, missing features, drift) are included to probe
  generalisation, but real-sensor validation is future work.
- Time-domain signals look similar across fault types — the discriminating
  information lives in the frequency and envelope domains. Use `pwelch` to
  inspect spectral signatures.
- THD is reported for completeness but is a poor standalone fault indicator;
  kurtosis and spectral band energy separate the classes far better.
