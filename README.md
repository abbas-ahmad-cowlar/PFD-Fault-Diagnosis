# PFD Fault Diagnosis System - User Guide
**Production ML Pipeline for Hydrodynamic Bearing Fault Classification**

Version 2.0 | October 2025

---

## 1. SYSTEM OVERVIEW

### What This System Does

This automated machine learning pipeline diagnoses faults in hydrodynamic bearings (PFD - Palier Fluide Dynamique) by analyzing vibration signals. The system:

- **Generates** realistic synthetic fault data with configurable characteristics
- **Extracts** 36 advanced features from vibration signals using signal processing
- **Trains** multiple ML models (SVM, Random Forest, Neural Networks) 
- **Evaluates** performance with comprehensive metrics and visualizations
- **Deploys** the best model for production fault diagnosis

### Fault Types Detected (11 Classes)

**Single Faults (7):**
- Misalignment (désalignement)
- Imbalance (déséquilibre)  
- Bearing clearance (jeu)
- Lubrication issues
- Cavitation
- Wear (usure)
- Oil whirl

**Mixed Faults (3):**
- Misalignment + Imbalance
- Wear + Lubrication
- Cavitation + Clearance

**Baseline:**
- Healthy bearing (sain)

### Expected Performance

After running the complete pipeline, expect:
- **Test Accuracy:** 92-96%
- **Training Time:** 10-15 minutes total
- **Model Type:** Random Forest or Neural Network (automatically selected)
- **Output:** Production-ready inference function with comprehensive reports

---

## 2. QUICK START

### System Requirements

**Software:**
- MATLAB R2020b or later
- Required Toolboxes:
  - Statistics and Machine Learning Toolbox
  - Signal Processing Toolbox
  - Wavelet Toolbox
  - Parallel Computing Toolbox (recommended)

**Hardware:**
- Minimum: 8GB RAM, 4 CPU cores
- Recommended: 16GB RAM, 8+ CPU cores
- Disk Space: 2-5GB for datasets and results

### Installation

1. Extract all pipeline files to a working directory
2. Ensure MATLAB has access to all required toolboxes
3. No additional installation needed - ready to run

### Basic Execution

Run the pipeline in sequence:

```matlab
% Step 1: Generate training data
run('data_generator_v4.m')          % ~3-5 minutes, creates 1100-1500 samples

% Step 2-3: Feature extraction & exploration
run('pipeline_v4_1_2_3.m')          % ~2-3 minutes

% Step 4: Model training
run('pipeline_v4_4.m')              % ~10-12 minutes (with hyperparameter optimization)

% Step 5A: Evaluation
run('pipeline_v4_5A.m')             % ~1 minute

% Step 5B: Advanced validation
run('pipeline_v4_5B.m')             % ~2-3 minutes

% Step 5C: Production deployment
run('pipeline_v4_5C.m')             % ~30 seconds
```

**Total Time:** 15-25 minutes for complete pipeline execution

---

## 3. KEY CONFIGURATIONS

### Data Generator Configuration (`data_generator_v4.m`)

#### Essential Settings

**Dataset Size:**
```matlab
CONFIG.num_signals_per_fault = 100;  % Signals per fault type
                                      % Total: 100 × 11 = 1100 samples
                                      % Range: 50-200 recommended
```

**Fault Type Selection:**
```matlab
% Enable/disable entire fault categories
CONFIG.faults.include_single = true;   % Include 7 single faults
CONFIG.faults.include_mixed = true;    % Include 3 mixed fault combinations
CONFIG.faults.include_healthy = true;  % Include healthy baseline

% Individual fault control (only if include_single = true)
CONFIG.faults.single.desalignement = true;   % Enable/disable each fault
CONFIG.faults.single.desequilibre = true;
CONFIG.faults.single.jeu = true;
CONFIG.faults.single.lubrification = true;
CONFIG.faults.single.cavitation = true;
CONFIG.faults.single.usure = true;
CONFIG.faults.single.oilwhirl = true;

% Mixed fault control (only if include_mixed = true)
CONFIG.faults.mixed.misalign_imbalance = true;
CONFIG.faults.mixed.wear_lube = true;
CONFIG.faults.mixed.cavit_jeu = true;
```

**Impact:** Disabling fault types reduces dataset size and training time proportionally.

**Severity Levels:**
```matlab
CONFIG.severity.enabled = true;  % Enable multi-severity levels
CONFIG.severity.levels = {'incipient', 'mild', 'moderate', 'severe'};
```

**Data Augmentation:**
```matlab
CONFIG.augmentation.enabled = true;   % Enable data augmentation
CONFIG.augmentation.ratio = 0.30;     % 30% additional augmented samples
                                       % Increases dataset size by 30%
```

**Noise Configuration:**
```matlab
% Individual noise sources (enable/disable each)
CONFIG.noise.measurement = true;      % Sensor electronics noise
CONFIG.noise.emi = true;              % Electromagnetic interference
CONFIG.noise.pink = true;             // 1/f noise
CONFIG.noise.drift = true;            % Environmental drift
CONFIG.noise.quantization = true;     % ADC quantization
CONFIG.noise.sensor_drift = true;     % Sensor offset drift
CONFIG.noise.impulse = true;          % Sporadic impulses
```

**Recommendation:** Start with defaults, adjust based on your specific application.

---

### Pipeline Configuration (`pipeline_v4_1_2_3.m`)

#### Model Selection

```matlab
CONFIG.models.trainSVM = true;                % Support Vector Machine
CONFIG.models.trainRandomForest = true;       % Random Forest (usually best)
CONFIG.models.trainGradientBoosting = true;   % Gradient Boosting Trees
CONFIG.models.trainNeuralNetwork = true;      // 3-layer MLP
CONFIG.models.trainStackedEnsemble = false;   % Meta-learner (unstable)
```

**Impact:** More models = longer training time but better model comparison.

#### Feature Engineering

```matlab
CONFIG.includeAdvancedFeatures = false;  % CWT, WPT, Non-linear dynamics
                                          % false: 36 features, fast (~2 min)
                                          % true: 52 features, slow (~20 min)
```

**Recommendation:** Keep `false` unless you need maximum accuracy (minimal gain: ~1%).

#### Data Splitting

```matlab
CONFIG.trainRatio = 0.70;   % 70% training
CONFIG.valRatio = 0.15;     % 15% validation  
CONFIG.testRatio = 0.15;    % 15% test
```

**Standard practice:** 70/15/15 split ensures robust evaluation.

#### Hyperparameter Optimization

```matlab
CONFIG.hyperOptIterations = 50;  % Bayesian optimization iterations
                                  // 30: fast (~5 min), 50: balanced (~10 min)
                                  // 100: thorough (~20 min)
```

#### Feature Selection

```matlab
CONFIG.useFeatureSelection = true;        // Enable feature selection
CONFIG.numFeaturesToSelect = 15;          // Select top 15 features
CONFIG.performFeatureSelectionAfterSplit = true;  // Prevent data leakage
```

#### Development Mode (Fast Testing)

```matlab
CONFIG.developmentMode = true;   // Quick testing mode
                                  // Reduces hyperOptIterations to 10
                                  // Disables some visualizations
                                  // Use for rapid iteration only
```

---

## 4. OUTPUT FILES & RESULTS

### Generated Outputs

After complete execution, find in `PFD_SVM_Results_Production/`:

**Data Files:**
- `features_pfd_production.csv` - Extracted feature matrix
- `dataset_metadata.mat` - Dataset configuration and metadata
- `Best_PFD_Model_Production.mat` - Trained model package
- `step4_results.mat`, `step5a_results.mat`, `step5b_results.mat` - Intermediate results

**Visualizations (9 figures):**
- `Fig0_PFD_Signal_Examples.png` - Time/frequency/spectrogram views
- `Fig1_Feature_Correlation.png` - Feature correlation heatmap
- `Fig2_Feature_Distributions.png` - Feature distributions by class
- `Fig3_tSNE_Clusters.png` - 2D feature space visualization
- `Fig4_Confusion_Matrix.png` - Classification confusion matrix
- `Fig5_ROC_Curves.png` - ROC curves (one-vs-rest)
- `Fig7_Class_Distribution.png` - Dataset class balance
- `Fig8_Performance_Comparison.png` - Model comparison chart

**Reports:**
- `PFD_Analysis_Report_Production.txt` - Comprehensive performance report
- `pipeline_run_production.log` - Complete execution log

**Inference Function:**
- `predictPFDFault_Production.m` - Standalone prediction function

### Key Metrics to Review

**Overall Performance:**
- Test Accuracy: 92-96% (excellent)
- Macro F1-Score: >93% (balanced performance)
- Mean AUC: >0.98 (excellent discrimination)

**Per-Class Metrics:**
- Precision/Recall/F1 for each fault type
- Identify problematic classes (typically mixed faults)

**Robustness:**
- Sensor noise tolerance: 75-85% accuracy (moderate degradation)
- Missing features tolerance: 85-92% accuracy
- Temporal drift: 90-95% accuracy

---

## 5. COMMON USE CASES

### Scenario 1: Standard Training (Default Settings)

**Goal:** Train a general-purpose fault classifier

**Configuration:**
- Keep all default settings
- All 11 fault types enabled
- 100 samples per fault
- 50 hyperparameter optimization iterations

**Expected:** 94-96% accuracy, 15-minute runtime

---

### Scenario 2: Fast Development Iteration

**Goal:** Quick testing of changes

**Configuration:**
```matlab
% In data_generator_v4.m:
CONFIG.num_signals_per_fault = 50;  % Reduce dataset size

% In pipeline_v4_1_2_3.m:
CONFIG.developmentMode = true;       % Fast mode
CONFIG.hyperOptIterations = 10;      // Minimal optimization
```

**Expected:** 88-92% accuracy, 7-minute runtime

---

### Scenario 3: Single Faults Only

**Goal:** Focus on individual fault types (simpler problem)

**Configuration:**
```matlab
% In data_generator_v4.m:
CONFIG.faults.include_mixed = false;  // Disable mixed faults
```

**Expected:** 95-98% accuracy (easier problem), 8 classes instead of 11

---

### Scenario 4: Maximum Accuracy

**Goal:** Achieve highest possible performance

**Configuration:**
```matlab
// In data_generator_v4.m:
CONFIG.num_signals_per_fault = 150;  % More training data

% In pipeline_v4_1_2_3.m:
CONFIG.includeAdvancedFeatures = true;   % All 52 features
CONFIG.hyperOptIterations = 100;         % Thorough optimization
```

**Expected:** 96-98% accuracy, 40-minute runtime

---

### Scenario 5: Custom Noise Profile

**Goal:** Test robustness to specific noise sources

**Configuration:**
```matlab
% In data_generator_v4.m:
% Disable all noise except target type
CONFIG.noise.measurement = false;
CONFIG.noise.emi = true;        % Only EMI noise
CONFIG.noise.pink = false;
CONFIG.noise.drift = false;
CONFIG.noise.quantization = false;
CONFIG.noise.sensor_drift = false;
CONFIG.noise.impulse = false;
```

**Expected:** 93-96% accuracy, evaluate EMI-specific performance

---

## 6. PRODUCTION DEPLOYMENT

### Using the Trained Model

After pipeline completion, use the generated inference function:

```matlab
% Load a new signal
data = load('path/to/new_signal.mat');
signal = data.x;
fs = data.fs;

% Predict fault
[predictedFault, scores] = predictPFDFault_Production(signal, fs);

% Display result
fprintf('Predicted Fault: %s\n', predictedFault);
fprintf('Confidence: %.2f%%\n', max(scores)*100);
```

### Model Retraining

Retrain when:
- New fault types emerge
- Operating conditions change significantly
- Accuracy degrades below 90% in production
- Dataset grows substantially (>50% more samples)

**Retraining frequency:** Every 3-6 months or when performance degrades

---

## 7. TROUBLESHOOTING

### Issue: "No .mat files found"
**Cause:** Data generator not run or wrong directory
**Fix:** Run `data_generator_v4.m` first, verify `data_signaux_sep_production/` exists

### Issue: SVM training fails
**Cause:** Using old pipeline version
**Fix:** Ensure using latest `pipeline_v4_4.m` with fitcecoc

### Issue: Out of memory
**Cause:** Too many samples or advanced features enabled
**Fix:** Reduce `CONFIG.num_signals_per_fault` or disable advanced features

### Issue: Low accuracy (<85%)
**Cause:** Insufficient training data or poor hyperparameters
**Fix:** Increase `num_signals_per_fault` to 150 or `hyperOptIterations` to 100

### Issue: Very slow execution
**Cause:** Advanced features enabled or no parallel computing
**Fix:** Set `CONFIG.includeAdvancedFeatures = false`, enable parallel pool

---

## 8. BEST PRACTICES

✅ **Always run complete pipeline sequence** (data_generator → step 1-3 → step 4 → step 5A/B/C)

✅ **Review confusion matrix** to identify problematic fault pairs

✅ **Check per-class metrics** - ensure all classes have >85% recall

✅ **Validate on holdout set** - confirm generalization (should match test accuracy ±3%)

✅ **Monitor robustness tests** - sensor noise tolerance indicates real-world viability

✅ **Save configuration** - document exact CONFIG settings used for reproducibility

✅ **Version models** - keep track of model versions for production deployment

✅ **Test inference function** - validate on sample signals before production deployment

---

## 9. SUPPORT & NEXT STEPS

### Getting Started Checklist

- [ ] Install MATLAB R2020b+ with required toolboxes
- [ ] Extract pipeline files to working directory
- [ ] Run `data_generator_v4.m` with default settings
- [ ] Run complete pipeline (steps 1-3, 4, 5A, 5B, 5C)
- [ ] Review generated figures and reports
- [ ] Test inference function with sample signals
- [ ] Adjust configurations based on your requirements
- [ ] Deploy model to production system

### Additional Resources

- `pipeline_run_production.log` - Detailed execution trace
- `PFD_Analysis_Report_Production.txt` - Comprehensive results
- Generated figures - Visual analysis of performance

---

**Pipeline Version:** Production v2.0  
**Last Updated:** October 30, 2025  
**Status:** Production-Ready ✅
