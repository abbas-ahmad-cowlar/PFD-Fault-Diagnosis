# PFD Signal Generator — Simulink Model

## Overview

This Simulink model provides a **visual block diagram representation** of the signal generation process used in `generator.m`. It demonstrates how synthetic vibration signals for hydrodynamic bearing (PFD) fault diagnosis are constructed step by step.

## Requirements

- MATLAB R2024b (or later)
- Simulink
- Signal Processing Toolbox (for `pwelch` in runner scripts)

## Files

| File | Purpose |
|---|---|
| `build_simulink_model.m` | Creates the `.slx` model programmatically |
| `run_pfd_simulation.m` | Interactive: simulate one signal with chosen fault |
| `run_pfd_batch.m` | Batch: generate 1,430 signals (matches `generator.m`) |
| `builds/` | All versioned builds (never overwritten) |
| `latest/PFD_Signal_Generator.slx` | Latest model (clean name) |

## Quick Start

### 1. Build the model

```matlab
cd Simulink
build_simulink_model()
```

This creates `PFD_Signal_Generator.slx` in `builds/` and `latest/`.

### 2. Run a single simulation

```matlab
run_pfd_simulation()
```

Follow the menu to select fault type, severity, and transient behavior.

### 3. Batch generation

```matlab
run_pfd_batch()
```

Generates 1,430 signals (11 fault types × 130 each) to `data_signaux_sep_simulink/`.

## Model Architecture

```
PFD_Signal_Generator.slx
│
├── Operating_Conditions    Speed, Load, Temperature → Omega, Sommerfeld
├── Base_Signal             Baseline vibration noise
├── Fault_Injection         11 fault types (selectable)
├── Severity_Control        Severity level with optional temporal evolution
├── Transient_Behavior      Speed ramp / Load step / Thermal expansion
├── Noise_Model             8 noise layers (measurement, EMI, pink, etc.)
├── Signal_Sum              Combines: Base + Fault×Severity×Transient + Noise
├── Quantizer               ADC quantization effect
├── Scope                   Real-time visualization
└── To_Workspace            Exports signal as 'x_sim'
```

## Fault Types

| # | Code | Name | Simulink Implementation |
|---|---|---|---|
| 1 | sain | Healthy | No fault signature |
| 2 | desalignement | Misalignment | 2X + 3X harmonics |
| 3 | desequilibre | Imbalance | 1X × speed² |
| 4 | jeu | Clearance | 0.43X + 1X + 2X |
| 5 | lubrification | Lubrication | Stick-slip (inverse Sommerfeld) |
| 6 | cavitation | Cavitation | HF bursts at 2000 Hz |
| 7 | usure | Wear | Amplitude-modulated harmonics |
| 8 | oilwhirl | Oil Whirl | 0.45X × 1/√S |
| 9 | mixed_misalign_imbalance | Mixed M+I | #2 + #3 combined |
| 10 | mixed_wear_lube | Mixed W+L | #7 + #5 combined |
| 11 | mixed_cavit_jeu | Mixed C+J | #6 + #4 combined |

## Tunable Parameters

Open the model and double-click any subsystem to modify:

- **Operating_Conditions**: `Speed_RPM`, `Load_Percent`, `Temperature_C`
- **Fault_Injection**: `Fault_Type` (1-11)
- **Severity_Control**: `Severity_Level` (0-1), `Enable_Evolution` (0/1)
- **Transient_Behavior**: `Transient_Type` (0=none, 1=ramp, 2=step, 3=thermal)

## Versioning

Every run of `build_simulink_model()` creates a new timestamped file:
```
builds/PFD_Signal_Generator_v1_20260425_0930.slx
builds/PFD_Signal_Generator_v2_20260425_1415.slx
```

The `latest/` folder always contains the most recent build with a clean name.
