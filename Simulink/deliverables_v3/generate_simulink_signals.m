function generate_simulink_signals()
% GENERATE_SIMULINK_SIGNALS  Generate one signal per fault type from Simulink
%
% Builds the model, runs it for each of the 11 fault types with fixed
% parameters (3600 RPM, 70% load, 60°C), and saves .mat files.
%
% Output: data_signaux_simulink/ directory with 11 .mat files
%
% Usage:
%   >> generate_simulink_signals()

fprintf('\n========================================\n');
fprintf('  Simulink Signal Generator\n');
fprintf('========================================\n\n');

%% Configuration
fs = 20480;
T  = 5;
speed_rpm  = 3600;   % 60 Hz
load_pct   = 70;     % percent
temp_C     = 60;     % Celsius
severity   = 0.7;    % default severity for faults
enable_evo = 0;      % no temporal evolution
trans_type = 1;      % 1 = None (1-based Multiport Switch)

% Fault definitions
fault_defs = {
    'sain',                      1,  'Healthy'
    'desalignement',             2,  'Misalignment'
    'desequilibre',              3,  'Imbalance'
    'jeu',                       4,  'Clearance'
    'lubrification',             5,  'Lubrication'
    'cavitation',                6,  'Cavitation'
    'usure',                     7,  'Wear'
    'oilwhirl',                  8,  'Oil Whirl'
    'mixed_misalign_imbalance',  9,  'Mixed: Misalign+Imbalance'
    'mixed_wear_lube',           10, 'Mixed: Wear+Lube'
    'mixed_cavit_jeu',           11, 'Mixed: Cavit+Clearance'
};

%% Output directory
scriptDir = fileparts(mfilename('fullpath'));
outputDir = fullfile(fileparts(scriptDir), 'data_signaux_simulink');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('Created output directory: %s\n', outputDir);
end

%% Step 1: Build the model
fprintf('Step 1: Building Simulink model...\n');

% Add deliverables_v3 to path temporarily so build_simulink_model is found
addpath(scriptDir);

% Build (creates/rebuilds the .slx model)
build_simulink_model();

fprintf('  ✓ Model built successfully\n\n');

%% Step 2: Load the model
fprintf('Step 2: Loading model...\n');

modelName = 'PFD_Signal_Generator';
latestDir = fullfile(scriptDir, 'latest');
modelFile = fullfile(latestDir, [modelName '.slx']);

if ~exist(modelFile, 'file')
    error('Model file not found at: %s', modelFile);
end

if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
load_system(modelFile);
fprintf('  ✓ Model loaded: %s\n\n', modelFile);

%% Step 3: Generate signals
fprintf('Step 3: Generating signals...\n');
fprintf('  Parameters: %d RPM, %d%% load, %d°C, severity=%.1f\n\n', ...
    speed_rpm, load_pct, temp_C, severity);

num_faults = size(fault_defs, 1);
gen_start = tic;

for k = 1:num_faults
    fault_name = fault_defs{k, 1};
    fault_code = fault_defs{k, 2};
    fault_desc = fault_defs{k, 3};
    
    fprintf('  [%2d/%d] %s (%s)... ', k, num_faults, fault_name, fault_desc);
    
    % Set fixed parameters
    set_param([modelName '/Operating_Conditions/Speed_RPM'], ...
        'Value', num2str(speed_rpm));
    set_param([modelName '/Operating_Conditions/Load_Percent'], ...
        'Value', num2str(load_pct));
    set_param([modelName '/Operating_Conditions/Temperature_C'], ...
        'Value', num2str(temp_C));
    set_param([modelName '/Fault_Injection/Fault_Type'], ...
        'Value', num2str(fault_code));
    
    % Severity: healthy gets 1.0 (nominal), faults get specified severity
    if fault_code == 1
        set_param([modelName '/Severity_Control/Severity_Level'], 'Value', '1.0');
    else
        set_param([modelName '/Severity_Control/Severity_Level'], ...
            'Value', num2str(severity));
    end
    
    set_param([modelName '/Severity_Control/Enable_Evolution'], ...
        'Value', num2str(enable_evo));
    set_param([modelName '/Transient_Behavior/Transient_Type'], ...
        'Value', num2str(trans_type));
    
    % Run simulation with ReturnWorkspaceOutputs='on'
    simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
    
    % Extract signal
    try
        x = simOut.get('x_sim');
    catch
        x = evalin('base', 'x_sim');
    end
    
    if isempty(x)
        fprintf('FAILED (empty output)\n');
        continue;
    end
    
    % Ensure column vector
    x = x(:);
    
    % Build metadata (matches format expected by LiveScripts)
    fault = fault_name;
    metadata = struct();
    metadata.fault = fault_name;
    metadata.fault_description = fault_desc;
    metadata.fault_code = fault_code;
    metadata.severity = 'nominal';
    if fault_code > 1
        metadata.severity = 'moderate';
    end
    metadata.severity_factor = severity;
    metadata.speed_rpm = speed_rpm;
    metadata.load_percent = load_pct;
    metadata.temperature_C = temp_C;
    metadata.sommerfeld_number = compute_sommerfeld(temp_C, speed_rpm/60, load_pct/100);
    metadata.fs = fs;
    metadata.duration_s = T;
    metadata.num_samples = length(x);
    metadata.signal_rms = rms(x);
    metadata.signal_peak = max(abs(x));
    metadata.transient_type = 'none';
    metadata.has_evolution = false;
    metadata.generation_timestamp = char(datetime('now'));
    metadata.generator_version = 'Simulink_v3.0';
    metadata.is_augmented = false;
    metadata.is_overlapping_fault = contains(fault_name, 'mixed_');
    
    % Save
    filename = fullfile(outputDir, sprintf('%s_001.mat', fault_name));
    save(filename, 'x', 'fs', 'fault', 'metadata', '-v7.3');
    
    fprintf('✓ (RMS=%.4f, N=%d)\n', rms(x), length(x));
end

gen_time = toc(gen_start);
close_system(modelName, 0);

%% Summary
fprintf('\n========================================\n');
fprintf('  GENERATION COMPLETE\n');
fprintf('========================================\n');
fprintf('  Signals:    %d\n', num_faults);
fprintf('  Time:       %.1f seconds\n', gen_time);
fprintf('  Output:     %s\n', outputDir);
fprintf('  Parameters: %d RPM, %d%% load, %d°C\n', speed_rpm, load_pct, temp_C);
fprintf('========================================\n\n');

fprintf('Next steps:\n');
fprintf('  1. Update LiveScript SIGNAL_FILE paths to point to data_signaux_simulink/\n');
fprintf('  2. Run Signal_Analysis_01_Healthy on sain_001.mat\n');
fprintf('  3. Run Signal_Analysis_02_Single_Faults on fault signals\n\n');

end

%% Helper: Compute Sommerfeld number (matches Simulink Sommerfeld_Calc)
function S = compute_sommerfeld(T_C, Omega_Hz, lf)
    S_base = 0.15;
    visc = exp(-0.03 * (T_C - 60));
    spd = Omega_Hz / 60;
    lf_inv = 1.0 / lf;
    S = S_base * visc * spd * lf_inv;
    S = max(0.05, min(0.5, S));
end
