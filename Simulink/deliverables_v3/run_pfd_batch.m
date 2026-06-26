function run_pfd_batch()
% RUN_PFD_BATCH  Batch-generate signals using the Simulink model
%
% Replicates the output of generator.m (1,430 signals) by running
% the Simulink model in a loop with varying parameters.
%
% Output: .mat files in data_signaux_sep_simulink/ (or user-specified)
%
% Usage:
%   >> run_pfd_batch()

fprintf('\n========================================\n');
fprintf('  PFD Batch Signal Generator (Simulink)\n');
fprintf('========================================\n\n');

%% Configuration (matches generator.m CONFIG)
CONFIG.fs = 20480;
CONFIG.T = 5;
CONFIG.Omega_base = 60;
CONFIG.num_signals_per_fault = 100;
CONFIG.rng_seed = 42;

% Output directory (separate from generator.m output)
CONFIG.output_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'data_signaux_sep_simulink');

% Severity ranges (matches generator.m)
CONFIG.severity_ranges = struct(...
    'incipient', [0.20, 0.45], ...
    'mild',      [0.45, 0.70], ...
    'moderate',  [0.70, 0.90], ...
    'severe',    [0.90, 1.00]);
CONFIG.severity_levels = {'incipient', 'mild', 'moderate', 'severe'};
CONFIG.temporal_evolution_prob = 0.30;  % 30% of signals

% Operating condition ranges (matches generator.m)
CONFIG.speed_variation = 0.10;   % +/-10%
CONFIG.load_range = [30, 100];   % percent
CONFIG.temp_range = [40, 80];    % Celsius

% Transient behavior (matches generator.m)
CONFIG.transient_prob = 0.25;    % 25% of signals
CONFIG.transient_types = [1, 2, 3]; % speed_ramp, load_step, thermal

% Noise: aliasing at 10% probability (matches generator.m)
CONFIG.aliasing_prob = 0.10;

% Augmentation (handled here, not in Simulink model)
CONFIG.augmentation_enabled = true;
CONFIG.augmentation_ratio = 0.30;  % 30% additional
CONFIG.aug_time_shift_max = 0.02;  % 2% of signal length
CONFIG.aug_amp_scale_range = [0.85, 1.15];
CONFIG.aug_noise_range = [0.02, 0.05];

% Fault types (matches generator.m order)
fault_types = {
    'sain',                      1   % Healthy
    'desalignement',             2   % Misalignment
    'desequilibre',              3   % Imbalance
    'jeu',                       4   % Clearance
    'lubrification',             5   % Lubrication
    'cavitation',                6   % Cavitation
    'usure',                     7   % Wear
    'oilwhirl',                  8   % Oil Whirl
    'mixed_misalign_imbalance',  9   % Mixed M+I
    'mixed_wear_lube',           10  % Mixed W+L
    'mixed_cavit_jeu',           11  % Mixed C+J
};

%% Setup
if ~exist(CONFIG.output_dir, 'dir')
    mkdir(CONFIG.output_dir);
end

% Load model
scriptDir = fileparts(mfilename('fullpath'));
modelFile = fullfile(scriptDir, 'latest', 'PFD_Signal_Generator.slx');
if ~exist(modelFile, 'file')
    error('Model not found at %s. Run build_simulink_model() first.', modelFile);
end

modelName = 'PFD_Signal_Generator';
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
load_system(modelFile);

% Set random seed
if ~isempty(CONFIG.rng_seed)
    rng(CONFIG.rng_seed);
end

%% Compute totals
num_base = CONFIG.num_signals_per_fault;
if CONFIG.augmentation_enabled
    num_aug = round(num_base * CONFIG.augmentation_ratio);
else
    num_aug = 0;
end
num_per_fault = num_base + num_aug;
num_faults = size(fault_types, 1);
total_signals = num_faults * num_per_fault;

fprintf('Configuration:\n');
fprintf('  Fault types:       %d\n', num_faults);
fprintf('  Signals per type:  %d (%d base + %d augmented)\n', ...
    num_per_fault, num_base, num_aug);
fprintf('  Total signals:     %d\n', total_signals);
fprintf('  Output:            %s\n\n', CONFIG.output_dir);

%% Main generation loop
total_count = 0;
gen_start = tic;

for k = 1:num_faults
    fault_name = fault_types{k, 1};
    fault_code = fault_types{k, 2};

    fprintf('Generating [%d/%d]: %s\n', k, num_faults, fault_name);

    for n = 1:num_per_fault
        % Per-signal seed for reproducibility
        if ~isempty(CONFIG.rng_seed)
            rng(CONFIG.rng_seed + total_count);
        end

        is_augmented = (n > num_base);

        % --- Severity ---
        if ~strcmp(fault_name, 'sain')
            sev_level = CONFIG.severity_levels{randi(4)};
            sev_range = CONFIG.severity_ranges.(sev_level);
            severity_factor = sev_range(1) + diff(sev_range) * rand;
        else
            sev_level = 'nominal';
            severity_factor = 1.0;
        end

        % Temporal evolution (30% probability)
        enable_evo = 0;
        if rand < CONFIG.temporal_evolution_prob
            enable_evo = 1;
        end

        % --- Operating conditions ---
        speed_var = 1.0 + (rand - 0.5) * 2 * CONFIG.speed_variation;
        speed_rpm = CONFIG.Omega_base * speed_var * 60;
        load_pct = CONFIG.load_range(1) + diff(CONFIG.load_range) * rand;
        temp_C = CONFIG.temp_range(1) + diff(CONFIG.temp_range) * rand;

        % --- Transient ---
        if rand < CONFIG.transient_prob
            trans_type = CONFIG.transient_types(randi(3));
        else
            trans_type = 1;  % 1 = None (1-based Multiport Switch)
        end

        % --- Set model parameters ---
        set_param([modelName '/Operating_Conditions/Speed_RPM'], ...
            'Value', num2str(speed_rpm));
        set_param([modelName '/Operating_Conditions/Load_Percent'], ...
            'Value', num2str(load_pct));
        set_param([modelName '/Operating_Conditions/Temperature_C'], ...
            'Value', num2str(temp_C));
        set_param([modelName '/Fault_Injection/Fault_Type'], ...
            'Value', num2str(fault_code));
        set_param([modelName '/Severity_Control/Severity_Level'], ...
            'Value', num2str(severity_factor));
        set_param([modelName '/Severity_Control/Enable_Evolution'], ...
            'Value', num2str(enable_evo));
        set_param([modelName '/Transient_Behavior/Transient_Type'], ...
            'Value', num2str(trans_type));

        % --- Aliasing control ---
        % Enable aliasing at 10% probability
        if rand < CONFIG.aliasing_prob
            set_param([modelName '/Noise_Model/Aliasing'], ...
                'Amplitude', '0.005');
        else
            set_param([modelName '/Noise_Model/Aliasing'], ...
                'Amplitude', '0');
        end

        % --- Run simulation ---
        simOut = sim(modelName, 'ReturnWorkspaceOutputs', 'on');
        x = simOut.get('x_sim');
        fs = CONFIG.fs;
        fault = fault_name;

        % --- Data augmentation (post-processing) ---
        aug_params = struct('method', 'none');
        if is_augmented && CONFIG.augmentation_enabled
            aug_methods = {'time_shift', 'amplitude_scale', 'noise_injection'};
            aug_method = aug_methods{randi(3)};

            switch aug_method
                case 'time_shift'
                    shift_max = round(CONFIG.aug_time_shift_max * length(x));
                    shift = randi([-shift_max, shift_max]);
                    x = circshift(x, shift);
                    aug_params = struct('method','time_shift','shift',shift);

                case 'amplitude_scale'
                    sc = CONFIG.aug_amp_scale_range(1) + ...
                         diff(CONFIG.aug_amp_scale_range) * rand;
                    x = x * sc;
                    aug_params = struct('method','amplitude_scale','scale',sc);

                case 'noise_injection'
                    nl = CONFIG.aug_noise_range(1) + ...
                         diff(CONFIG.aug_noise_range) * rand;
                    x = x + nl * randn(size(x));
                    aug_params = struct('method','noise_injection','level',nl);
            end
        end

        % --- Metadata ---
        metadata = struct();
        metadata.fault = fault;
        metadata.severity = sev_level;
        metadata.severity_factor_initial = severity_factor;
        metadata.has_evolution = logical(enable_evo);
        metadata.speed_rpm = speed_rpm;
        metadata.speed_variation_factor = speed_var;
        metadata.load_percent = load_pct;
        metadata.temperature_C = temp_C;
        metadata.transient_type = trans_type;
        metadata.fs = fs;
        metadata.duration_s = CONFIG.T;
        metadata.num_samples = length(x);
        metadata.signal_rms = rms(x);
        metadata.signal_peak = max(abs(x));
        metadata.is_augmented = is_augmented;
        metadata.augmentation = aug_params;
        metadata.generation_timestamp = char(datetime('now'));
        metadata.generator_version = 'Simulink_v1.0';
        metadata.rng_seed = CONFIG.rng_seed;
        metadata.is_overlapping_fault = contains(fault, 'mixed_');

        % --- Save ---
        if is_augmented
            filename = fullfile(CONFIG.output_dir, ...
                sprintf('%s_%03d_aug.mat', fault, n));
        else
            filename = fullfile(CONFIG.output_dir, ...
                sprintf('%s_%03d.mat', fault, n));
        end
        save(filename, 'x', 'fs', 'fault', 'metadata', '-v7.3');

        total_count = total_count + 1;
    end

    fprintf('  Done: %d signals (%d base + %d aug)\n', ...
        num_per_fault, num_base, num_aug);
end

gen_time = toc(gen_start);
close_system(modelName, 0);

fprintf('\n========================================\n');
fprintf('  BATCH GENERATION COMPLETE\n');
fprintf('========================================\n');
fprintf('  Total:  %d signals\n', total_count);
fprintf('  Time:   %.1f seconds (%.1f sig/s)\n', ...
    gen_time, total_count/gen_time);
fprintf('  Output: %s\n', CONFIG.output_dir);
fprintf('========================================\n\n');
end
