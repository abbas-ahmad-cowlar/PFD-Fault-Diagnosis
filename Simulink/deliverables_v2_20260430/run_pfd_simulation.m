function run_pfd_simulation()
% RUN_PFD_SIMULATION  Interactive runner for PFD Simulink model
%
% Opens the latest PFD_Signal_Generator model, lets you configure
% fault type, severity, and transient behavior, then runs and plots.
%
% Usage:
%   >> run_pfd_simulation()

fprintf('\n========================================\n');
fprintf('  PFD Signal Generator - Interactive\n');
fprintf('========================================\n\n');

% Find latest model
scriptDir = fileparts(mfilename('fullpath'));
modelFile = fullfile(scriptDir, 'latest', 'PFD_Signal_Generator.slx');

if ~exist(modelFile, 'file')
    error('Model not found. Run build_simulink_model() first.');
end

% Load model
modelName = 'PFD_Signal_Generator';
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
load_system(modelFile);
fprintf('Loaded: %s\n\n', modelFile);

% Fault type menu
fault_names = {
    '1.  Healthy (sain)'
    '2.  Misalignment (desalignement)'
    '3.  Imbalance (desequilibre)'
    '4.  Clearance (jeu)'
    '5.  Lubrication (lubrification)'
    '6.  Cavitation'
    '7.  Wear (usure)'
    '8.  Oil Whirl (oilwhirl)'
    '9.  Mixed: Misalign + Imbalance'
    '10. Mixed: Wear + Lube'
    '11. Mixed: Cavit + Clearance'
};

fprintf('Select fault type:\n');
for i = 1:length(fault_names)
    fprintf('  %s\n', fault_names{i});
end
fault_type = input('\nFault type [1-11]: ');
if isempty(fault_type), fault_type = 1; end
fault_type = max(1, min(11, round(fault_type)));

% Severity
severity = input('Severity [0.0-1.0, default 0.7]: ');
if isempty(severity), severity = 0.7; end

% Evolution
evo = input('Enable temporal evolution? [0/1, default 0]: ');
if isempty(evo), evo = 0; end

% Transient
fprintf('\nTransient type:\n');
fprintf('  0. None\n  1. Speed ramp\n  2. Load step\n  3. Thermal expansion\n');
trans = input('Transient [0-3, default 0]: ');
if isempty(trans), trans = 0; end

% Set parameters in model
set_param([modelName '/Fault_Injection/Fault_Type'], 'Value', num2str(fault_type));
set_param([modelName '/Severity_Control/Severity_Level'], 'Value', num2str(severity));
set_param([modelName '/Severity_Control/Enable_Evolution'], 'Value', num2str(evo));
set_param([modelName '/Transient_Behavior/Transient_Type'], 'Value', num2str(trans));

fprintf('\n--- Running simulation ---\n');
fprintf('  Fault: %s\n', fault_names{fault_type});
fprintf('  Severity: %.2f (evolution: %s)\n', severity, iif(evo, 'ON', 'OFF'));
fprintf('  Transient: %d\n\n', trans);

% Run
simOut = sim(modelName);

% Extract signal
x = simOut.get('x_sim');
t = simOut.get('tout');
fs = 20480;

if isempty(x)
    fprintf('Warning: No output signal. Check model configuration.\n');
    return;
end

% Plot results
figure('Name', 'PFD Simulation Result', ...
    'Position', [100, 100, 1200, 600], 'Color', 'white');

% Time domain
subplot(2,1,1);
plot(t, x, 'b-', 'LineWidth', 0.5);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('PFD Signal: %s | Severity=%.2f', ...
    fault_names{fault_type}, severity));
grid on;
xlim([0, max(t)]);

% Frequency domain
subplot(2,1,2);
[Pxx, f] = pwelch(x, hann(4096), 2048, 4096, fs);
semilogy(f, Pxx, 'b-', 'LineWidth', 0.8);
xlabel('Frequency (Hz)');
ylabel('PSD');
title('Power Spectral Density');
grid on;
xlim([0, fs/4]);

fprintf('Done. Signal in workspace as x_sim.\n');

% Optionally save
save_opt = input('\nSave to .mat? [y/n, default n]: ', 's');
if strcmpi(save_opt, 'y')
    fault_codes = {'sain','desalignement','desequilibre','jeu', ...
        'lubrification','cavitation','usure','oilwhirl', ...
        'mixed_misalign_imbalance','mixed_wear_lube','mixed_cavit_jeu'};
    fault = fault_codes{fault_type};
    filename = sprintf('sim_%s_sev%.0f.mat', fault, severity*100);
    save(filename, 'x', 'fs', 'fault');
    fprintf('Saved: %s\n', filename);
end

end

function result = iif(cond, t, f)
    if cond, result = t; else, result = f; end
end
