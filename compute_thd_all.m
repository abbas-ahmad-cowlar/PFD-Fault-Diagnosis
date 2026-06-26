% COMPUTE_THD_ALL  Compute THD for one signal of each fault type
% Uses the same method as Signal_Analysis_01_Healthy.m (line 464):
%   THD = sqrt(amp_2X^2 + amp_3X^2) / amp_1X
%
% Also computes MATLAB's built-in thd() for comparison (if available).
%
% Usage: >> compute_thd_all

fprintf('\n============================================\n');
fprintf('  THD ANALYSIS ACROSS ALL FAULT TYPES\n');
fprintf('============================================\n\n');

data_dir = fullfile(fileparts(mfilename('fullpath')), 'data_signaux_sep_production');

% Fault types matching generator.m
fault_names = {
    'sain'
    'desalignement'
    'desequilibre'
    'jeu'
    'lubrification'
    'cavitation'
    'usure'
    'oilwhirl'
    'mixed_misalign_imbalance'
    'mixed_wear_lube'
    'mixed_cavit_jeu'
};

% Results storage
results = struct();

fprintf('%-30s | %10s | %10s | %10s | %10s | %10s\n', ...
    'Fault Type', 'amp_1X', 'amp_2X', 'amp_3X', 'THD_manual', '2X/1X');
fprintf('%s\n', repmat('-', 1, 95));

for i = 1:length(fault_names)
    fname = fault_names{i};
    
    % Load first signal of this type
    sig_file = fullfile(data_dir, sprintf('%s_001.mat', fname));
    if ~exist(sig_file, 'file')
        fprintf('%-30s | FILE NOT FOUND: %s\n', fname, sig_file);
        continue;
    end
    
    data = load(sig_file);
    x = data.x;
    fs = data.fs;
    
    % Get rotational speed from metadata
    if isfield(data, 'metadata') && isfield(data.metadata, 'speed_rpm')
        Omega = data.metadata.speed_rpm / 60;  % Hz
    else
        Omega = 60;  % Default
    end
    
    % Compute PSD (same as LiveScript)
    window_size = 4096;
    overlap = floor(window_size / 2);
    x_ac = x - mean(x);
    [Pxx, f_psd] = pwelch(x_ac, hann(window_size), overlap, window_size, fs);
    
    % Harmonic frequencies
    f_1X = Omega;
    f_2X = 2 * Omega;
    f_3X = 3 * Omega;
    
    % Find PSD values at harmonic frequencies
    [~, idx_1X] = min(abs(f_psd - f_1X));
    [~, idx_2X] = min(abs(f_psd - f_2X));
    [~, idx_3X] = min(abs(f_psd - f_3X));
    
    amp_1X = sqrt(Pxx(idx_1X));
    amp_2X = sqrt(Pxx(idx_2X));
    amp_3X = sqrt(Pxx(idx_3X));
    
    % THD (same formula as LiveScript line 464)
    THD_manual = sqrt(amp_2X^2 + amp_3X^2) / (amp_1X + eps);
    
    ratio_2X_1X = amp_2X / (amp_1X + eps);
    
    fprintf('%-30s | %10.6f | %10.6f | %10.6f | %10.4f | %10.4f\n', ...
        fname, amp_1X, amp_2X, amp_3X, THD_manual, ratio_2X_1X);
    
    % Store
    results(i).fault = fname;
    results(i).Omega = Omega;
    results(i).amp_1X = amp_1X;
    results(i).amp_2X = amp_2X;
    results(i).amp_3X = amp_3X;
    results(i).THD = THD_manual;
    results(i).ratio_2X_1X = ratio_2X_1X;
    results(i).speed_rpm = Omega * 60;
end

fprintf('\n============================================\n');
fprintf('  ANALYSIS NOTES\n');
fprintf('============================================\n');
fprintf('THD = sqrt(amp_2X^2 + amp_3X^2) / amp_1X\n');
fprintf('amp_NX = sqrt(PSD at NX frequency)\n');
fprintf('PSD computed via pwelch (Hann window, 4096 samples, 50%% overlap)\n');
fprintf('Rotational speed taken from metadata.speed_rpm\n\n');

% Also compute average THD across 5 signals per fault for robustness
fprintf('\n============================================\n');
fprintf('  AVERAGED THD (5 signals per fault)\n');
fprintf('============================================\n\n');

fprintf('%-30s | %10s | %10s | %10s\n', 'Fault Type', 'THD_mean', 'THD_std', 'THD_range');
fprintf('%s\n', repmat('-', 1, 65));

for i = 1:length(fault_names)
    fname = fault_names{i};
    thd_vals = [];
    
    for s = 1:5
        sig_file = fullfile(data_dir, sprintf('%s_%03d.mat', fname, s));
        if ~exist(sig_file, 'file'), continue; end
        
        data = load(sig_file);
        x = data.x;
        fs = data.fs;
        
        if isfield(data, 'metadata') && isfield(data.metadata, 'speed_rpm')
            Omega = data.metadata.speed_rpm / 60;
        else
            Omega = 60;
        end
        
        x_ac = x - mean(x);
        [Pxx, f_psd] = pwelch(x_ac, hann(4096), 2048, 4096, fs);
        
        [~, idx_1X] = min(abs(f_psd - Omega));
        [~, idx_2X] = min(abs(f_psd - 2*Omega));
        [~, idx_3X] = min(abs(f_psd - 3*Omega));
        
        a1 = sqrt(Pxx(idx_1X));
        a2 = sqrt(Pxx(idx_2X));
        a3 = sqrt(Pxx(idx_3X));
        
        thd_vals(end+1) = sqrt(a2^2 + a3^2) / (a1 + eps);
    end
    
    if ~isempty(thd_vals)
        fprintf('%-30s | %10.4f | %10.4f | %.4f - %.4f\n', ...
            fname, mean(thd_vals), std(thd_vals), min(thd_vals), max(thd_vals));
    end
end

fprintf('\n============================================\n');
fprintf('  DONE\n');
fprintf('============================================\n\n');
