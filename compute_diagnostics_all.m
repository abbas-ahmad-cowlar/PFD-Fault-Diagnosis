% COMPUTE_DIAGNOSTICS_ALL  Compute key diagnostic features for all fault types
% Computes: THD, Kurtosis, Crest Factor, RMS, Spectral Band Energies
% Averages over 5 signals per fault type for robustness.
%
% Usage: >> compute_diagnostics_all

fprintf('\n================================================================\n');
fprintf('  COMPREHENSIVE DIAGNOSTIC FEATURES — ALL FAULT TYPES\n');
fprintf('================================================================\n\n');

data_dir = fullfile(fileparts(mfilename('fullpath')), 'data_signaux_sep_production');

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

N_avg = 5;  % average over 5 signals per fault

% Storage
all_results = struct();

for i = 1:length(fault_names)
    fname = fault_names{i};
    
    kurt_vals = []; crest_vals = []; rms_vals = []; thd_vals = [];
    E_sub = []; E_1X = []; E_2X3X = []; E_mid = []; E_high = [];
    amp1x_vals = []; amp2x_vals = []; amp3x_vals = [];
    
    for s = 1:N_avg
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
        
        % === TIME-DOMAIN FEATURES ===
        kurt_vals(end+1) = kurtosis(x);
        crest_vals(end+1) = max(abs(x)) / rms(x);
        rms_vals(end+1) = rms(x);
        
        % === FREQUENCY-DOMAIN (PSD) ===
        x_ac = x - mean(x);
        [Pxx, f_psd] = pwelch(x_ac, hann(4096), 2048, 4096, fs);
        df = f_psd(2) - f_psd(1);  % frequency resolution
        
        % Harmonic amplitudes
        [~, idx_1X] = min(abs(f_psd - Omega));
        [~, idx_2X] = min(abs(f_psd - 2*Omega));
        [~, idx_3X] = min(abs(f_psd - 3*Omega));
        
        a1 = sqrt(Pxx(idx_1X));
        a2 = sqrt(Pxx(idx_2X));
        a3 = sqrt(Pxx(idx_3X));
        
        amp1x_vals(end+1) = a1;
        amp2x_vals(end+1) = a2;
        amp3x_vals(end+1) = a3;
        thd_vals(end+1) = sqrt(a2^2 + a3^2) / (a1 + eps);
        
        % === SPECTRAL BAND ENERGY ===
        % Band 1: Sub-synchronous (0.3X to 0.5X) — oil whirl region
        band_sub = f_psd >= 0.3*Omega & f_psd <= 0.5*Omega;
        E_sub(end+1) = sum(Pxx(band_sub)) * df;
        
        % Band 2: 1X region (0.9X to 1.1X) — imbalance
        band_1X = f_psd >= 0.9*Omega & f_psd <= 1.1*Omega;
        E_1X(end+1) = sum(Pxx(band_1X)) * df;
        
        % Band 3: 2X-3X region (1.8X to 3.2X) — misalignment
        band_2X3X = f_psd >= 1.8*Omega & f_psd <= 3.2*Omega;
        E_2X3X(end+1) = sum(Pxx(band_2X3X)) * df;
        
        % Band 4: Mid-frequency (500-2000 Hz) — general wear/friction
        band_mid = f_psd >= 500 & f_psd <= 2000;
        E_mid(end+1) = sum(Pxx(band_mid)) * df;
        
        % Band 5: High-frequency (2000-5000 Hz) — cavitation bursts
        band_high = f_psd >= 2000 & f_psd <= 5000;
        E_high(end+1) = sum(Pxx(band_high)) * df;
    end
    
    all_results(i).fault = fname;
    all_results(i).kurtosis = mean(kurt_vals);
    all_results(i).crest = mean(crest_vals);
    all_results(i).rms = mean(rms_vals);
    all_results(i).thd = mean(thd_vals);
    all_results(i).amp_1X = mean(amp1x_vals);
    all_results(i).amp_2X = mean(amp2x_vals);
    all_results(i).amp_3X = mean(amp3x_vals);
    all_results(i).E_sub = mean(E_sub);
    all_results(i).E_1X = mean(E_1X);
    all_results(i).E_2X3X = mean(E_2X3X);
    all_results(i).E_mid = mean(E_mid);
    all_results(i).E_high = mean(E_high);
end

%% ==================== DISPLAY RESULTS ====================

% --- TABLE 1: Time-Domain Features ---
fprintf('TABLE 1: TIME-DOMAIN FEATURES (averaged over %d signals)\n', N_avg);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('%-28s | %8s | %8s | %8s | %8s\n', ...
    'Fault Type', 'RMS', 'Kurtosis', 'Crest F.', 'THD');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:length(all_results)
    r = all_results(i);
    % Flag anomalies
    kurt_flag = '';
    if r.kurtosis > 4, kurt_flag = ' ⚠'; end
    if r.kurtosis > 6, kurt_flag = ' ❌'; end
    
    fprintf('%-28s | %8.5f | %7.2f%s | %8.2f | %8.4f\n', ...
        r.fault, r.rms, r.kurtosis, kurt_flag, r.crest, r.thd);
end

% --- TABLE 2: Harmonic Amplitudes ---
fprintf('\n\nTABLE 2: HARMONIC AMPLITUDES (averaged over %d signals)\n', N_avg);
fprintf('%s\n', repmat('=', 1, 80));
fprintf('%-28s | %10s | %10s | %10s | %8s\n', ...
    'Fault Type', 'amp_1X', 'amp_2X', 'amp_3X', '2X/1X');
fprintf('%s\n', repmat('-', 1, 80));
for i = 1:length(all_results)
    r = all_results(i);
    ratio = r.amp_2X / (r.amp_1X + eps);
    fprintf('%-28s | %10.6f | %10.6f | %10.6f | %8.4f\n', ...
        r.fault, r.amp_1X, r.amp_2X, r.amp_3X, ratio);
end

% --- TABLE 3: Spectral Band Energies ---
fprintf('\n\nTABLE 3: SPECTRAL BAND ENERGY (averaged over %d signals)\n', N_avg);
fprintf('%s\n', repmat('=', 1, 95));
fprintf('%-28s | %10s | %10s | %10s | %10s | %10s\n', ...
    'Fault Type', 'Sub-sync', '1X band', '2X-3X band', 'Mid(500-2k)', 'High(2k-5k)');
fprintf('%-28s | %10s | %10s | %10s | %10s | %10s\n', ...
    '', '0.3-0.5X', '0.9-1.1X', '1.8-3.2X', 'Hz', 'Hz');
fprintf('%s\n', repmat('-', 1, 95));
for i = 1:length(all_results)
    r = all_results(i);
    fprintf('%-28s | %10.2e | %10.2e | %10.2e | %10.2e | %10.2e\n', ...
        r.fault, r.E_sub, r.E_1X, r.E_2X3X, r.E_mid, r.E_high);
end

% --- INTERPRETATION ---
fprintf('\n\n================================================================\n');
fprintf('  PHYSICS VALIDATION CHECK\n');
fprintf('================================================================\n\n');

% Find reference values (healthy)
sain = all_results(1);

fprintf('Expected vs. Measured behavior:\n\n');

checks = {
    'desalignement',  '2X-3X band >> healthy',     2, 'E_2X3X';
    'desequilibre',   '1X band >> healthy',         2, 'E_1X';
    'jeu',            'Sub-sync energy elevated',   2, 'E_sub';
    'lubrification',  'Kurtosis elevated (impacts)',2, 'kurtosis';
    'cavitation',     'High-freq energy elevated',  2, 'E_high';
    'usure',          'Mid-freq energy elevated',   2, 'E_mid';
    'oilwhirl',       'Sub-sync energy >> healthy', 2, 'E_sub';
};

for c = 1:size(checks, 1)
    fault_name = checks{c, 1};
    expected = checks{c, 2};
    min_ratio = checks{c, 3};
    field = checks{c, 4};
    
    % Find this fault
    idx = find(strcmp({all_results.fault}, fault_name));
    if isempty(idx), continue; end
    
    val = all_results(idx).(field);
    ref = sain.(field);
    ratio = val / (ref + eps);
    
    if ratio >= min_ratio
        status = '✅ PASS';
    else
        status = '❌ FAIL';
    end
    
    fprintf('  %s %-25s: %-35s (ratio vs healthy: %.2fx)\n', ...
        status, fault_name, expected, ratio);
end

fprintf('\n================================================================\n');
fprintf('  DONE\n');
fprintf('================================================================\n\n');
