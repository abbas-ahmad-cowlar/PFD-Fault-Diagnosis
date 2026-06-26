fprintf('\nStep 3.2: Generating Fig0 (Raw Signal Examples)...\n');

% --- Define Fault Groups ---
% We split the classes to make the plots less cluttered
basicFaults = {'cavitation', 'desalignement', 'desequilibre', 'jeu', ...
                'lubrication', 'oilwhirl', 'sain', 'usure'};
            
mixedFaults = {'mixed_cavit_jeu', 'mixed_misalign_imbalance', 'mixed_wear_lube'};

% --- Figure 1: Basic Faults ---
fprintf('  Generating Fig0a (Basic Faults)...\n');
fig1 = figure('Name', 'PFD Basic Fault Signal Examples', 'Position', [100, 100, 1600, 900]);
t_sig1 = tiledlayout(length(basicFaults), 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop through BASIC faults
for k = 1:length(basicFaults)
    fault = basicFaults{k};
    fileName = fullfile(CONFIG.inputDir, sprintf('%s_001.mat', fault));
    
    if ~isfile(fileName)
        fprintf('  Warning: Could not find %s. Skipping.\n', fileName);
        nexttile; nexttile; nexttile;
        continue;
    end
    
    sigData = load(fileName);
    x = sigData.x;
    fs = sigData.fs;
    t_vec = (0:length(x)-1)/fs;
    
    % --- Plotting ---
    % Time Domain
    ax1 = nexttile;
    plot(ax1, t_vec, x);
    title(ax1, ['Time: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax1, [0, 0.5]);
    grid(ax1, 'on');
    
    % Frequency Domain (Welch PSD)
    ax2 = nexttile;
    [Pxx, F] = pwelch(x, hann(1024), 512, 2048, fs);
    semilogy(ax2, F, Pxx);
    title(ax2, ['Frequency: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax2, [0, fs/8]); % Keeping your original fs/8 limit
    grid(ax2, 'on');
    
    % Time-Frequency Spectrogram
    ax3 = nexttile;
    spectrogram(x, hann(256), 128, 256, fs, 'yaxis');
    title(ax3, ['Spectrogram: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    
    % --- CLUTTER REDUCTION: Only add labels to outer plots ---
    if k == 1
        % Add Y-labels only to the first row
        ylabel(ax1, 'Amplitude');
        ylabel(ax2, 'PSD');
    end
    if k == length(basicFaults)
        % Add X-labels only to the last row
        xlabel(ax1, 'Time (s)');
        xlabel(ax2, 'Frequency (Hz)');
    end
end

title(t_sig1, 'Fig 0a: Time, Frequency, and Spectrogram Analysis (Basic Faults)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
saveas(fig1, fullfile(CONFIG.outputDir, 'Fig0a_Basic_Faults.png'));
fprintf('✓ Saved: Fig0a_Basic_Faults.png\n');


% --- Figure 2: Mixed Faults ---
fprintf('  Generating Fig0b (Mixed Faults)...\n');
fig2 = figure('Name', 'PFD Mixed Fault Signal Examples', 'Position', [100, 100, 1600, 500]);
t_sig2 = tiledlayout(length(mixedFaults), 3, 'Padding', 'compact', 'TileSpacing', 'compact');

% Loop through MIXED faults
for k = 1:length(mixedFaults)
    fault = mixedFaults{k};
    fileName = fullfile(CONFIG.inputDir, sprintf('%s_001.mat', fault));
    
    if ~isfile(fileName)
        fprintf('  Warning: Could not find %s. Skipping.\n', fileName);
        nexttile; nexttile; nexttile;
        continue;
    end
    
    sigData = load(fileName);
    x = sigData.x;
    fs = sigData.fs;
    t_vec = (0:length(x)-1)/fs;
    
    % --- Plotting ---
    % Time Domain
    ax1 = nexttile;
    plot(ax1, t_vec, x);
    title(ax1, ['Time: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax1, [0, 0.5]);
    grid(ax1, 'on');
    
    % Frequency Domain (Welch PSD)
    ax2 = nexttile;
    [Pxx, F] = pwelch(x, hann(1024), 512, 2048, fs);
    semilogy(ax2, F, Pxx);
    title(ax2, ['Frequency: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');
    xlim(ax2, [0, fs/8]);
    grid(ax2, 'on');
    
    % Time-Frequency Spectrogram
    ax3 = nexttile;
    spectrogram(x, hann(256), 128, 256, fs, 'yaxis');
    title(ax3, ['Spectrogram: ' fault], 'Interpreter', 'none', 'FontWeight', 'normal');

    % --- CLUTTER REDUCTION: Only add labels to outer plots ---
    if k == 1
        % Add Y-labels only to the first row
        ylabel(ax1, 'Amplitude');
        ylabel(ax2, 'PSD');
    end
    if k == length(mixedFaults)
        % Add X-labels only to the last row
        xlabel(ax1, 'Time (s)');
        xlabel(ax2, 'Frequency (Hz)');
    end
end

title(t_sig2, 'Fig 0b: Time, Frequency, and Spectrogram Analysis (Mixed Faults)', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
saveas(fig2, fullfile(CONFIG.outputDir, 'Fig0b_Mixed_Faults.png'));
fprintf('✓ Saved: Fig0b_Mixed_Faults.png\n');