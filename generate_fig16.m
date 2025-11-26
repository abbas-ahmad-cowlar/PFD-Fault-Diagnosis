%% REGENERATE ADVANCED VISUALIZATIONS (Fig16) - ENHANCED SVM
% This script generates Fig 16 with:
% 1. Filled Decision Regions (Contours)
% 2. Discrete Color Scheme (Red/Blue/Green)
% 3. Support Vector Highlighting
% 4. Jittered Data Points
% 5. Clean Titles

clc; close all;
fprintf('\n========================================================================\n');
fprintf('REGENERATING FIG 16: ENHANCED SVM DECISION BOUNDARIES\n');
fprintf('========================================================================\n\n');

%% STEP 1: Load Results
fprintf('Step 1: Loading saved results...\n');
dirs = dir('PFD_*Results*');
if isempty(dirs), error('No results directory found!'); end
dirs = dirs([dirs.isdir]);
[~, idx] = max([dirs.datenum]);
latestDir = dirs(idx).name;
resultFile = fullfile(latestDir, 'step5a_results.mat');

if ~exist(resultFile, 'file'), error('Results file missing.'); end
load(resultFile);
fprintf('   ✓ Loaded: %s\n\n', resultFile);

CONFIG.outputDir = latestDir;

%% STEP 2: Prepare SVM & Feature Importance
% (Skipping RF/NN as requested)
fprintf('Step 2: Preparing SVM Data...\n');

if ~isfield(modelResults, 'SVM') || ~isfield(modelResults.SVM, 'model')
    error('SVM model not found in loaded data.');
end

svmModel = modelResults.SVM.model;

% Calculate Permutation Importance for SVM (Quick Calc)
fprintf('   - Calculating feature importance for visualization...\n');
basePred = predict(svmModel, X_test_norm);
baseAcc = sum(basePred == Y_test) / length(Y_test);
imp = zeros(length(featureNames), 1);

for f = 1:length(featureNames)
    X_perm = X_test_norm;
    X_perm(:,f) = X_perm(randperm(size(X_perm,1)), f);
    pred_perm = predict(svmModel, X_perm);
    imp(f) = max(0, baseAcc - (sum(pred_perm == Y_test)/length(Y_test)));
end

[~, sortedIdx] = sort(imp, 'descend');
topFeatures = sortedIdx(1:min(4, length(sortedIdx)));

%% STEP 3: Generate The Enhanced Figure
fprintf('Step 3: Generating Plot...\n');

if length(topFeatures) >= 4
    % Define Pairs
    pair1 = topFeatures(1:2);
    pair2 = topFeatures(3:4);

    % Create Figure
    fig16 = figure('Position', [100, 100, 1400, 650], 'Color', 'w');

    % --- Subplot 1 ---
    subplot(1, 2, 1);
    plotEnhancedSVM(svmModel, X_test_norm, Y_test, pair1, featureNames, classNames);

    % --- Subplot 2 ---
    subplot(1, 2, 2);
    plotEnhancedSVM(svmModel, X_test_norm, Y_test, pair2, featureNames, classNames);

    % Fix 5: Main Title Redundancy -> Clean Main Title
    sgtitle('Fig 16: SVM Classification Regions for Top Feature Pairs', ...
        'FontSize', 16, 'FontWeight', 'bold');

    % Save
    outFile = fullfile(CONFIG.outputDir, 'Fig16_SVM_Decision_Boundaries_Enhanced.png');
    saveas(fig16, outFile);
    fprintf('   ✓✓✓ SAVED: %s\n', outFile);
else
    fprintf('   ⚠️ Not enough features to plot.\n');
end

fprintf('\nDONE.\n');

%% -----------------------------------------------------------------------
%  HELPER FUNCTION: THE ENHANCED PLOTTER
%  -----------------------------------------------------------------------
function plotEnhancedSVM(svmModel, X_data, Y_data, featurePair, featureNames, classNames)
    % Extract Indices
    f1 = featurePair(1);
    f2 = featurePair(2);
    
    X_pair = X_data(:, [f1, f2]);
    
    % Ensure Numeric Labels
    if iscategorical(Y_data), Y_data = double(Y_data); end
    classes = unique(Y_data);
    nClasses = length(classes);

    % --- FIX 2: Ambiguous Color Scheme -> Distinct Colors ---
    % 1=Red, 2=Blue, 3=Green, 4=Purple (Customize as needed)
    colors = [0.85, 0.33, 0.31;  % Red
              0.29, 0.57, 0.84;  % Blue
              0.42, 0.76, 0.40;  % Green
              0.60, 0.30, 0.70]; % Purple
    % Resize color map to match number of classes
    if nClasses > 4, colors = lines(nClasses); end
    
    % --- FIX 1: No Decision Boundaries -> Filled Contours ---
    % Create Grid
    res = 200; % Grid resolution
    margin = 0.5;
    x_min = min(X_pair(:,1))-margin; x_max = max(X_pair(:,1))+margin;
    y_min = min(X_pair(:,2))-margin; y_max = max(X_pair(:,2))+margin;
    [xx, yy] = meshgrid(linspace(x_min, x_max, res), linspace(y_min, y_max, res));
    
    % Prepare Full Predictor Matrix (Fill others with Mean)
    X_grid_full = repmat(mean(X_data,1), numel(xx), 1);
    X_grid_full(:, f1) = xx(:);
    X_grid_full(:, f2) = yy(:);
    
    try
        % Predict
        preds = predict(svmModel, X_grid_full);
        if iscategorical(preds), preds = double(preds); end
        Z = reshape(preds, size(xx));
        
        hold on;
        % Filled Contour (Regions)
        [~, hC] = contourf(xx, yy, Z, 'LineColor', 'none');
        colormap(gca, colors(1:nClasses, :));
        hC.FaceAlpha = 0.2; % Transparent background
        
        % Boundary Lines (Where Z changes)
        contour(xx, yy, Z, 'LineColor', 'k', 'LineWidth', 2, 'LevelList', 1.5:1:nClasses);
    catch
        warning('Could not generate boundary (dimension mismatch likely).');
    end
    hold on;

    % --- FIX 4: Data Distribution -> Jitter & Alpha ---
    % Jitter Amount (2% of range)
    jitterAmt = (x_max - x_min) * 0.02;
    
    legendHandles = [];
    legendLabels = {};

    for i = 1:nClasses
        c = classes(i);
        idx = Y_data == c;
        if sum(idx) > 0
            % Add Jitter to X
            x_plot = X_pair(idx, 1) + (rand(sum(idx),1)-0.5)*jitterAmt;
            y_plot = X_pair(idx, 2); 
            
            % Scatter with Transparency (Alpha)
            h = scatter(x_plot, y_plot, 50, colors(i,:), 'filled', ...
                'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.7, 'LineWidth', 0.5);
            
            legendHandles(end+1) = h;
            if i <= length(classNames)
                legendLabels{end+1} = char(classNames{i});
            else
                legendLabels{end+1} = sprintf('Class %d', c);
            end
        end
    end

    % --- FIX 3: Missing Support Vectors -> Highlight Them ---
    try
        if isprop(svmModel, 'IsSupportVector')
            sv_idx = svmModel.IsSupportVector;
            if any(sv_idx)
                sv_X = X_pair(sv_idx, :);
                % Draw hollow black circles around SVs
                scatter(sv_X(:,1), sv_X(:,2), 120, 'k', 'LineWidth', 1.5);
                
                % Dummy Legend for SV
                hSV = plot(NaN,NaN, 'ko', 'MarkerSize', 10, 'LineWidth', 1.5);
                legendHandles(end+1) = hSV;
                legendLabels{end+1} = 'Support Vectors';
            end
        end
    catch
        % Ignore if SVs aren't accessible
    end

    % --- FIX 5: Title Redundancy -> Clean Subtitles ---
    % Replace underscores for display
    name1 = strrep(featureNames{f1}, '_', ' ');
    name2 = strrep(featureNames{f2}, '_', ' ');
    
    xlabel(name1, 'FontWeight', 'bold', 'Interpreter', 'none');
    ylabel(name2, 'FontWeight', 'bold', 'Interpreter', 'none');
    title(sprintf('%s vs %s', name1, name2), 'Interpreter', 'none', 'FontSize', 11);
    
    legend(legendHandles, legendLabels, 'Location', 'best', 'Interpreter', 'none', 'Box', 'on');
    grid on; box on; axis tight;
    hold off;
end