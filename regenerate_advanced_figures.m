%% REGENERATE ADVANCED VISUALIZATIONS (Fig16-18)
% This script loads trained models and regenerates only the advanced
% visualizations without retraining

fprintf('\n========================================================================\n');
fprintf('REGENERATING ADVANCED VISUALIZATIONS (Fig16-18)\n');
fprintf('========================================================================\n\n');

%% STEP 1: Find and load results
fprintf('Step 1: Loading saved results...\n');

dirs = dir('PFD_*Results*');
if ~isempty(dirs)
    dirs = dirs([dirs.isdir]);
    [~, idx] = max([dirs.datenum]);
    latestDir = dirs(idx).name;
    resultFile = fullfile(latestDir, 'step5a_results.mat');
    fprintf('  Loading from: %s\n', resultFile);
else
    error('No PFD results directory found!');
end

if ~exist(resultFile, 'file')
    error('Results file not found: %s', resultFile);
end

% Load workspace
load(resultFile);
fprintf('  ✓ Results loaded successfully\n\n');

%% STEP 2: Set output directory
CONFIG = struct();
CONFIG.outputDir = latestDir;
fprintf('Step 2: Output directory set to: %s\n\n', CONFIG.outputDir);

%% STEP 3: Regenerate feature importance data
fprintf('Step 3: Regenerating feature importance data...\n');

featureImportanceData = struct();

% Random Forest
if isfield(modelResults, 'RandomForest') && isfield(modelResults.RandomForest, 'model')
    fprintf('  - Random Forest...\n');
    try
        rfModel = modelResults.RandomForest.model;
        featureImportanceData.RandomForest = oobPermutedPredictorImportance(rfModel);
        fprintf('    ✓ RF importance computed\n');
    catch ME
        fprintf('    ⚠️  Error: %s\n', ME.message);
        featureImportanceData.RandomForest = ones(length(featureNames), 1) / length(featureNames);
    end
end

% SVM (permutation-based)
if isfield(modelResults, 'SVM') && isfield(modelResults.SVM, 'model')
    fprintf('  - SVM (permutation-based)...\n');
    try
        svmModel = modelResults.SVM.model;
        [Y_test_svm_pred, ~] = predict(svmModel, X_test_norm);
        baselineAcc = sum(Y_test_svm_pred == Y_test) / length(Y_test);

        importance_svm = zeros(length(featureNames), 1);
        for f = 1:length(featureNames)
            X_test_permuted = X_test_norm;
            X_test_permuted(:, f) = X_test_permuted(randperm(size(X_test_permuted, 1)), f);
            [Y_test_perm_pred, ~] = predict(svmModel, X_test_permuted);
            permutedAcc = sum(Y_test_perm_pred == Y_test) / length(Y_test);
            importance_svm(f) = max(0, baselineAcc - permutedAcc);
        end

        if sum(importance_svm) > 0
            featureImportanceData.SVM = importance_svm / sum(importance_svm);
        else
            featureImportanceData.SVM = ones(length(featureNames), 1) / length(featureNames);
        end
        fprintf('    ✓ SVM importance computed\n');
    catch ME
        fprintf('    ⚠️  Error: %s\n', ME.message);
        featureImportanceData.SVM = ones(length(featureNames), 1) / length(featureNames);
    end
end

% Neural Network (permutation-based)
if isfield(modelResults, 'NeuralNetwork') && isfield(modelResults.NeuralNetwork, 'model')
    fprintf('  - Neural Network (permutation-based)...\n');
    try
        nnModel = modelResults.NeuralNetwork.model;
        [Y_test_nn_pred, ~] = predict(nnModel, X_test_norm);
        baselineAcc = sum(Y_test_nn_pred == Y_test) / length(Y_test);

        importance_nn = zeros(length(featureNames), 1);
        for f = 1:length(featureNames)
            X_test_permuted = X_test_norm;
            X_test_permuted(:, f) = X_test_permuted(randperm(size(X_test_permuted, 1)), f);
            [Y_test_perm_pred, ~] = predict(nnModel, X_test_permuted);
            permutedAcc = sum(Y_test_perm_pred == Y_test) / length(Y_test);
            importance_nn(f) = max(0, baselineAcc - permutedAcc);
        end

        if sum(importance_nn) > 0
            featureImportanceData.NeuralNetwork = importance_nn / sum(importance_nn);
        else
            featureImportanceData.NeuralNetwork = ones(length(featureNames), 1) / length(featureNames);
        end
        fprintf('    ✓ NN importance computed\n');
    catch ME
        fprintf('    ⚠️  Error: %s\n', ME.message);
        featureImportanceData.NeuralNetwork = ones(length(featureNames), 1) / length(featureNames);
    end
end

fprintf('  ✓ Feature importance data ready\n');
fprintf('    Fields: %s\n\n', strjoin(fieldnames(featureImportanceData), ', '));

%% STEP 4: Check allModelMetrics
fprintf('Step 4: Checking allModelMetrics structure...\n');

if ~exist('allModelMetrics', 'var')
    fprintf('  ❌ allModelMetrics does not exist!\n');
    fprintf('  Creating from modelResults...\n');

    % This shouldn't happen if the updated pipeline ran, but handle it
    error('allModelMetrics not found. Please re-run the full pipeline with the updated code.');
end

fprintf('  ✓ allModelMetrics exists\n');
fprintf('  Models available: %s\n', strjoin(fieldnames(allModelMetrics), ', '));

% Check each model
for modelName = fieldnames(allModelMetrics)'
    modelName = modelName{1};
    if isfield(allModelMetrics.(modelName), 'model')
        fprintf('  ✓ %s has .model field (type: %s)\n', modelName, class(allModelMetrics.(modelName).model));
    else
        fprintf('  ❌ %s MISSING .model field!\n', modelName);
    end
end
fprintf('\n');

%% STEP 5: Generate Fig16 (SVM Decision Boundaries)
fprintf('========================================================================\n');
fprintf('Step 5: Generating Fig16 - SVM Decision Boundaries\n');
fprintf('========================================================================\n');

try
    fprintf('Checking prerequisites...\n');

    % Check 1: allModelMetrics.SVM exists
    if ~isfield(allModelMetrics, 'SVM')
        fprintf('  ❌ allModelMetrics.SVM does not exist!\n');
        error('SVM not in allModelMetrics');
    end
    fprintf('  ✓ allModelMetrics.SVM exists\n');

    % Check 2: model field exists
    if ~isfield(allModelMetrics.SVM, 'model')
        fprintf('  ❌ allModelMetrics.SVM.model does not exist!\n');
        error('SVM model field missing');
    end
    fprintf('  ✓ allModelMetrics.SVM.model exists\n');

    % Check 3: featureImportanceData exists
    if ~exist('featureImportanceData', 'var')
        fprintf('  ❌ featureImportanceData variable does not exist!\n');
        error('featureImportanceData missing');
    end
    fprintf('  ✓ featureImportanceData variable exists\n');

    % Check 4: SVM field in featureImportanceData
    if ~isfield(featureImportanceData, 'SVM')
        fprintf('  ❌ featureImportanceData.SVM does not exist!\n');
        error('SVM importance missing');
    end
    fprintf('  ✓ featureImportanceData.SVM exists\n');

    svmModel = allModelMetrics.SVM.model;
    [sortedImp, sortedIdx] = sort(featureImportanceData.SVM, 'descend');

    fprintf('  Top 4 features:\n');
    for i = 1:min(4, length(sortedIdx))
        fprintf('    %d. %s (importance: %.4f)\n', i, featureNames{sortedIdx(i)}, sortedImp(i));
    end

    topFeatures = sortedIdx(1:min(4, length(sortedIdx)));

    if length(topFeatures) >= 4
        pair1 = topFeatures(1:2);
        pair2 = topFeatures(3:4);

        fprintf('\n  Creating figure...\n');
        fig16 = figure('Position', [100, 100, 1200, 600], 'Visible', 'off');

        % Pair 1
        fprintf('  - Plotting pair 1: %s vs %s\n', featureNames{pair1(1)}, featureNames{pair1(2)});
        subplot(1, 2, 1);
        plotSVMDecisionBoundary(svmModel, X_test_norm, Y_test, pair1, featureNames, classNames);
        title(sprintf('SVM Decision Boundary: %s vs %s', ...
            strrep(featureNames{pair1(1)}, '_', ' '), ...
            strrep(featureNames{pair1(2)}, '_', ' ')), ...
            'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

        % Pair 2
        fprintf('  - Plotting pair 2: %s vs %s\n', featureNames{pair2(1)}, featureNames{pair2(2)});
        subplot(1, 2, 2);
        plotSVMDecisionBoundary(svmModel, X_test_norm, Y_test, pair2, featureNames, classNames);
        title(sprintf('SVM Decision Boundary: %s vs %s', ...
            strrep(featureNames{pair2(1)}, '_', ' '), ...
            strrep(featureNames{pair2(2)}, '_', ' ')), ...
            'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');

        sgtitle('Fig 16: SVM Decision Boundaries in Top Feature Pairs', ...
            'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

        % Save
        saveas(fig16, fullfile(CONFIG.outputDir, 'Fig16_SVM_Decision_Boundaries.png'));
        close(fig16);
        fprintf('  ✓✓✓ SAVED: Fig16_SVM_Decision_Boundaries.png\n\n');
    else
        fprintf('  ⚠️  Insufficient features (need 4, have %d)\n\n', length(topFeatures));
    end

catch ME
    fprintf('  ❌❌❌ ERROR generating Fig16:\n');
    fprintf('     Message: %s\n', ME.message);
    fprintf('     Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('       %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
end

%% STEP 6: Generate Fig17 (Random Forest Decision Paths)
fprintf('========================================================================\n');
fprintf('Step 6: Generating Fig17 - Random Forest Decision Paths\n');
fprintf('========================================================================\n');

try
    fprintf('Checking prerequisites...\n');

    if ~isfield(allModelMetrics, 'RandomForest')
        fprintf('  ❌ allModelMetrics.RandomForest does not exist!\n');
        error('RandomForest not in allModelMetrics');
    end
    fprintf('  ✓ allModelMetrics.RandomForest exists\n');

    if ~isfield(allModelMetrics.RandomForest, 'model')
        fprintf('  ❌ allModelMetrics.RandomForest.model does not exist!\n');
        error('RandomForest model field missing');
    end
    fprintf('  ✓ allModelMetrics.RandomForest.model exists\n');

    rfModel = allModelMetrics.RandomForest.model;
    nTrees = rfModel.NumTrained;  % For ClassificationBaggedEnsemble
    fprintf('  Random Forest has %d trees\n', nTrees);

    fprintf('\n  Analyzing tree structures...\n');
    splitFeatures = cell(nTrees, 1);
    splitDepths = cell(nTrees, 1);

    for t = 1:nTrees
        if mod(t, 20) == 0
            fprintf('    Processing tree %d/%d\n', t, nTrees);
        end
        tree = rfModel.Trained{t};  % For ClassificationBaggedEnsemble

        % Get cut predictors (handle both cell and numeric arrays)
        cutPredictors = tree.CutPredictor;
        if iscell(cutPredictors)
            % Convert cell array to numeric array
            % Each cell should contain either empty or a feature index
            cutPredictors_numeric = zeros(length(cutPredictors), 1);
            for i = 1:length(cutPredictors)
                if isempty(cutPredictors{i})
                    cutPredictors_numeric(i) = 0;
                elseif isnumeric(cutPredictors{i})
                    cutPredictors_numeric(i) = cutPredictors{i}(1);  % Take first element if array
                else
                    cutPredictors_numeric(i) = 0;
                end
            end
            cutPredictors = cutPredictors_numeric;
        end

        nodeDepths = zeros(length(cutPredictors), 1);
        for n = 1:length(cutPredictors)
            nodeDepths(n) = getNodeDepth(tree, n);
        end

        decisionNodes = cutPredictors > 0;
        splitFeatures{t} = cutPredictors(decisionNodes);
        splitDepths{t} = nodeDepths(decisionNodes);
    end

    fprintf('  ✓ Tree analysis complete\n');

    % Aggregate
    allSplitFeatures = cat(1, splitFeatures{:});
    allSplitDepths = cat(1, splitDepths{:});
    fprintf('  Total splits analyzed: %d\n', length(allSplitFeatures));

    splitCounts = zeros(length(featureNames), 1);
    avgDepthPerFeature = zeros(length(featureNames), 1);

    for f = 1:length(featureNames)
        featureMask = allSplitFeatures == f;
        splitCounts(f) = sum(featureMask);
        if splitCounts(f) > 0
            avgDepthPerFeature(f) = mean(allSplitDepths(featureMask));
        end
    end

    splitFrequency = splitCounts / sum(splitCounts);

    fprintf('\n  Creating figure...\n');
    fig17 = figure('Position', [100, 100, 1400, 500], 'Visible', 'off');

    % Panel 1
    subplot(1, 3, 1);
    [sortedFreq, sortedIdx] = sort(splitFrequency, 'descend');
    barh(sortedFreq);
    set(gca, 'YTick', 1:length(featureNames), ...
        'YTickLabel', featureNames(sortedIdx), ...
        'TickLabelInterpreter', 'none', 'FontSize', 9);
    xlabel('Split Frequency', 'FontSize', 11, 'Interpreter', 'none');
    ylabel('Feature', 'FontSize', 11, 'Interpreter', 'none');
    title('Split Frequency Across Forest', 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    grid on;

    % Panel 2
    subplot(1, 3, 2);
    validFeatures = avgDepthPerFeature > 0;
    [sortedDepth, sortedIdx2] = sort(avgDepthPerFeature(validFeatures), 'descend');
    validFeatureNames = featureNames(validFeatures);
    barh(sortedDepth);
    set(gca, 'YTick', 1:length(sortedDepth), ...
        'YTickLabel', validFeatureNames(sortedIdx2(1:length(sortedDepth))), ...
        'TickLabelInterpreter', 'none', 'FontSize', 9);
    xlabel('Average Depth', 'FontSize', 11, 'Interpreter', 'none');
    ylabel('Feature', 'FontSize', 11, 'Interpreter', 'none');
    title('Mean Split Depth per Feature', 'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    grid on;

    % Panel 3
    subplot(1, 3, 3);
    histogram(allSplitDepths, 'BinWidth', 1, 'FaceColor', [0.2, 0.7, 0.3], 'EdgeColor', 'k');
    xlabel('Split Depth', 'FontSize', 11, 'Interpreter', 'none');
    ylabel('Frequency', 'FontSize', 11, 'Interpreter', 'none');
    title(sprintf('Split Depth Distribution\n(%d trees, %d total splits)', ...
        nTrees, length(allSplitDepths)), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Interpreter', 'none');
    grid on;

    sgtitle('Fig 17: Random Forest Decision Path Analysis', ...
        'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Save
    saveas(fig17, fullfile(CONFIG.outputDir, 'Fig17_RandomForest_Decision_Paths.png'));
    close(fig17);
    fprintf('  ✓✓✓ SAVED: Fig17_RandomForest_Decision_Paths.png\n\n');

catch ME
    fprintf('  ❌❌❌ ERROR generating Fig17:\n');
    fprintf('     Message: %s\n', ME.message);
    fprintf('     Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('       %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
end

%% STEP 7: Generate Fig18 (Neural Network Architecture)
fprintf('========================================================================\n');
fprintf('Step 7: Generating Fig18 - Neural Network Architecture\n');
fprintf('========================================================================\n');

try
    fprintf('Checking prerequisites...\n');

    if ~isfield(allModelMetrics, 'NeuralNetwork')
        fprintf('  ❌ allModelMetrics.NeuralNetwork does not exist!\n');
        error('NeuralNetwork not in allModelMetrics');
    end
    fprintf('  ✓ allModelMetrics.NeuralNetwork exists\n');

    if ~isfield(allModelMetrics.NeuralNetwork, 'model')
        fprintf('  ❌ allModelMetrics.NeuralNetwork.model does not exist!\n');
        error('NeuralNetwork model field missing');
    end
    fprintf('  ✓ allModelMetrics.NeuralNetwork.model exists\n');

    nnModel = allModelMetrics.NeuralNetwork.model;

    fprintf('\n  Creating figure...\n');
    fig18 = figure('Position', [100, 100, 1200, 800], 'Visible', 'off');

    % Panel 1: Architecture
    subplot(2, 1, 1);
    axis off;
    hold on;

    % Extract architecture from ClassificationNeuralNetwork
    try
        layerSizes = nnModel.LayerSizes;
        nFeatures = size(X_test_norm, 2);
        nClasses = length(classNames);

        % Build layer array: [input, hidden layers, output]
        allLayers = [nFeatures, layerSizes, nClasses];
        nLayers = length(allLayers);

        fprintf('  Network structure: %s\n', mat2str(allLayers));
    catch
        % Fallback if LayerSizes not available
        allLayers = [size(X_test_norm, 2), 50, length(classNames)];
        nLayers = 3;
    end

    xPositions = linspace(0.1, 0.9, nLayers);
    yCenter = 0.5;

    % Draw layers
    for i = 1:nLayers
        if i == 1
            layerName = sprintf('Input\n%d', allLayers(i));
            color = [0.9, 0.9, 1];
        elseif i == nLayers
            layerName = sprintf('Output\n%d', allLayers(i));
            color = [1, 0.9, 0.9];
        else
            layerName = sprintf('Hidden\n%d', allLayers(i));
            color = [0.8, 0.9, 1];
        end

        % Draw box
        rectangle('Position', [xPositions(i) - 0.05, yCenter - 0.15, 0.1, 0.3], ...
            'FaceColor', color, 'EdgeColor', 'k', 'LineWidth', 1.5);

        % Add text
        text(xPositions(i), yCenter, layerName, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 9, 'FontWeight', 'bold', 'Interpreter', 'none');

        % Draw arrows between layers
        if i < nLayers
            annotation('arrow', [xPositions(i) + 0.05, xPositions(i+1) - 0.05], ...
                [yCenter, yCenter], 'LineWidth', 2, 'HeadStyle', 'cback1', ...
                'HeadLength', 8, 'HeadWidth', 8);
        end
    end

    xlim([0, 1]);
    ylim([0, 1]);

    % Add activation function info
    try
        activation = nnModel.Activations;
        text(0.5, 0.15, sprintf('Activation: %s', activation), ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'FontSize', 10, 'Interpreter', 'none');
    catch
        % Ignore if not available
    end

    title('Neural Network Architecture', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Panel 2: Activations
    subplot(2, 1, 2);

    try
        fprintf('  Computing output activations...\n');
        activations = predict(nnModel, X_test_norm);

        avgActivations = zeros(length(classNames), length(classNames));
        for c = 1:length(classNames)
            classMask = Y_test == c;
            if sum(classMask) > 0
                avgActivations(c, :) = mean(activations(classMask, :), 1);
            end
        end

        imagesc(avgActivations);
        colormap(jet);
        colorbar;

        set(gca, 'XTick', 1:length(classNames), 'XTickLabel', classNames, ...
            'XTickLabelRotation', 45, 'YTick', 1:length(classNames), ...
            'YTickLabel', classNames, 'TickLabelInterpreter', 'none', 'FontSize', 9);

        xlabel('Predicted Class (Output Neuron)', 'FontSize', 11, 'Interpreter', 'none');
        ylabel('True Class', 'FontSize', 11, 'Interpreter', 'none');
        title('Average Output Layer Activations per Class', 'FontSize', 12, ...
            'FontWeight', 'bold', 'Interpreter', 'none');

        for c = 1:length(classNames)
            text(c, c, sprintf('%.2f', avgActivations(c, c)), ...
                'HorizontalAlignment', 'center', 'Color', 'white', ...
                'FontSize', 9, 'FontWeight', 'bold');
        end
    catch ME2
        fprintf('    ⚠️  Activation computation failed: %s\n', ME2.message);
        text(0.5, 0.5, 'Activation computation not supported', ...
            'Units', 'normalized', 'HorizontalAlignment', 'center', ...
            'FontSize', 12, 'Interpreter', 'none');
        axis off;
    end

    sgtitle('Fig 18: Neural Network Architecture & Output Activations', ...
        'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

    % Save
    saveas(fig18, fullfile(CONFIG.outputDir, 'Fig18_NeuralNetwork_Architecture.png'));
    close(fig18);
    fprintf('  ✓✓✓ SAVED: Fig18_NeuralNetwork_Architecture.png\n\n');

catch ME
    fprintf('  ❌❌❌ ERROR generating Fig18:\n');
    fprintf('     Message: %s\n', ME.message);
    fprintf('     Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('       %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    fprintf('\n');
end

%% COMPLETION
fprintf('========================================================================\n');
fprintf('✅ REGENERATION COMPLETE\n');
fprintf('========================================================================\n');
fprintf('Check %s for the new figures:\n', CONFIG.outputDir);
fprintf('  - Fig16_SVM_Decision_Boundaries.png\n');
fprintf('  - Fig17_RandomForest_Decision_Paths.png\n');
fprintf('  - Fig18_NeuralNetwork_Architecture.png\n');
fprintf('========================================================================\n\n');

%% HELPER FUNCTIONS

function plotSVMDecisionBoundary(svmModel, X_data, Y_data, featurePair, featureNames, classNames)
    % Plot SVM decision boundary for a specific pair of features
    try
        % Extract the two features
        X_pair = X_data(:, featurePair);

        % Create grid for decision boundary
        x1_range = linspace(min(X_pair(:,1)) - 0.5, max(X_pair(:,1)) + 0.5, 200);
        x2_range = linspace(min(X_pair(:,2)) - 0.5, max(X_pair(:,2)) + 0.5, 200);
        [X1_grid, X2_grid] = meshgrid(x1_range, x2_range);

        % Create full feature matrix for prediction (use mean for other features)
        nFeatures = size(X_data, 2);
        X_grid_full = zeros(numel(X1_grid), nFeatures);

        % Fill in the two features of interest
        X_grid_full(:, featurePair(1)) = X1_grid(:);
        X_grid_full(:, featurePair(2)) = X2_grid(:);

        % Fill other features with their mean values
        for f = 1:nFeatures
            if ~ismember(f, featurePair)
                X_grid_full(:, f) = mean(X_data(:, f));
            end
        end

        % Predict on grid
        Z_grid = predict(svmModel, X_grid_full);
        Z_grid = reshape(Z_grid, size(X1_grid));

        % Plot decision regions
        contourf(X1_grid, X2_grid, Z_grid, length(classNames), 'LineStyle', 'none');
        alpha(0.3); % Make background semi-transparent
        hold on;

        % Plot data points
        colormap(gca, jet(length(classNames)));
        scatter(X_pair(:,1), X_pair(:,2), 50, Y_data, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1);

        % Format
        xlabel(strrep(featureNames{featurePair(1)}, '_', ' '), ...
            'FontSize', 11, 'Interpreter', 'none');
        ylabel(strrep(featureNames{featurePair(2)}, '_', ' '), ...
            'FontSize', 11, 'Interpreter', 'none');

        % Add colorbar with class labels
        cb = colorbar;
        cb.Ticks = 1:length(classNames);
        cb.TickLabels = classNames;
        cb.TickLabelInterpreter = 'none';

        grid on;
        hold off;

    catch ME
        % If prediction fails, just plot the data points
        scatter(X_pair(:,1), X_pair(:,2), 50, Y_data, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1);
        xlabel(strrep(featureNames{featurePair(1)}, '_', ' '), ...
            'FontSize', 11, 'Interpreter', 'none');
        ylabel(strrep(featureNames{featurePair(2)}, '_', ' '), ...
            'FontSize', 11, 'Interpreter', 'none');
        title(sprintf('Error: %s', ME.message), 'FontSize', 10, 'Color', 'r');
    end
end

function depth = getNodeDepth(tree, nodeIdx)
    % Calculate depth of a node in a decision tree
    % Depth = 0 for root, increases by 1 for each level down

    depth = 0;
    currentNode = nodeIdx;

    % Walk up the tree to root by finding parent nodes
    while currentNode ~= 1  % Node 1 is the root
        % Find parent: node i is a child of floor((i+1)/2)
        % This works for MATLAB's compact tree representation
        parentNode = floor(currentNode / 2);

        if parentNode < 1
            break;  % Safety check
        end

        depth = depth + 1;
        currentNode = parentNode;

        % Safety limit to prevent infinite loops
        if depth > 100
            break;
        end
    end
end

function result = iif(condition, trueVal, falseVal)
    % Inline if-else function
    if condition
        result = trueVal;
    else
        result = falseVal;
    end
end
