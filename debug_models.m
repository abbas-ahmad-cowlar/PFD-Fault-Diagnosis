% Debug script to check if models are available for advanced visualizations

fprintf('=== DIAGNOSTIC: Checking Model Availability ===\n\n');

% Load the saved results
resultFile = 'step5a_results.mat';

% Find the most recent results directory
dirs = dir('PFD_*Results*');
if ~isempty(dirs)
    % Filter to only directories
    dirs = dirs([dirs.isdir]);

    if isempty(dirs)
        fprintf('ERROR: No PFD results directory found!\n');
        return;
    end

    [~, idx] = max([dirs.datenum]);
    latestDir = dirs(idx).name;
    resultFile = fullfile(latestDir, 'step5a_results.mat');
    fprintf('Loading results from: %s\n\n', resultFile);
else
    fprintf('ERROR: No PFD results directory found!\n');
    fprintf('Looking for directories matching: PFD_*Results*\n');
    return;
end

% Check if file exists
if ~exist(resultFile, 'file')
    fprintf('ERROR: %s not found!\n', resultFile);
    return;
end

% Load the workspace
load(resultFile);

fprintf('=== Checking allModelMetrics structure ===\n');

% Check if allModelMetrics exists
if ~exist('allModelMetrics', 'var')
    fprintf('❌ allModelMetrics variable does NOT exist in workspace!\n');
    fprintf('   This means the model evaluation loop did not store the models.\n');
    return;
else
    fprintf('✓ allModelMetrics exists\n\n');
end

% Check what fields exist in allModelMetrics
modelFields = fieldnames(allModelMetrics);
fprintf('Models found in allModelMetrics: %d\n', length(modelFields));
for i = 1:length(modelFields)
    fprintf('  - %s\n', modelFields{i});
end
fprintf('\n');

% Check each model in detail
for i = 1:length(modelFields)
    modelName = modelFields{i};
    fprintf('=== Checking %s ===\n', modelName);

    % Check if model field exists
    if isfield(allModelMetrics.(modelName), 'model')
        fprintf('  ✓ .model field EXISTS\n');
        modelObj = allModelMetrics.(modelName).model;
        fprintf('  ✓ Model type: %s\n', class(modelObj));
    else
        fprintf('  ❌ .model field MISSING!\n');
        fprintf('  Available fields: %s\n', strjoin(fieldnames(allModelMetrics.(modelName)), ', '));
    end

    % Check test accuracy
    if isfield(allModelMetrics.(modelName), 'testAccuracy')
        fprintf('  ✓ Test Accuracy: %.2f%%\n', allModelMetrics.(modelName).testAccuracy);
    end

    fprintf('\n');
end

% Check feature importance data
fprintf('=== Checking featureImportanceData ===\n');
if exist('featureImportanceData', 'var')
    fprintf('✓ featureImportanceData exists\n');
    fiFields = fieldnames(featureImportanceData);
    fprintf('  Models with feature importance: %s\n', strjoin(fiFields, ', '));
else
    fprintf('❌ featureImportanceData NOT found\n');
end
fprintf('\n');

% Check required variables for visualizations
fprintf('=== Checking Required Variables ===\n');
requiredVars = {'X_test_norm', 'Y_test', 'featureNames', 'classNames'};
for i = 1:length(requiredVars)
    varName = requiredVars{i};
    if exist(varName, 'var')
        fprintf('  ✓ %s exists\n', varName);
    else
        fprintf('  ❌ %s MISSING\n', varName);
    end
end

fprintf('\n=== DIAGNOSTIC COMPLETE ===\n');
