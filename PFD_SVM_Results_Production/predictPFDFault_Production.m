function result = predictPFDFault_Production(signalData, varargin)
% PREDICTPFDFAULT_PRODUCTION  Production inference function for PFD fault diagnosis
%
% SYNTAX:
%   result = predictPFDFault_Production(signalData)
%   result = predictPFDFault_Production(signalData, 'ModelPath', path)
%
% INPUTS:
%   signalData - Vibration signal vector (time series)
%   ModelPath  - (Optional) Path to model file
%
% OUTPUTS:
%   result - Structure containing:
%     .fault         - Predicted fault class
%     .confidence    - Confidence score (0-100%)
%     .probabilities - Probabilities for all classes
%     .top3_faults   - Top 3 predicted faults
%     .top3_probs    - Top 3 probabilities
%     .warning       - Warning message if confidence low
%
% EXAMPLE:
%   load('signal_data.mat', 'x');
%   result = predictPFDFault_Production(x);
%   fprintf('Detected: %s (Confidence: %.1f%%)\n', result.fault, result.confidence);
%
% Generated: 26-Nov-2025 21:57:40
% Model: Random Forest
% Version: Production v2.0

    % Parse inputs
    p = inputParser;
    addRequired(p, 'signalData', @isvector);
    addParameter(p, 'ModelPath', 'PFD_SVM_Results_Production\Best_PFD_Model_Production.mat', @ischar);
    parse(p, signalData, varargin{:});
    
    modelPath = p.Results.ModelPath;
    x = signalData(:);  % Ensure column vector
    
    % Load model package
    if ~exist(modelPath, 'file')
        error('Model file not found: %s', modelPath);
    end
    
    load(modelPath, 'modelPackage');
    
    % Extract features
    fs = 20480;  % Sampling frequency from training
    
    try
        featValues = extractFeaturesForInference(x, fs, modelPackage);
    catch ME
        error('Feature extraction failed: %s', ME.message);
    end
    
    % Normalize features
    featNorm = (featValues - modelPackage.normalization.mu) ./ modelPackage.normalization.sigma;
    
    % Predict
    [predictedClass, scores] = predict(modelPackage.model, featNorm);
    
    % Handle score format
    if size(scores, 2) == 1
        scores = [1 - scores, scores];
    end
    
    % Apply calibration if enabled
    if modelPackage.calibration.enabled
        scoresCalibrated = applyCalibratio(scores, modelPackage);
        scores = scoresCalibrated;
    end
    
    % Build result structure
    result = struct();
    result.fault = char(predictedClass);
    
    [~, predIdx] = ismember(predictedClass, modelPackage.classes.names);
    result.confidence = 100 * scores(predIdx);
    
    result.probabilities = array2table(scores, ...
        'VariableNames', modelPackage.classes.names);
    
    % Top 3 predictions
    [sortedProbs, sortIdx] = sort(scores, 'descend');
    result.top3_faults = modelPackage.classes.names(sortIdx(1:min(3, end)));
    result.top3_probs = sortedProbs(1:min(3, end));
    
    % Confidence warning
    if result.confidence < 70
        result.warning = 'Low confidence - verify with domain expert';
    elseif result.confidence < 85
        result.warning = 'Moderate confidence - recommend manual review';
    else
        result.warning = '';
    end
    
end

function featValues = extractFeaturesForInference(x, fs, modelPackage)
    % Simplified feature extraction for inference
    % For production use, copy full extractFeatures function from training pipeline
    
    numFeatures = modelPackage.features.count;
    featValues = zeros(1, numFeatures);
    
    % TODO: Implement full feature extraction matching training pipeline
    % This is a placeholder - copy extractFeatures function from training code
    
    error('Feature extraction not yet implemented. Copy extractFeatures function from training pipeline.');
end

function scoresCalibrated = applyCalibration(scores, modelPackage)
    % Apply probability calibration
    
    scoresCalibrated = scores;
    calibModels = modelPackage.calibration.models;
    
    if strcmp(modelPackage.calibration.method, 'sigmoid')
        for c = 1:length(calibModels)
            if ~isempty(calibModels{c})
                scoresCalibrated(c) = predict(calibModels{c}, scores(c));
                scoresCalibrated(c) = max(0, min(1, scoresCalibrated(c)));
            end
        end
    end
    
    % Normalize to sum to 1
    scoresCalibrated = scoresCalibrated / sum(scoresCalibrated);
end
