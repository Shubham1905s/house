% Naive Bayes & Decision Tree Classifier for Weather Dataset

% Step 1: Load dataset
filename = 'weather_data.csv';  % Use your actual filename
data = readtable(filename);

% Step 2: Convert all predictors and target to categorical
for i = 1:width(data)
    data.(i) = categorical(data.(i));
end

% Step 3: Separate predictors and target
X = data(:, 1:end-1);              % All columns except last
y = data{:, end};                  % Last column as target
y = categorical(y);                % Ensure target is categorical

% Step 4: Split dataset (70% train, 30% test)
cv = cvpartition(height(data), HoldOut=0.30);

idxTrain = training(cv);
idxTest = test(cv);

XTrain = X(idxTrain, :);
yTrain = y(idxTrain);

XTest = X(idxTest, :);
yTest = y(idxTest);

% Step 5: Train Naive Bayes model
nbModel = fitcnb(XTrain, yTrain);

% Step 6: Predict and evaluate
yPred = predict(nbModel, XTest);
accuracy = sum(yPred == yTest) / numel(yTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Step 6b: Confusion Matrix
confMat = confusionmat(yTest, yPred);
disp('Confusion Matrix:');
disp(confMat);

% Step 6c: Evaluation Metrics (Assuming binary classification)
% Extract TP, FN, FP, TN from Confusion Matrix
if size(confMat,1) == 2 && size(confMat,2) == 2
    TP = confMat(1,1);
    FN = confMat(1,2);
    FP = confMat(2,1);
    TN = confMat(2,2);

    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
    f1 = 2 * (precision * recall) / (precision + recall);

    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 Score: %.2f\n', f1);
else
    disp('Confusion matrix is not 2x2. Metrics only applicable for binary classification.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DECISION TREE

% Train Decision Tree model
dtModel = fitctree(XTrain, yTrain);

% Step 8: Predict and evaluate
yPredDT = predict(dtModel, XTest);

% Step 8a: Accuracy
accuracyDT = sum(yPredDT == yTest) / numel(yTest);
fprintf('\nDecision Tree - Test Accuracy: %.2f%%\n', accuracyDT * 100);

% Step 8b: Confusion Matrix
confMatDT = confusionmat(yTest, yPredDT);
disp('Decision Tree - Confusion Matrix:');
disp(confMatDT);

% Step 8c: Evaluation Metrics (Binary only)
if size(confMatDT,1) == 2
    TP = confMatDT(1,1); FN = confMatDT(1,2);
    FP = confMatDT(2,1); TN = confMatDT(2,2);

    precisionDT = TP / (TP + FP);
    recallDT = TP / (TP + FN);
    f1DT = 2 * (precisionDT * recallDT) / (precisionDT + recallDT);

    fprintf('Decision Tree - Precision: %.2f\n', precisionDT);
    fprintf('Decision Tree - Recall: %.2f\n', recallDT);
    fprintf('Decision Tree - F1 Score: %.2f\n', f1DT);
end

%% ---------------------- Prediction on New Input ----------------------
newData = table(categorical("Sunny"), categorical("Cool"), ...
                categorical("Normal"), categorical("Strong"), ...
                'VariableNames', X.Properties.VariableNames);

newPrediction = predict(nbModel, newData);
 
 newPredictionNB = predict(nbModel, newData);
newPredictionDT = predict(dtModel, newData);

fprintf('\nPrediction for new sample (Naive Bayes): %s\n', string(newPredictionNB));
fprintf('Prediction for new sample (Decision Tree): %s\n', string(newPredictionDT));