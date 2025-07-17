 % Step 0: Reset the Command Window and Worksace
 clear; clc;
 % Step 1: Load dataset
 filename = 'weather_data.csv';  % Replace with actual path if needed
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
 cv = cvpartition(height(data), 'HoldOut', 0.30);
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
 % Step 7: Confusion Matrix Chart (Graphical)
 cmChart = confusionchart(yTest, yPred);
 cmChart.Title = 'Naive Bayes Confusion Matrix';
 % Step 8: Predict on new input
 % Example: Outlook=Sunny, Temperature=Cool, Humidity=Normal, Wind=Strong
 newData = table(categorical("Sunny"), categorical("Cool"), ("Normal"), categorical("Strong"), ...
                'VariableNames', X.Properties.VariableNames);
 newPrediction = predict(nbModel, newData);
 fprintf('Prediction for new sample: %s\n', string(newPrediction));