%Naive bayes algorithm
% Step 1: Import dataset (assume CSV with headers)
filename = 'weather_data.csv';  % Replace with your file name
data = readtable(filename);

for i = 1:width(data)
    data.(i) = categorical(data.(i));
end

X = data (:,1:end-1);
y = data{:,end};
y = categorical(y);

cv = cvpartition(height(data),'HoldOut',0.40);

idxTrain = training(cv);
idxTest = test(cv);

XTrain = X(idxTrain,:);
yTrain = y(idxTrain);

XTest = X(idxTest,:);
yTest = y(idxTest);

nbModel = fitcnb(XTrain,yTrain);
yPred = predict(nbModel,XTest);
accuracy = sum(yPred == yTest)/ numel(yTest);
fprintf('Test Accuracy:%.2f%%\n',accuracy * 100);

newData = table(categorical("Sunny"), categorical("Cool"), categorical("Normal"),categorical("Strong"),'VariableNames', X.Properties.VariableNames);
newPrediction = predict(nbModel,newData);
fprintf('Prediction for new sample : %s\n',string(newPrediction));

