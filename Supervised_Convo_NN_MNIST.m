% Step 1: Load the data
 trainData = readmatrix('mnist_train.csv');
 testData  = readmatrix('mnist_test.csv');
 % Step 2: Extract features and labels
 XTrain = trainData(:, 2:end);
 YTrain = categorical(trainData(:, 1));
 XTest = testData(:, 2:end);
 YTest = categorical(testData(:, 1));
 % Step 3: Reshape the data to [28, 28, 1, NumImages]
 numTrain = size(XTrain, 1);
 numTest = size(XTest, 1);
 XTrain = reshape(XTrain', 28, 28, 1, numTrain);
 XTrain = permute(XTrain, [2 1 3 4]);  % Ensure correct orientation
 XTest = reshape(XTest', 28, 28, 1, numTest);
 XTest = permute(XTest, [2 1 3 4]);
 % Step 4: Define CNN architecture
 layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
 ];
 % Step 5: Set training options
 options = trainingOptions('adam', MaxEpochs=5, MiniBatchSize=64, ...
    Shuffle='every-epoch', Plots='training-progress', Verbose=false);
% Step 6: Train the network
 net = trainNetwork(XTrain, YTrain, layers, options);
 % Step 7: Evaluate on test data
 YPred = classify(net, XTest);
 accuracy = sum(YPred == YTest) / numel(YTest);
 fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);
 % Step 8 (Optional): Show confusion matrix
 confusionchart(YTest, YPred);