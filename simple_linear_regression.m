% Linear Regression using fitlm()
 clear; clc;
 % Step 1: Load dataset (assume two columns: Feature, Target)
 data = readmatrix('swedish_insurance.csv');
 % Separate input feature and target output
 X = data(:, 1);   % Feature (e.g., number of claims)
 y = data(:, 2);   % Target (e.g., insurance payment)
 % Step 2: Fit a linear regression model using fitlm()
 model = fitlm(X, y);
 % Display summary of the model
 disp(model);
 % Step 3: Plot the data and regression line
 figure;
 plot(model);   % Automatically plots data and fitted line
 hold on;
 xlabel('Feature');
 ylabel('Target');
 title('Linear Regression using fitlm()');
 grid on;
 % Step 4: Predict for a new input (e.g., x = 6)
 x_new = 10;
 y_pred = predict(model, x_new);
 fprintf('\nPrediction for x = %.2f: y = %.2f\n', x_new, y_pred);
 plot(x_new,y_pred, 'gs', 'MarkerSize', 12);
