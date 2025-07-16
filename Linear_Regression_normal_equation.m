 % Step 1: Load Dataset (assume CSV with two columns: Feature, Target)
 data = readmatrix('swedish_insurance.csv');   
%data = readmatrix('swedishinsurance2.xls') ;
 X = data(:, 1);   % Feature column
 y = data(:, 2);   % Target/output column
 m = length(y);    
% Number of training examples
 % Step 2: Add intercept term
 X_b = [ones(m, 1), X];   % Augmented feature matrix
 % Step 3: Compute theta using Normal Equation
 theta = inv(X_b' * X_b) * X_b' * y;
 fprintf('Learned parameters:\n');
 disp(theta);
 fprintf('Estimated Slope: %.4f\n', theta(2));
 fprintf('Estimated Intercept: %.4f\n', theta(1));
 % Step 4: Predict values
 y_pred = X_b * theta;
 % Step 5: Plot origial data and regression line
 % Blue circles (`bo`) for original data
 % Red line (`r-`) for the fitted regression line
 figure;
 plot(X, y, 'bo', 'MarkerSize', 7);      
% Original data
 hold on;
 plot(X, y_pred, 'r-', 'LineWidth', 2);  % Regression line
 % Plot prediction at x = 10
 X_new = [1, 10];
 Y_new_pred = X_new * theta;
 plot(X_new, Y_new_pred, 'gs', 'MarkerSize', 12);
 xlabel('Feature');
 ylabel('Target');
 title('Linear Regression Fit');
 legend('Training data', 'Linear regression');
 grid on;
 Y_new_pred = X_new * theta;
 plot(X_new, Y_new_pred, 'gs', 'MarkerSize', 12);
 xlabel('Feature');
 ylabel('Target');
 title('Linear Regression Fit');
 legend('Training data', 'Linear regression');
 grid on;
