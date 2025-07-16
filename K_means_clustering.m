% K-Means Algorithm
filename = 'Mall_Customers.csv';

%data = readtable(filename,opts);
data = readtable(filename);

genderNumeric = double(data.Gender == "Male");
%fprintf('Gender numeric : %d\n',genderNumeric);

X = [genderNumeric,data.Age,data.Annual_Income,data.Spending_Score];

% choose the number of clusters
k = 5;
%Apply K-means clustering.
[idx , C] = kmeans(X,k);

% visualize clusters(2D)
figure;
%plot a colored scattering plot.
gscatter(X(:,3),X(:,4),idx);% Annual Income vs spending score

hold on; % tells the MATLAB to keep the plot open so the next plot doesnt overwrite it.
plot(C(:,3),C(:,4),'kX',MarkerSize=15,LineWidth=2);

xlabel('Annual Income(k$)');
ylabel('Spending Score (1-200)');
title('K-Means Clustering with Gender Included');

legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Centroids');

grid on;

new_customer = [0,27,55,60];
%find the nearest centroid 
predicted_cluster = knnsearch(C,new_customer);
%display result
fprintf('The new customer is assigned to Cluster %d.\n',predicted_cluster);

