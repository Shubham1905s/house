% Load dataset
filename = 'wine_dataset_for_hiearchical_clusterig.csv';
data = readtable(filename, 'VariableNamingRule', 'preserve');

% Normalize data
X = zscore(table2array(data));

%% ----- Agglomerative Hierarchical Clustering -----
% Compute pairwise distances
distances = pdist(X, 'euclidean');

% Linkage using Ward's method
Z = linkage(distances, 'ward');

% Plot dendrogram
figure;
dendrogram(Z);
title('Agglomerative Hierarchical Clustering (Ward)');
xlabel('Sample Index');
ylabel('Distance');

% Cophenetic correlation coefficient
cophCorr = cophenet(Z, distances);
fprintf('Cophenetic Correlation Coefficient: %.4f\n', cophCorr);

% Cluster into a specified number of clusters (e.g., 3)
numClusters = 3;
clusterLabels = cluster(Z, 'maxclust', numClusters);

% Silhouette analysis
figure;
silhouette(X, clusterLabels);
title('Silhouette Plot - Agglomerative Clustering');
avgSilhouette = mean(silhouette(X, clusterLabels));
fprintf('Average Silhouette Score: %.4f\n', avgSilhouette);


%% ----- Divisive Hierarchical Clustering (Recursive K-means) -----
fprintf('\nDivisive Hierarchical Clustering (Simulated using Recursive K-means)\n');

% Recursive divisive clustering function
function divisiveClustering(X, clusterNames, depth)
    if nargin < 2
        clusterNames = {'Root'};
        depth = 0;
    end

    % Stop if the cluster is too small
    if size(X,1) <= 2
        fprintf('%s%s: Leaf Cluster with %d samples\n', repmat(' ', 1, depth), clusterNames{1}, size(X,1));
        return;
    end

    % Apply K-means with K=2 to simulate split
    [idx, ~] = kmeans(X, 2);

    % Display clusters and recurse
    for i = 1:2
        subCluster = X(idx == i, :);
        newName = sprintf('%s-%d', clusterNames{1}, i);
        fprintf('%s%s: %d samples\n', repmat(' ', 1, depth), newName, size(subCluster,1));
        divisiveClustering(subCluster, {newName}, depth + 1);
    end
end

% Call divisive clustering
divisiveClustering(X);
