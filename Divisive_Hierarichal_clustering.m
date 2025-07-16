%Divisive Algorithm

% MATLAB does not have built-in divisive hierarchical clustering 
% like it does for agglomerative. 
% But we can simulate it using a recursive K-means strategy.

function divisiveClustering(X, clusterNames, depth)
    if nargin < 2
        clusterNames = {'Root'};
        depth = 0;
    end

    % Base case: stop if only one point or cluster is too small
    if size(X,1) <= 2
        fprintf('%s: Leaf Cluster with %d samples\n', clusterNames{1}, size(X,1));
        return;
    end

    % Apply K-means with K=2 to simulate a split
    [idx, ~] = kmeans(X, 2);

    % Display the cluster names
    for i = 1:2
        subCluster = X(idx == i, :);
        newName = sprintf('%s-%d', clusterNames{1}, i);
        fprintf('%s: %d samples\n', repmat('  ', 1, depth), newName, size(subCluster,1));
        % Recursive call
        divisiveClustering(subCluster, {newName}, depth + 1);
    end
end

% Usage:
% Load dataset
filename = 'wine_dataset_for_hiearchical_clusterig.csv';
data = readtable(filename);
X = zscore(table2array(data));

% Start divisive clustering
fprintf('Divisive Hierarchical Clustering (Recursive K-Means)\n');
divisiveClustering(X);